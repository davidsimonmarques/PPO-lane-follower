"""Wrapper de ambiente CARLA para lane follower."""

import random
import time
from typing import Any, Dict, Optional, Tuple

import carla
import numpy as np

import cv2

class CarlaEnvironment:
    def __init__(self, config: Dict):
        self.config = config
        self.verbose = bool(self.config.get("verbose", False))
        self.max_lane_offset = float(self.config.get("max_lane_offset", 2.0))
        self.waypoints_drawn = False
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.map: Optional[carla.Map] = None
        self.vehicle: Optional[carla.Actor] = None
        self.blueprint_library: Optional[carla.BlueprintLibrary] = None
        self.camera: Optional[carla.Sensor] = None
        self.collision_sensor: Optional[carla.Sensor] = None
        self.spectator: Optional[carla.Actor] = None
        self.driver_image: Optional[np.ndarray] = None
        self.collision_detected = False
        self.distance_traveled = 0.0  # Distância percorrida em metros
        self.previous_location: Optional[carla.Location] = None
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Inicializa o cliente CARLA e o mundo."""
        host = self.config.get("host", "127.0.0.1")
        port = self.config.get("port", 2000)
        timeout = self.config.get("timeout", 60.0)

        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)

        if self.config.get("map_name"):
            self.client.load_world(self.config["map_name"])

        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        settings = self.world.get_settings()
        settings.synchronous_mode = self.config.get("synchronous", True)
        settings.fixed_delta_seconds = self.config.get("fixed_delta_seconds", 0.05)
        settings.no_rendering_mode = self.config.get("no_rendering", False)
        self.world.apply_settings(settings)

        # 🚀 OTIMIZAÇÃO: Limitação de FPS para reduzir carga de CPU
        if self.config.get("max_fps"):
            settings.max_substep_delta_time = 1.0 / self.config["max_fps"]
            settings.max_substeps = 1
            self.world.apply_settings(settings)

        self._spawn_vehicle()
        self._setup_sensors()
        if self.config.get("render", False):
            self._draw_waypoints()

    def _get_spawn_point(self) -> carla.Transform:
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("Nenhum ponto de spawn disponível no mapa CARLA.")
        return random.choice(spawn_points)

    def _spawn_vehicle(self) -> None:
        self._destroy_sensors()
        if self.vehicle is not None:
            self._safe_destroy(self.vehicle)
            self.vehicle = None

        self._cleanup_previous_hero_vehicles()

        filter_name = self.config.get("vehicle_filter", "vehicle.tesla.model3")
        blueprint = self.blueprint_library.filter(filter_name)[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")

        # Try multiple spawn points if collision occurs
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)

        for spawn_point in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
                self.vehicle.set_autopilot(False)
                break
            except RuntimeError as e:
                if "collision" in str(e).lower():
                    continue
                else:
                    raise e
        else:
            raise RuntimeError("Failed to spawn vehicle: all spawn points are blocked")

    def reset(self) -> Dict[str, Any]:
        """Reinicia o episódio e retorna a observação inicial."""
        self._spawn_vehicle()
        self._setup_sensors()
        self.collision_detected = False
        self.distance_traveled = 0.0
        self.previous_location = None

        # 🚀 OTIMIZAÇÃO: Remover sleep desnecessário em modo assíncrono
        if self.config.get("synchronous", True):
            if self.world is not None:
                self.world.tick()
        # else: No sleep needed in async mode for faster training

        self._update_spectator()
        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Executa a ação e retorna (observação, recompensa, done, info)."""
        self._apply_action(action)

        # 🚀 OTIMIZAÇÃO: Remover sleep desnecessário em modo assíncrono
        if self.config.get("synchronous", True):
            if self.world is not None:
                self.world.tick()
        # else: No sleep needed in async mode for faster training

        observation = self._get_observation()

        # 🚀 OTIMIZAÇÃO: Cálculo de distância mais eficiente
        current_location = observation["location"]
        if self.previous_location is not None:
            # Calcular apenas diferença horizontal (x,y) para performance
            dx = current_location.x - self.previous_location.x
            dy = current_location.y - self.previous_location.y
            distance = (dx * dx + dy * dy) ** 0.5  # Evitar np.linalg.norm
            self.distance_traveled += distance
        self.previous_location = current_location

        reward = 0.0

        # Centralizar a lógica de término do episódio
        lane_offset_val = float(observation.get("lane_offset", 0.0))
        is_offroad = observation.get("offroad", False) or abs(lane_offset_val) > self.max_lane_offset
        done = self.collision_detected or is_offroad

        success_distance = self.config.get("success_distance", 2000)
        reached_success_distance = (self.distance_traveled >= success_distance)
        done = done or reached_success_distance

        # Determine human-readable termination reason
        termination_reason: Optional[str] = None
        if self.collision_detected:
            termination_reason = "collision"
        elif is_offroad:
            termination_reason = f"offroad (lane_offset={lane_offset_val:.2f}m)"
        elif reached_success_distance:
            termination_reason = f"success_distance_reached ({self.distance_traveled:.2f}m)"

        info: Dict[str, Any] = {
            "collision": self.collision_detected,
            "distance_traveled": self.distance_traveled,
            "success": reached_success_distance,
            "termination_reason": termination_reason,
        }

        if done and self.verbose:
            print(f"[CarlaEnvironment] Episode finished: {termination_reason}")
        self._update_spectator()
        return observation, reward, done, info

    def _apply_action(self, action) -> None:
        control = carla.VehicleControl()
        if isinstance(action, (list, tuple, np.ndarray)):
            throttle = float(action[0]) if len(action) > 0 else 0.0
            steer = float(action[1]) if len(action) > 1 else 0.0
        else:
            throttle = float(action)
            steer = 0.0

        throttle = max(0.0, min(1.0, throttle))
        steer = max(-1.0, min(1.0, steer))
        control.throttle = throttle
        control.steer = steer
        control.brake = 0.0
        self.vehicle.apply_control(control)

    def _get_observation(self) -> Dict[str, Any]:
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        # 🚀 OTIMIZAÇÃO: Cálculo de velocidade mais eficiente (evitar np.linalg.norm)
        speed = (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z) ** 0.5

        waypoint = self.map.get_waypoint(transform.location)

        lane_offset = self._compute_lane_offset(transform.location, waypoint)
        heading_error = self._compute_heading_error(transform.rotation.yaw, waypoint.transform.rotation.yaw if waypoint else transform.rotation.yaw)

        # Armazenar velocidade atual para renderização
        self.current_speed = speed

        return {
            "location": transform.location,
            "speed": speed,
            "lane_offset": lane_offset,
            "heading_error": heading_error,
            "offroad": waypoint is None,
        }

    def _compute_lane_offset(self, location: carla.Location, waypoint: carla.Waypoint) -> float:
        if waypoint is None:
            return 2.0  # Retornar um offset grande para indicar que está fora da pista

        dx = location.x - waypoint.transform.location.x
        dy = location.y - waypoint.transform.location.y
        yaw = np.deg2rad(waypoint.transform.rotation.yaw)
        side = -np.sin(yaw) * dx + np.cos(yaw) * dy
        return float(side)

    def _compute_heading_error(self, yaw: float, target_yaw: float) -> float:
        error = (target_yaw - yaw + 180.0) % 360.0 - 180.0
        return float(error)

    def _setup_sensors(self) -> None:
        # 🚀 OTIMIZAÇÃO: Pular configuração de câmera se desabilitada
        if self.config.get("disable_camera", False):
            # Pular câmera completamente para performance máxima
            self.camera = None
            self.driver_image = None
        elif self.camera is None:
            camera_bp = self.blueprint_library.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(self.config.get("camera_width", 800)))
            camera_bp.set_attribute("image_size_y", str(self.config.get("camera_height", 400)))
            camera_bp.set_attribute("fov", str(self.config.get("camera_fov", 90)))

            camera_transform = carla.Transform(carla.Location(x=self.config.get("camera_x", 1.5), z=self.config.get("camera_z", 1.4)))
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            self.camera.listen(self._camera_callback)

        # Sensor de colisão (sempre necessário para detectar fim de episódio)
        if self.collision_sensor is None:
            collision_bp = self.blueprint_library.find("sensor.other.collision")
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            self.collision_sensor.listen(self._collision_callback)

        self.spectator = self.world.get_spectator()
        self._update_spectator()

    def _draw_waypoints(self) -> None:
        if self.world is None or self.map is None or self.waypoints_drawn:
            return

        distance = float(self.config.get("waypoint_draw_distance", 2.0))
        waypoints = self.map.generate_waypoints(distance=distance)
        debug = self.world.debug
        for waypoint in waypoints:
            debug.draw_point(
                waypoint.transform.location,
                size=0.1,
                color=carla.Color(255, 0, 0),
                life_time=0,
            )
        self.waypoints_drawn = True

    def _safe_destroy(self, actor: Optional[carla.Actor]) -> None:
        if actor is None:
            return
        try:
            # CORREÇÃO: is_alive é um atributo, não uma função
            if hasattr(actor, 'is_alive') and not actor.is_alive:
                return
                
            actor.destroy()
            
            # Em modo síncrono, precisamos de um tick para o servidor 
            # processar a remoção antes de tentarmos spawnar outro.
            if self.config.get("synchronous", True) and self.world is not None:
                self.world.tick()
        except Exception as e:
            pass 
            # print(f"Erro ao destruir ator: {e}")

    def _destroy_sensors(self) -> None:
        """Destrói sensores associados à instância."""
        if self.camera is not None:
            try:
                self.camera.stop()
                self._safe_destroy(self.camera)
            except Exception:
                pass
            finally:
                self.camera = None
                self.driver_image = None
        
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
                self._safe_destroy(self.collision_sensor)
            except Exception:
                pass
            finally:
                self.collision_sensor = None

    def _cleanup_previous_hero_vehicles(self) -> None:
        """Limpa veículos de episódios anteriores e sensores órfãos."""
        if self.world is None:
            return

        try:
            # 1. Primeiro, destrói todos os veículos antigos (exceto self.vehicle)
            # Isso destroirá seus sensores acoplados automaticamente
            vehicles = self.world.get_actors().filter('vehicle.*')
            for actor in vehicles:
                try:
                    # Pula o veículo atual
                    if self.vehicle and actor.id == self.vehicle.id:
                        continue
                    # Destrói qualquer outro veículo
                    self._safe_destroy(actor)
                except Exception:
                    pass

            # 2. Depois, limpar qualquer sensor órfão que possa ter restado
            sensors = self.world.get_actors().filter('sensor.*')
            for sensor in sensors:
                try:
                    # Skip sensores do veículo atual
                    if self.vehicle and sensor.parent and sensor.parent.id == self.vehicle.id:
                        continue
                    # Destruir sensor órfão (sem veículo pai ou com veículo destruído)
                    self._safe_destroy(sensor)
                except Exception:
                    pass
        except Exception:
            pass

    def _hard_cleanup(self) -> None:
        """Limpeza abrangente: destrói todos os veículos e sensores. Usado como fallback."""
        if self.world is None:
            return
        
        try:
            actors = self.world.get_actors()
            
            # Destrói todos os veículos
            for actor in actors.filter('vehicle.*'):
                try:
                    self._safe_destroy(actor)
                except Exception as e:
                    print(f"Erro ao destruir veículo {actor.id}: {e}")
            
            # Destrói todos os sensores
            for actor in actors.filter('sensor.*'):
                try:
                    self._safe_destroy(actor)
                except Exception as e:
                    print(f"Erro ao destruir sensor {actor.id}: {e}")
        except Exception as e:
            print(f"Erro geral em _hard_cleanup: {e}")

    def _camera_callback(self, image: carla.Image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.driver_image = array

    def _collision_callback(self, event: carla.CollisionEvent) -> None:
        self.collision_detected = True

    def _update_spectator(self) -> None:
        if self.vehicle is None or self.spectator is None:
            return

        transform = self.vehicle.get_transform()
        position = transform.location + carla.Location(z=self.config.get("top_down_height", 40.0))
        rotation = carla.Rotation(pitch=self.config.get("top_down_pitch", -90.0), yaw=0.0)
        self.spectator.set_transform(carla.Transform(position, rotation))

    def render(self) -> None:
        # 🚀 OTIMIZAÇÃO: Pular renderização completamente se câmera desabilitada
        if self.config.get("disable_camera", False) or not self.config.get("render", False):
            return

        if self.driver_image is not None and cv2 is not None:
            # Copiar a imagem para adicionar texto
            display_image = self.driver_image.copy()

            try:
                vehicle_id = self.vehicle.id if self.vehicle else "N/A"
                vehicle_type = self.vehicle.type_id if self.vehicle else "N/A"
                speed = f"{getattr(self, 'current_speed', 0.0) * 3.6:.2f} km/h"
                distance = f"{self.distance_traveled:.2f} m"

                cv2.putText(display_image, f"ID: {vehicle_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Type: {vehicle_type}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Speed: {speed}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Distance: {distance}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception as e:
                print(f"Erro ao adicionar texto na imagem: {e}")

            cv2.imshow("CARLA Driver View", display_image)
            cv2.waitKey(1)

    def shutdown(self) -> None:
        """Libera recursos do CARLA."""
        self._destroy_sensors()
        if self.vehicle is not None:
            self._safe_destroy(self.vehicle)
            self.vehicle = None

        if self.world is not None:
            # Limpeza abrangente como fallback para remover atores órfãos
            self._hard_cleanup()
            
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        if cv2 is not None:
            cv2.destroyAllWindows()
