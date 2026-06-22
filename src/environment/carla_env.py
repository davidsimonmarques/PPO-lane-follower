"""Wrapper de ambiente CARLA para lane follower.
Otimizado para máxima performance durante treinamento."""

import random
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
        self.distance_traveled = 0.0
        self.previous_location: Optional[carla.Location] = None

        # Detecção de "preso"
        self.stuck_timer = 0.0
        self.stuck_speed_threshold = float(self.config.get("stuck_speed_threshold", 0.1))
        self.stuck_time_threshold = float(self.config.get("stuck_time_threshold", 5.0))

        # Flag de rendering (evita lookup repetido)
        self._no_rendering = self.config.get("no_rendering_mode", not self.config.get("render", False))

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

        # Configurar tudo de uma vez
        settings = self.world.get_settings()
        settings.synchronous_mode = self.config.get("synchronous", True)
        settings.fixed_delta_seconds = self.config.get("fixed_delta_seconds", 0.05)
        settings.no_rendering_mode = self._no_rendering

        if self.config.get("max_fps"):
            settings.max_substep_delta_time = 1.0 / self.config["max_fps"]
            settings.max_substeps = 1

        self.world.apply_settings(settings)

        self._spawn_vehicle()
        self._setup_sensors()

        if self.config.get("draw_waypoints", False):
            self._draw_waypoints()

    def _spawn_vehicle(self) -> None:
        self._destroy_sensors()
        if self.vehicle is not None:
            self._batch_destroy([self.vehicle])
            self.vehicle = None

        self._cleanup_previous_hero_vehicles()

        filter_name = self.config.get("vehicle_filter", "vehicle.tesla.model3")
        blueprint = self.blueprint_library.filter(filter_name)[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")

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
        self.stuck_timer = 0.0

        if self.config.get("synchronous", True):
            if self.world is not None:
                self.world.tick()

        if not self._no_rendering:
            self._update_spectator()

        return self._get_observation()

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Executa a ação e retorna (observação, recompensa, done, info)."""
        self._apply_action(action)

        if self.config.get("synchronous", True):
            if self.world is not None:
                self.world.tick()

        observation = self._get_observation()

        # Detecção de "preso"
        is_stuck = False
        if observation["speed"] < self.stuck_speed_threshold:
            self.stuck_timer += self.config.get("fixed_delta_seconds", 0.05)
            if self.stuck_timer > self.stuck_time_threshold:
                is_stuck = True
        else:
            self.stuck_timer = 0.0

        # Cálculo de distância eficiente
        current_location = observation["location"]
        if self.previous_location is not None:
            dx = current_location.x - self.previous_location.x
            dy = current_location.y - self.previous_location.y
            distance = (dx * dx + dy * dy) ** 0.5
            self.distance_traveled += distance
        self.previous_location = current_location

        reward = 0.0

        lane_offset_val = float(observation.get("lane_offset", 0.0))
        is_offroad = observation.get("offroad", False) or abs(lane_offset_val) > self.max_lane_offset
        done = self.collision_detected or is_offroad or is_stuck

        success_distance = self.config.get("success_distance", 2000)
        reached_success_distance = (self.distance_traveled >= success_distance)
        done = done or reached_success_distance

        observation["stuck"] = is_stuck

        termination_reason: Optional[str] = None
        if self.collision_detected:
            termination_reason = "collision"
        elif is_offroad:
            termination_reason = f"offroad (lane_offset={lane_offset_val:.2f}m)"
        elif is_stuck:
            termination_reason = "stuck"
        elif reached_success_distance:
            termination_reason = f"success_distance_reached ({self.distance_traveled:.2f}m)"

        info: Dict[str, Any] = {
            "collision": self.collision_detected,
            "distance_traveled": self.distance_traveled,
            "success": reached_success_distance,
            "termination_reason": termination_reason,
            "stuck": is_stuck,
        }

        if done and self.verbose:
            print(f"[CarlaEnvironment] Episode finished: {termination_reason}")

        # Pular spectator update em modo no_rendering (treinamento)
        if not self._no_rendering:
            self._update_spectator()

        return observation, reward, done, info

    def _apply_action(self, action) -> None:
        control = carla.VehicleControl()

        if not isinstance(action, (list, tuple, np.ndarray)) or len(action) < 2:
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 1.0
            self.vehicle.apply_control(control)
            return

        action_throttle_brake = float(action[0])
        steer = float(action[1])

        control.steer = max(-1.0, min(1.0, steer))

        if action_throttle_brake > 0:
            control.throttle = 0.2 + action_throttle_brake * (1.0 - 0.2)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = abs(action_throttle_brake)

        self.vehicle.apply_control(control)

    def _get_observation(self) -> Dict[str, Any]:
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z) ** 0.5

        waypoint = self.map.get_waypoint(transform.location)

        lane_offset = self._compute_lane_offset(transform.location, waypoint)
        heading_error = self._compute_heading_error(
            transform.rotation.yaw,
            waypoint.transform.rotation.yaw if waypoint else transform.rotation.yaw
        )

        self.current_speed = speed

        return {
            "location": transform.location,
            "speed": speed,
            "lane_offset": lane_offset,
            "heading_error": heading_error,
            "offroad": waypoint is None,
            "stuck": False,
        }

    def _compute_lane_offset(self, location: carla.Location, waypoint: carla.Waypoint) -> float:
        if waypoint is None:
            return 2.0
        dx = location.x - waypoint.transform.location.x
        dy = location.y - waypoint.transform.location.y
        yaw = np.deg2rad(waypoint.transform.rotation.yaw)
        side = -np.sin(yaw) * dx + np.cos(yaw) * dy
        return float(side)

    def _compute_heading_error(self, yaw: float, target_yaw: float) -> float:
        error = (target_yaw - yaw + 180.0) % 360.0 - 180.0
        return float(error)

    def _setup_sensors(self) -> None:
        # Pular câmera completamente para performance máxima
        if self.config.get("disable_camera", False):
            self.camera = None
            self.driver_image = None
        elif self.camera is None:
            camera_bp = self.blueprint_library.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(self.config.get("camera_width", 800)))
            camera_bp.set_attribute("image_size_y", str(self.config.get("camera_height", 400)))
            camera_bp.set_attribute("fov", str(self.config.get("camera_fov", 90)))
            camera_transform = carla.Transform(
                carla.Location(x=self.config.get("camera_x", 1.5), z=self.config.get("camera_z", 1.4))
            )
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            self.camera.listen(self._camera_callback)

        # Sensor de colisão (sempre necessário)
        if self.collision_sensor is None:
            collision_bp = self.blueprint_library.find("sensor.other.collision")
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            self.collision_sensor.listen(self._collision_callback)

        # Pular spectator em modo no_rendering (treinamento)
        if not self._no_rendering:
            self.spectator = self.world.get_spectator()
            self._update_spectator()
        else:
            self.spectator = None

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
        """Destroi um actor individual SEM world.tick() — tick será feito em batch."""
        if actor is None:
            return
        try:
            if hasattr(actor, 'is_alive') and not actor.is_alive:
                return
            actor.destroy()
        except Exception:
            pass

    def _batch_destroy(self, actors: list) -> None:
        """Destroi uma lista de actors de uma vez e chama 1 único world.tick() no final."""
        for actor in actors:
            if actor is None:
                continue
            try:
                if hasattr(actor, 'is_alive') and not actor.is_alive:
                    continue
                actor.destroy()
            except Exception:
                pass
        if self.config.get("synchronous", True) and self.world is not None:
            try:
                self.world.tick()
            except Exception:
                pass

    def _destroy_sensors(self) -> None:
        """Destroi camera e collision sensor em batch (1 tick)."""
        actors = []
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            actors.append(self.camera)
            self.camera = None
            self.driver_image = None

        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
            except Exception:
                pass
            actors.append(self.collision_sensor)
            self.collision_sensor = None

        self._batch_destroy(actors)

    def _cleanup_previous_hero_vehicles(self) -> None:
        """Limpa veículos e sensores órfãos com batch destroy (1 tick)."""
        if self.world is None:
            return
        actors_to_destroy = []
        try:
            my_id = self.vehicle.id if self.vehicle else -1
            for actor in self.world.get_actors().filter('vehicle.*'):
                if actor.id != my_id:
                    actors_to_destroy.append(actor)
            for sensor in self.world.get_actors().filter('sensor.*'):
                if sensor.parent is None or sensor.parent.id != my_id:
                    actors_to_destroy.append(sensor)
        except Exception:
            pass
        if actors_to_destroy:
            self._batch_destroy(actors_to_destroy)

    def _hard_cleanup(self) -> None:
        """Limpa todos os actors com batch destroy (1 tick)."""
        if self.world is None:
            return
        actors_to_destroy = []
        try:
            for actor in self.world.get_actors().filter('vehicle.*'):
                actors_to_destroy.append(actor)
            for actor in self.world.get_actors().filter('sensor.*'):
                actors_to_destroy.append(actor)
        except Exception:
            pass
        if actors_to_destroy:
            self._batch_destroy(actors_to_destroy)

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
        if self.config.get("disable_camera", False) or not self.config.get("render", False):
            return
        if self.driver_image is not None and cv2 is not None:
            display_image = self.driver_image.copy()
            try:
                speed = f"{getattr(self, 'current_speed', 0.0) * 3.6:.2f} km/h"
                distance = f"{self.distance_traveled:.2f} m"
                cv2.putText(display_image, f"Speed: {speed}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Distance: {distance}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception:
                pass
            cv2.imshow("CARLA Driver View", display_image)
            cv2.waitKey(1)

    def shutdown(self) -> None:
        """Libera recursos do CARLA."""
        self._destroy_sensors()
        if self.vehicle is not None:
            self._batch_destroy([self.vehicle])
            self.vehicle = None
        if self.world is not None:
            self._hard_cleanup()
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        if cv2 is not None:
            cv2.destroyAllWindows()