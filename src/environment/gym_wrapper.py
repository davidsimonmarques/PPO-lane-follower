"""
Wrapper para adaptar o CarlaEnvironment personalizado à interface do Gymnasium.
Isso permite que ele seja usado com bibliotecas como Stable-Baselines3.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environment.carla_env import CarlaEnvironment
from reward.reward_function import RewardFunction


class CarlaGymWrapper(gym.Env):
    """
    Esta classe envolve nosso CarlaEnvironment em uma interface padrão do Gymnasium.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # O ambiente CARLA personalizado e a função de recompensa
        self.env = CarlaEnvironment(config)
        self.reward_fn = RewardFunction(config)

        # Definir o espaço de ação (contínuo: [throttle, steer])
        self.action_space = spaces.Box(
            low=np.array(config.get("action_low")),
            high=np.array(config.get("action_high")),
            dtype=np.float32
        )

        # Definir o espaço de observação (contínuo: [lane_offset, heading_error, speed])
        # Normalizado entre -1 e 1
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.get("state_size", 3),),
            dtype=np.float32
        )

    def _to_state(self, observation: dict) -> np.ndarray:
        """Converte a observação do ambiente em um vetor de estado normalizado."""
        lane_offset = observation.get("lane_offset", 0.0)
        heading_error = observation.get("heading_error", 0.0)
        speed = observation.get("speed", 0.0)
        
        max_lane = float(self.config.get("max_lane_offset", 4.0))
        state = np.array([
            lane_offset / max_lane,          # Normaliza o desvio pelo limite configurado
            heading_error / 180.0,           # Normaliza o ângulo de -180 a 180
            speed / self.config.get("max_speed_ref", 10.0), # Normaliza a velocidade
        ], dtype=np.float32)
        
        return np.clip(state, -1.0, 1.0)

    def step(self, action: np.ndarray):
        # A lógica de término do episódio (done) agora está centralizada no CarlaEnvironment.
        # O wrapper apenas passa os valores adiante.
        observation, _, done, info = self.env.step(action)

        reward = self.reward_fn.compute(observation, action, done, info)
        state = self._to_state(observation)

        # Expor métricas úteis para callbacks / TensorBoard
        info = dict(info)
        info.update({
            "speed": float(observation.get("speed", 0.0)),
            "success": bool(info.get("success", False)),
            "distance_traveled": float(info.get("distance_traveled", 0.0)),
            "lane_offset": float(observation.get("lane_offset", 0.0)),
            "heading_error": float(observation.get("heading_error", 0.0)),
        })

        # A API do Gymnasium retorna `terminated` e `truncated`
        terminated = done
        truncated = info.get("max_steps_reached", False)

        return state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.env.reset()
        state = self._to_state(observation)
        info = {}
        return state, info

    def render(self, mode='human'):
        self.env.render()

    def close(self):
        print("Fechando o ambiente CARLA...")
        self.env.shutdown()