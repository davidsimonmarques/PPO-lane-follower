"""Callbacks customizados para o Stable-Baselines3."""

from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np


class EpisodeCountCallback(BaseCallback):
    """Conta episódios e para o treinamento ao atingir o alvo."""
    def __init__(self, target_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.target_episodes = target_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        for done in self.locals.get('dones', []):
            if done:
                self.episode_count += 1

        self.logger.record('rollout/episodes', self.episode_count)

        if self.episode_count >= self.target_episodes:
            if self.verbose > 0:
                print(f"Parando treinamento: {self.episode_count} episódios atingidos.")
            return False
        return True


class SuccessRateCallback(BaseCallback):
    """Loga a taxa de sucesso (janela móvel) no TensorBoard."""
    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.success_buffer = deque(maxlen=self.window_size)

    def _on_step(self) -> bool:
        for done in self.locals.get('dones', []):
            if done:
                self.success_buffer.append(0.0)  # Default: falhou

        if len(self.success_buffer) > 0:
            self.logger.record('rollout/success_rate', np.mean(self.success_buffer))

        return True

    def on_rollout_end(self) -> None:
        """Chamado no fim de cada rollout - bom para atualizar métricas acumuladas."""
        pass


class EpisodeMetricsCallback(BaseCallback):
    """
    Loga métricas de episódio no TensorBoard:
    - episode_success: sucesso binário
    - episode_lane_offset: lane offset médio (absoluto)
    - episode_heading_error: heading error médio (absoluto)
    - episode_reward: recompensa total do episódio
    - episode_distance: distância percorrida
    - episode_speed: velocidade média
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_data = None

    def _on_training_start(self) -> None:
        num_envs = getattr(self.training_env, 'num_envs', 1)
        self.episode_data = [{
            'speed_sum': 0.0,
            'lane_offset_sum': 0.0,
            'heading_error_sum': 0.0,
            'reward_sum': 0.0,
            'step_count': 0,
            'speeds': [],
            'lane_offsets': [],
            'heading_errors': [],
        } for _ in range(num_envs)]

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        rewards = self.locals.get('rewards', [])
        observations = self.locals.get('new_obs', [])

        num_envs = len(dones)

        # Debug: logar uma vez a estrutura dos dados recebidos
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            print(f"[EpisodeMetrics DEBUG] num_envs={num_envs}")
            print(f"  dones type={type(dones)}, len={len(dones)}")
            print(f"  infos type={type(infos)}, len={len(infos)}")
            if len(infos) > 0:
                print(f"  infos[0] type={type(infos[0])}, keys={list(infos[0].keys()) if isinstance(infos[0], dict) else 'N/A'}")
            print(f"  rewards type={type(rewards)}, len={len(rewards)}")
            print(f"  new_obs type={type(observations)}, shape={getattr(observations, 'shape', 'N/A')}")

        for i in range(num_envs):
            done = bool(dones[i]) if i < len(dones) else False
            info = infos[i] if i < len(infos) else {}
            reward = float(rewards[i]) if i < len(rewards) else 0.0

            # Acumular recompensa
            self.episode_data[i]['reward_sum'] += reward

            # Extrair métricas do info
            speed = info.get('speed')
            lane_offset = info.get('lane_offset')
            heading_error = info.get('heading_error')

            # Se não encontrou no info, tentar extrair da observação
            if speed is None and i < len(observations) and len(observations[i]) >= 3:
                # A observação normalizada: [lane_offset/4, heading_error/180, speed/10]
                speed = float(observations[i][2]) * 10.0
                lane_offset = float(observations[i][0]) * 4.0
                heading_error = float(observations[i][1]) * 180.0

            if speed is not None:
                self.episode_data[i]['speed_sum'] += float(speed)
                self.episode_data[i]['speeds'].append(float(speed))
                self.episode_data[i]['step_count'] += 1
            if lane_offset is not None:
                self.episode_data[i]['lane_offset_sum'] += abs(float(lane_offset))
                self.episode_data[i]['lane_offsets'].append(abs(float(lane_offset)))
            if heading_error is not None:
                self.episode_data[i]['heading_error_sum'] += abs(float(heading_error))
                self.episode_data[i]['heading_errors'].append(abs(float(heading_error)))

            if done:
                d = self.episode_data[i]
                steps = max(d['step_count'], 1)

                avg_speed = d['speed_sum'] / steps
                avg_lane_offset = d['lane_offset_sum'] / steps
                avg_heading_error = d['heading_error_sum'] / steps
                total_reward = d['reward_sum']

                # Sucesso
                success = 1.0 if info.get('success', False) else 0.0
                distance = float(info.get('distance_traveled', 0.0))

                # Logar no TensorBoard
                self.logger.record('rollout/episode_success', success)
                self.logger.record('rollout/episode_lane_offset', avg_lane_offset)
                self.logger.record('rollout/episode_heading_error', avg_heading_error)
                self.logger.record('rollout/episode_reward', total_reward)
                self.logger.record('rollout/episode_distance', distance)
                self.logger.record('rollout/episode_speed', avg_speed)

                if self.verbose > 0:
                    print(
                        f"[Metrics] "
                        f"success={int(success)} "
                        f"lane_off={avg_lane_offset:.3f} "
                        f"head_err={avg_heading_error:.3f} "
                        f"reward={total_reward:.1f} "
                        f"dist={distance:.0f}m "
                        f"speed={avg_speed:.2f}m/s"
                    )

                # Resetar
                self.episode_data[i] = {
                    'speed_sum': 0.0,
                    'lane_offset_sum': 0.0,
                    'heading_error_sum': 0.0,
                    'reward_sum': 0.0,
                    'step_count': 0,
                    'speeds': [],
                    'lane_offsets': [],
                    'heading_errors': [],
                }

        return True