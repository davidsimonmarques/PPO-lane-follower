"""Callbacks customizados para o Stable-Baselines3."""

from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np

class EpisodeCountCallback(BaseCallback):
    """
    Um callback para parar o treinamento após um número específico de episódios.
    """
    def __init__(self, target_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.target_episodes = target_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Verifica se algum dos ambientes terminou um episódio
        for done in self.locals['dones']:
            if done:
                self.episode_count += 1
        
        # Loga o número atual de episódios no TensorBoard
        self.logger.record('rollout/episodes', self.episode_count)

        # Se o número de episódios atingiu o alvo, para o treinamento
        if self.episode_count >= self.target_episodes:
            if self.verbose > 0:
                print(f"Parando treinamento: {self.episode_count} de {self.target_episodes} episódios atingidos.")
            return False # Retornar False para o treinamento
        return True # Retornar True para continuar

class SuccessRateCallback(BaseCallback):
    """
    Um callback para logar a taxa de sucesso em episódios recentes no TensorBoard.
    """
    def __init__(self, check_freq: int, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.window_size = window_size
        self.success_buffer = deque(maxlen=self.window_size)

    def _on_step(self) -> bool:
        # Roda a cada `check_freq` passos
        if self.n_calls % self.check_freq == 0:
            # Checa se algum episódio terminou neste passo
            for i, done in enumerate(self.locals['dones']):
                if done:
                    # Quando um episódio termina, o SB3 armazena a 'info' final em 'final_info'.
                    # A 'info' regular (self.locals['infos'][i]) é do primeiro passo do *novo* episódio.
                    info_dict = self.locals['infos'][i]

                    # --- INÍCIO DO DIAGNÓSTICO ---
                    if self.verbose > 0:
                        print(f"\n[SuccessRateCallback] Episódio terminou. Verificando info: {info_dict}")
                    # --- FIM DO DIAGNÓSTICO ---

                    final_info = info_dict.get('final_info')
                    if final_info is not None:
                        is_success = final_info.get('success', False)
                        self.success_buffer.append(1.0 if is_success else 0.0)
                    elif self.verbose > 0:
                        # Se 'final_info' não for encontrado, nos avise.
                        print("[SuccessRateCallback] A chave 'final_info' não foi encontrada no dicionário de informações.")

        # Loga a taxa de sucesso da janela móvel no TensorBoard
        if len(self.success_buffer) > 0:
            self.logger.record('rollout/success_rate', np.mean(self.success_buffer))
        
        return True


class EpisodeMetricsCallback(BaseCallback):
    """
    Callback para gravar métricas de episódio no TensorBoard:
    - velocidade média do episódio
    - sucesso binário do episódio
    - distância média percorrida por episódio
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_speed_sum = None
        self.episode_step_count = None
        self.episode_distance_buffer = None

    def _on_training_start(self) -> None:
        num_envs = getattr(self.training_env, 'num_envs', 1)
        self.episode_speed_sum = [0.0 for _ in range(num_envs)]
        self.episode_step_count = [0 for _ in range(num_envs)]
        self.episode_distance_buffer = [deque(maxlen=100) for _ in range(num_envs)]

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])

        for i, info in enumerate(infos):
            speed = info.get('speed')
            if speed is not None:
                self.episode_speed_sum[i] += float(speed)
                self.episode_step_count[i] += 1

            if dones[i]:
                final_info = info.get('final_info', info)
                avg_speed = 0.0
                if self.episode_step_count[i] > 0:
                    avg_speed = self.episode_speed_sum[i] / float(self.episode_step_count[i])

                success = 1.0 if final_info.get('success', False) else 0.0
                distance = float(final_info.get('distance_traveled', 0.0))
                self.episode_distance_buffer[i].append(distance)
                avg_distance = float(np.mean(self.episode_distance_buffer[i])) if len(self.episode_distance_buffer[i]) > 0 else 0.0

                self.logger.record('rollout/episode_avg_speed', avg_speed)
                self.logger.record('rollout/episode_success', success)
                self.logger.record('rollout/episode_avg_distance', avg_distance)

                if self.verbose > 0:
                    print(f"[EpisodeMetricsCallback] episode_avg_speed={avg_speed:.3f} success={int(success)} episode_avg_distance={avg_distance:.3f}")

                self.episode_speed_sum[i] = 0.0
                self.episode_step_count[i] = 0

        return True


class ActionNoiseDecayCallback(BaseCallback):
    """
    Callback para reduzir gradualmente o desvio padrão do ruído de ação ao longo do treinamento.
    """
    def __init__(self, initial_sigma: float, final_sigma: float, decay_end_step: int, verbose: int = 0):
        super().__init__(verbose)
        self.initial_sigma = float(initial_sigma)
        self.final_sigma = float(final_sigma)
        self.decay_end_step = int(decay_end_step)

    def _on_step(self) -> bool:
        if not hasattr(self.model, 'action_noise') or self.model.action_noise is None:
            return True

        progress = min(1.0, float(self.num_timesteps) / float(max(1, self.decay_end_step)))
        new_sigma = self.initial_sigma + (self.final_sigma - self.initial_sigma) * progress

        if isinstance(self.model.action_noise, list):
            for noise in self.model.action_noise:
                if hasattr(noise, 'sigma'):
                    noise.sigma = np.full_like(noise.sigma, new_sigma)
                elif hasattr(noise, '_sigma'):
                    noise._sigma = np.full_like(noise._sigma, new_sigma)
        else:
            if hasattr(self.model.action_noise, 'sigma'):
                self.model.action_noise.sigma = np.full_like(self.model.action_noise.sigma, new_sigma)
            elif hasattr(self.model.action_noise, '_sigma'):
                self.model.action_noise._sigma = np.full_like(self.model.action_noise._sigma, new_sigma)

        if self.verbose > 0 and self.n_calls % 1000 == 0:
            print(f"[ActionNoiseDecayCallback] step={self.num_timesteps} sigma={new_sigma:.6f}")

        return True