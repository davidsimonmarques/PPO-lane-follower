"""Loop de treinamento usando Stable-Baselines3 e o wrapper Gymnasium."""

import numpy as np
from typing import Dict
import os

from environment.gym_wrapper import CarlaGymWrapper
from utils.logger import setup_logger
from training.callbacks import SuccessRateCallback, EpisodeCountCallback, ActionNoiseDecayCallback, EpisodeMetricsCallback # Importar os callbacks
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList


def train(config: Dict) -> None:
    logger = setup_logger(config)
    env = None  # Inicializar env como None para o bloco finally

    try:
        # 1. Criar o ambiente com o nosso wrapper
        logger.info("Criando ambiente CarlaGymWrapper...")
        env = CarlaGymWrapper(config)
        logger.info("Ambiente criado com sucesso.")


        # 2. Configurar o ruído para exploração do DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), 
            sigma=config.get("noise_initial_sigma", 0.1) * np.ones(n_actions)
        )

        # Callback para salvar checkpoints
        checkpoint_callback = CheckpointCallback(
          save_freq=10000,  # Salva a cada 10.000 passos
          save_path='./logs/checkpoints/',
          name_prefix='ddpg_model',
          save_replay_buffer=True,
          save_vecnormalize=True,
        )

        # Callback para logar a taxa de sucesso (janela de 100 episódios)
        success_callback = SuccessRateCallback(check_freq=1, window_size=100)

        # Callback para parar o treinamento com base no número de episódios
        episode_callback = EpisodeCountCallback(target_episodes=config.get("total_episodes", 1000), verbose=1)

        # Callback para decair o ruído de exploração
        noise_decay_callback = ActionNoiseDecayCallback(
            initial_sigma=config.get("noise_initial_sigma", 0.3),
            final_sigma=config.get("noise_final_sigma", 0.01),
            decay_end_step=config.get("noise_decay_end_step", 500000)
        )

        # Callback para gravar métricas de episódio no TensorBoard
        episode_metrics_callback = EpisodeMetricsCallback(verbose=1)

        # Combinar os callbacks em uma lista para passar ao `learn`
        callback_list = CallbackList([
            checkpoint_callback,
            success_callback,
            episode_callback,
            episode_metrics_callback,
            noise_decay_callback
        ])

        # 3. Instanciar ou carregar o agente DDPG
        model_path = config.get("model_path")
        if config.get("load_pretrained") and model_path and os.path.exists(model_path):
            logger.info(f"Carregando modelo pré-treinado de: {model_path}")
            model = DDPG.load(
                model_path,
                env=env, # Passar o ambiente para continuar o treinamento
                tensorboard_log="./logs/tensorboard/"
            )
            # O replay buffer é carregado automaticamente se foi salvo com o modelo
            logger.info("Modelo e replay buffer carregados.")
        else:
            logger.info("Criando um novo modelo DDPG.")
            model = DDPG(
                "MlpPolicy",
                env,
                action_noise=action_noise,
                gamma=config.get("gamma"),
                tau=config.get("tau"),
                learning_rate=config.get("actor_lr"), # SB3 usa um único learning_rate
                buffer_size=config.get("buffer_size"),
                batch_size=config.get("batch_size"),
                verbose=1,
                tensorboard_log="./logs/tensorboard/"
            )

        # 4. Treinar o modelo
        logger.info("Iniciando o treinamento do modelo DDPG...")
        model.learn(
            total_timesteps=config.get("total_timesteps", 1000000), # Deixamos um valor alto, o callback irá parar
            callback=callback_list  # Adiciona a lista de callbacks
        )

        # 5. Salvar o modelo final
        model.save(config.get("model_path", "assets/ddpg_lane_follower"))
        logger.info(f"Modelo salvo em: {config.get('model_path')}")

    except KeyboardInterrupt:
        logger.info("Treinamento interrompido pelo usuário (Ctrl+C).")
    except Exception as e:
        logger.error(f"Erro crítico durante o treinamento: {e}")
        raise e # Re-levanta o erro para debug após a limpeza
    finally:
        if env is not None:
            logger.info("Encerrando o ambiente CARLA...")
            env.close()
            logger.info("Ambiente encerrado.")