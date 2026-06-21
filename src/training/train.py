"""Loop de treinamento usando Stable-Baselines3 e o wrapper Gymnasium."""

import numpy as np
from typing import Dict
import os

from environment.gym_wrapper import CarlaGymWrapper
from utils.logger import setup_logger
from training.callbacks import EpisodeCountCallback, EpisodeMetricsCallback, SuccessRateCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv


def train(config: Dict) -> None:
    logger = setup_logger(config)
    env = None

    # Caminhos absolutos baseados na raiz do projeto
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _tb_log = os.path.join(_project_root, 'logs', 'tensorboard')
    _ckpt_log = os.path.join(_project_root, 'logs', 'checkpoints')

    try:
        # Garantir que os diretórios de logs existem
        os.makedirs(_tb_log, exist_ok=True)
        os.makedirs(_ckpt_log, exist_ok=True)
        logger.info(f"TensorBoard log: {_tb_log}")

        # 1. Criar o ambiente
        logger.info("Criando ambiente CarlaGymWrapper...")
        env = CarlaGymWrapper(config)
        vec_env = DummyVecEnv([lambda: env])
        logger.info("Ambiente criado com sucesso.")

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=_ckpt_log,
            name_prefix='ppo_model',
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        success_callback = SuccessRateCallback(window_size=100)
        episode_callback = EpisodeCountCallback(target_episodes=config.get("total_episodes", 1000), verbose=1)
        episode_metrics_callback = EpisodeMetricsCallback(verbose=1)

        callback_list = CallbackList([
            checkpoint_callback,
            success_callback,
            episode_callback,
            episode_metrics_callback,
        ])

        # 2. Instanciar ou carregar o agente PPO
        model_path = config.get("model_path")
        if config.get("load_pretrained") and model_path and os.path.exists(model_path):
            logger.info(f"Carregando modelo pré-treinado de: {model_path}")
            model = PPO.load(model_path, env=vec_env, tensorboard_log=_tb_log)
            logger.info("Modelo carregado.")
        else:
            logger.info("Criando um novo modelo PPO.")
            model = PPO(
                "MlpPolicy",
                vec_env,
                gamma=config.get("gamma"),
                learning_rate=config.get("actor_lr"),
                n_steps=config.get("n_steps", 2048),
                batch_size=config.get("batch_size", 64),
                n_epochs=config.get("n_epochs", 10),
                clip_range=config.get("clip_range", 0.2),
                ent_coef=config.get("ent_coef", 0.01),
                gae_lambda=config.get("gae_lambda", 0.95),
                vf_coef=config.get("vf_coef", 0.5),
                max_grad_norm=config.get("max_grad_norm", 0.5),
                verbose=1,
                tensorboard_log=_tb_log,
            )

        # 3. Treinar
        logger.info("Iniciando o treinamento do modelo PPO...")
        model.learn(
            total_timesteps=config.get("total_timesteps", 1000000),
            callback=callback_list
        )

        # 4. Salvar o modelo final
        model.save(config.get("model_path", "assets/ppo_lane_follower"))
        logger.info(f"Modelo salvo em: {config.get('model_path')}")

    except KeyboardInterrupt:
        logger.info("Treinamento interrompido pelo usuário (Ctrl+C).")
    except Exception as e:
        logger.error(f"Erro crítico durante o treinamento: {e}")
        raise e
    finally:
        if env is not None:
            logger.info("Encerrando o ambiente CARLA...")
            env.close()
            logger.info("Ambiente encerrado.")