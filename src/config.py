CONFIG = {
    # Conexão CARLA
    "host": "127.0.0.1",
    "port": 2000,
    "map_name": "Town10HD_Opt", #"Town10HD", 
    "verbose": True,
    # Treino geral com Stable-Baselines3
    "total_episodes": 7000, # Alvo de episódios para treinar
    "total_timesteps": 10000000, # Um número alto para garantir que o callback de episódios pare primeiro
    "state_size": 3,
    "action_size": 2,

    # Hiperparâmetros do DDPG (para SB3)
    "gamma": 0.99,  # desconto de recompensa: 1.0 = olhar longo prazo, <1 = priorizar curto prazo
    "tau": 0.005,  # soft-update dos alvos: maior = atualizações mais rápidas (menos estáveis)
    "actor_lr": 0.0005,  # Taxa de aprendizado para as redes (ator e crítico)
    "noise_decay": 0.999997,  # Decaimento do ruído durante o treinamento

    # Replay buffer e batch
    "buffer_size": 200000,  # maior = mais diversidade / memória, porém mais amostras antigas
    "batch_size": 64,  # batch maior = estimativa de gradiente mais estável

    # Exploração (ruído de ação)
    "noise_std": 0.1,  # Desvio padrão do ruído Gaussiano

    # Frequência de atualização e logging
    "log_level": "INFO",
    "model_path": "logs/checkpoints/ddpg_model_0000_steps.zip", # Exemplo: carregar um checkpoint
    "load_pretrained": False, # Mude para True para carregar um modelo existente

    # Espaço de ação (clamp)
    "action_low": [0.2, -1.0], # [throttle, steer]
    "action_high": [1.0, 1.0],

    # Recompensa / shaping
    "lane_center_reward": 4.0,  # Recompensa por centralização na pista.
    "heading_penalty": 0.03,  # penaliza diferença de orientação; ajustar para reduzir curvas incorretas
    "offroad_penalty": 200.0,  # penalidade forte por sair da pista (aplicada preferencialmente ao fim do episódio)
    "success_reward": 300.0,  # bônus por completar a rota
    "max_speed_ref": 40/3.6,  # km/h / 3.6 = m/s; referência para recompensa de velocidade; 
    "speed_reward": 0.15,  # recompensa máxima por velocidade (proporcional à velocidade atual)
    "success_distance": 2000,

    # Configurações de Câmera e Render
    "render": False, 
    "camera_width": 800,
    "camera_height": 400,
    "camera_fov": 90,
    "camera_x": 1.5,
    "camera_z": 1.4,
    "top_down_height": 20.0,
    "top_down_pitch": -90.0,

    # Configurações de Simulação e Performance
    "synchronous": True,  # ESSENCIAL para treinamento estável e reprodutível em RL.
    "fixed_delta_seconds": 0.05,  # Passo de simulação menor (20Hz) para controle mais fino.
    "disable_camera": True,
    "disable_collision_sensor": False,
    "max_fps": 30,
}
