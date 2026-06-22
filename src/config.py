CONFIG = {
    # Conexão CARLA
    "host": "127.0.0.1",
    "port": 2000,
    "map_name": "Town01_Opt",
    "verbose": True,

    # Treino geral com Stable-Baselines3
    # Para lane following com PPO, são necessários ~20-50M de timesteps.
    # O critério de parada é total_timesteps (mais confiável que episódios).
    "total_episodes": 50000,  # Limite alto de segurança (parada por timesteps é preferível)
    "total_timesteps": 15000000,  # 15M com delta=0.1 = mesmo tempo real que 30M com delta=0.05
    "state_size": 3,
    "action_size": 2,  # [throttle_brake, steer]

    # Hiperparâmetros do PPO (para SB3) - OTIMIZADOS PARA PERFORMANCE
    # n_steps × n_envs(1) = batch efetivo por update
    # Com n_steps=4096, o modelo atualiza a cada 4096 passos (~410s reais com delta=0.1)
    # Mais dados por update = gradientes mais estáveis
    "gamma": 0.99,
    "actor_lr": 0.0003,
    "clip_range": 0.2,
    "n_steps": 4096,      # Aumentado de 2048: coleta mais dados antes de cada update
    "n_epochs": 20,        # Aumentado de 10: mais épocas de treino por batch (dados mais ricos)
    "batch_size": 256,     # Aumentado de 64: processamento mais eficiente em GPU/CPU

    # Entropia e advantage
    "ent_coef": 0.01,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    # Arquitetura da rede neural
    "policy_net_arch": [128, 128],  # Duas camadas ocultas de 128 neurônios (vs 64x64 padrão)

    # Logging
    "log_level": "INFO",
    "model_path": "src\\logs\\checkpoints\\ppo_model_450000_steps.zip",
    "load_pretrained": False,

    # Espaço de ação (clamp)
    "action_low": [-1.0, -1.0],  # [throttle_brake, steer]
    "action_high": [0.6, 1.0],

    # Recompensa / shaping
    "lane_center_reward": 5.0,
    "heading_penalty": 0.05,
    "offroad_penalty": 200.0,
    "stuck_penalty": 200.0,
    "success_reward": 400.0,
    "max_speed_ref": 40 / 3.6,
    "speed_reward": 0.05,
    "success_distance": 2000,

    # Câmera e Render
    "render": False,
    "camera_width": 800,
    "camera_height": 400,
    "camera_fov": 90,
    "camera_x": 1.5,
    "camera_z": 1.4,
    "top_down_height": 20.0,
    "top_down_pitch": -90.0,

    # Simulação e Performance
    # fixed_delta_seconds=0.05 = 20Hz. Para máxima performance, use 0.1 (10Hz).
    # Quanto maior, mais rápido treina mas menos preciso o controle.
    "synchronous": True,
    "fixed_delta_seconds": 0.02,
    "stuck_speed_threshold": 0.1,
    "stuck_time_threshold": 4.0,
    "disable_camera": True,
    "disable_collision_sensor": False,
    "max_fps": 30,

    # Otimização de performance
    # Desabilitar rendering completo do CARLA
    "no_rendering_mode": True,
}