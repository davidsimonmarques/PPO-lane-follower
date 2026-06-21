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
    "total_timesteps": 30000000,  # 30M de passos — suficiente para PPO convergir em lane following
    "state_size": 3,
    "action_size": 2,  # [throttle_brake, steer]

    # Hiperparâmetros do PPO (para SB3)
    # n_steps × n_envs(1) = batch efetivo por update
    # Com 1 env e n_steps=2048, o modelo atualiza a cada 2048 passos
    "gamma": 0.99,
    "actor_lr": 0.0003,
    "clip_range": 0.2,
    "n_steps": 2048,
    "n_epochs": 10,
    "batch_size": 64,

    # Entropia e advantage
    "ent_coef": 0.01,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    # Logging
    "log_level": "INFO",
    "model_path": "src\\logs\\checkpoints\\ppo_model_450000_steps.zip",
    "load_pretrained": False,

    # Espaço de ação (clamp)
    "action_low": [-1.0, -1.0],  # [throttle_brake, steer]
    "action_high": [0.7, 1.0],

    # Recompensa / shaping
    "lane_center_reward": 4.0,
    "heading_penalty": 0.03,
    "offroad_penalty": 200.0,
    "stuck_penalty": 200.0,
    "brake_penalty": 0.05,
    "success_reward": 400.0,
    "max_speed_ref": 40 / 3.6,
    "speed_reward": 0.1,
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
    "fixed_delta_seconds": 0.05,
    "stuck_speed_threshold": 0.1,
    "stuck_time_threshold": 5.0,
    "disable_camera": True,
    "disable_collision_sensor": False,
    "max_fps": 30,

    # Otimização de performance (AMD RX6600 8GB + 16GB RAM)
    # Desabilitar rendering completo do CARLA
    "no_rendering_mode": True,
}