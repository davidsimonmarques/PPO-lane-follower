"""Cálculo de recompensa específico para lane following."""

from typing import Dict
import numpy as np

class RewardFunction:
    def __init__(self, config: Dict):
        self.lane_center_reward = config.get("lane_center_reward", 1.5)
        self.heading_penalty = config.get("heading_penalty", 0.5)
        self.offroad_penalty = config.get("offroad_penalty", 50.0)
        self.success_reward = config.get("success_reward", 50.0)
        self.speed_reward = config.get("speed_reward", 0.15)
        self.max_speed_ref = config.get("max_speed_ref", 10.0)
        self.max_lane_offset = float(config.get("max_lane_offset", 4.0))


    def compute(self, observation: Dict, action: list, done: bool) -> float:
        lane_offset = abs(observation.get("lane_offset", 0.0))
        is_offroad = observation.get("offroad", False) or (lane_offset > self.max_lane_offset)
        speed = observation.get("speed", 0.0)
        
        # 1. Penalidade Terminal
        if is_offroad:
            return -self.offroad_penalty
            
        # 2. Bônus Terminal de Sucesso
        if done:
            return self.success_reward
            
        # 3. Cálculo de Recompensa Passo a Passo (Step Reward)
        reward = 0.0
        
        # O quão centrado o carro está? (Varia de 1.0 no centro absoluto até 0.0 na borda)
        centering_factor = max(0.0, 1.0 - (lane_offset / self.max_lane_offset))
        
        # Fator de velocidade: O carro só ganha pontos expressivos se estiver se movendo
        # Evita que o agente pare no centro da pista para "farmar" pontos infinitos
        speed_factor = min(1.0, speed / self.max_speed_ref)
        
        # A recompensa principal é a combinação de estar centrado E em boa velocidade
        reward += (self.lane_center_reward * centering_factor * speed_factor)
        
        # 4. Penalidades para Suavidade e Orientação
        # Descomentado e ativo: Essencial para o carro não andar de lado ou zig-zaguear
        heading_error = abs(observation.get("heading_error", 0.0))
        reward -= self.heading_penalty * heading_error
        
        # Penaliza esterçamento brusco do volante (assumindo que action[0] seja steering no range [-1, 1])
        # Ajuste o índice do array 'action' conforme o design do seu environment
        steering_action = abs(action[0]) 
        steering_penalty_weight = 0.06
        reward -= steering_penalty_weight * (steering_action ** 2) # Ao quadrado penaliza mais os extremos
        
        # (Opcional) Penalidade por step para incentivar o término rápido do percurso
        reward -= 0.05 
        
        return float(reward)


    # def compute(self, observation: Dict, action: int, done: bool) -> float:
    #     lane_offset = abs(observation.get("lane_offset", 0.0))
    #     is_offroad = observation.get("offroad", False) or (lane_offset > self.max_lane_offset)
        
    #     if is_offroad:
    #         # Penalidade única por sair da pista (não acumula por steps)
    #         return -self.offroad_penalty
        
    #     # Recompensa por manter-se na pista (maior no centro)
    #     reward = self.lane_center_reward - (lane_offset / (2*self.lane_center_reward))  # 1.0 no centro, 0 nas bordas
    #     #reward -= self.heading_penalty * abs(observation.get("heading_error", 0.0))  # Penalidade por erro de direção
    #     # Bônus de velocidade (substituindo o bônus de progresso)
    #     speed = observation.get("speed", 0.0)
    #     speed_bonus = min(self.speed_reward , (speed / self.max_speed_ref) * self.speed_reward)
    #     reward += speed_bonus
        
    #     # Bônus de sucesso (apenas quando completa a distância)
    #     if done and not is_offroad:
    #         reward += self.success_reward
        
    #     return float(reward)
