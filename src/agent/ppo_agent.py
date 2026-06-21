"""Agente PPO para controle contínuo em CARLA usando redes neurais simples em NumPy."""

import logging
import os
from typing import Dict, List, Tuple

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float32)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


class RolloutBuffer:
    """Buffer de rollout para PPO: armazena transições de um rollout completo."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.capacity = int(capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.advantages = np.zeros(self.capacity, dtype=np.float32)
        self.returns = np.zeros(self.capacity, dtype=np.float32)

        self.position = 0
        self.size = 0
        self._last_value = 0.0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, done: bool, value: float, log_prob: float) -> None:
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = 1.0 if done else 0.0
        self.values[self.position] = value
        self.log_probs[self.position] = log_prob
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def compute_gae(self, last_value: float) -> None:
        """Calcula Generalized Advantage Estimation (GAE) e retornos."""
        self._last_value = last_value
        last_gae_lam = 0.0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        self.returns = self.advantages + self.values[:self.size]

    def get_batches(self, batch_size: int) -> List[Tuple[np.ndarray, ...]]:
        """Divide o rollout em mini-batches para treinamento."""
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        batches = []
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            batch_indices = indices[start:end]
            batches.append((
                self.states[batch_indices],
                self.actions[batch_indices],
                self.log_probs[batch_indices],
                self.advantages[batch_indices],
                self.returns[batch_indices],
                self.values[batch_indices],
            ))
        return batches

    def reset(self) -> None:
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_range = config.get("clip_range", 0.2)
        self.n_epochs = config.get("n_epochs", 10)
        self.batch_size = config.get("batch_size", 64)
        self.ent_coef = config.get("ent_coef", 0.01)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.actor_lr = config.get("actor_lr", 0.0003)

        self.action_low = np.array(config.get("action_low", [-1.0, -1.0]), dtype=np.float32)
        self.action_high = np.array(config.get("action_high", [1.0, 1.0]), dtype=np.float32)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self.n_steps = config.get("n_steps", 2048)
        self.buffer = RolloutBuffer(state_dim, action_dim, self.n_steps, self.gamma, self.gae_lambda)
        self.step_count = 0

        self.actor = self._build_actor_network(state_dim, action_dim)
        self.critic = self._build_critic_network(state_dim)

        # Estado para forward pass atual
        self._current_log_prob = 0.0
        self._current_value = 0.0

        self.model_path = config.get("model_path", "assets/ppo_agent.npz")
        self.load_pretrained = config.get("load_pretrained", False)
        self.logger = logging.getLogger("lane_follower")
        if self.load_pretrained and os.path.exists(self.model_path):
            self.load(self.model_path)

    def _build_actor_network(self, state_dim: int, action_dim: int) -> Dict[str, np.ndarray]:
        return {
            "W1": self._init_weights(state_dim, 64),
            "b1": np.zeros((1, 64), dtype=np.float32),
            "W2": self._init_weights(64, 64),
            "b2": np.zeros((1, 64), dtype=np.float32),
            "W3": self._init_weights(64, action_dim, std=3e-4),
            "b3": np.zeros((1, action_dim), dtype=np.float32),
            # Log std para a política estocástica
            "log_std": np.full((1, action_dim), -0.5, dtype=np.float32),
        }

    def _build_critic_network(self, state_dim: int) -> Dict[str, np.ndarray]:
        return {
            "W1": self._init_weights(state_dim, 64),
            "b1": np.zeros((1, 64), dtype=np.float32),
            "W2": self._init_weights(64, 64),
            "b2": np.zeros((1, 64), dtype=np.float32),
            "W3": self._init_weights(64, 1, std=3e-4),
            "b3": np.zeros((1, 1), dtype=np.float32),
        }

    def _init_weights(self, in_dim: int, out_dim: int, std: float = None) -> np.ndarray:
        if std is None:
            std = np.sqrt(2.0 / max(1, in_dim))
        return np.random.randn(in_dim, out_dim).astype(np.float32) * std

    def observation_to_state(self, observation: Dict) -> np.ndarray:
        lane_offset = float(observation.get("lane_offset", 0.0))
        heading_error = float(observation.get("heading_error", 0.0))
        speed = float(observation.get("speed", 0.0))
        state = np.array([
            lane_offset / 2.0,
            heading_error / 180.0,
            speed / 10.0,
        ], dtype=np.float32)
        return np.clip(state, -1.0, 1.0)

    def _actor_forward(self, network: Dict[str, np.ndarray], state: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """Forward pass do ator: retorna média da distribuição e cache."""
        z1 = state @ network["W1"] + network["b1"]
        a1 = relu(z1)
        z2 = a1 @ network["W2"] + network["b2"]
        a2 = relu(z2)
        z3 = a2 @ network["W3"] + network["b3"]
        mean = np.tanh(z3)
        return mean, (state, z1, a1, z2, a2, z3)

    def _critic_forward(self, network: Dict[str, np.ndarray], state: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """Forward pass do crítico: retorna valor estimado e cache."""
        z1 = state @ network["W1"] + network["b1"]
        a1 = relu(z1)
        z2 = a1 @ network["W2"] + network["b2"]
        a2 = relu(z2)
        v = a2 @ network["W3"] + network["b3"]
        return v, (state, z1, a1, z2, a2)

    def _sample_action(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Amostra ação de uma distribuição gaussiana e retorna (ação, log_prob)."""
        log_std = self.actor["log_std"]
        std = np.exp(log_std)
        noise = np.random.randn(*mean.shape).astype(np.float32)
        raw_action = mean + std * noise
        action = np.tanh(raw_action)

        # Log prob da distribução gaussiana considerando a transformação tanh
        log_prob = -0.5 * (((raw_action - mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        # Ajuste pela derivada do tanh
        log_prob -= np.log(1 - action ** 2 + 1e-6)
        log_prob = np.sum(log_prob, axis=-1)

        # Escalar ação para o espaço de ação
        action = action * self.action_scale + self.action_bias
        return action, log_prob

    def choose_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Escolhe uma ação (determinística quando explore=False)."""
        state = state.reshape(1, -1).astype(np.float32)
        mean, _ = self._actor_forward(self.actor, state)

        if explore:
            action, log_prob = self._sample_action(mean)
            self._current_log_prob = float(log_prob[0, 0]) if log_prob.ndim > 0 else float(log_prob)
            value, _ = self._critic_forward(self.critic, state)
            self._current_value = float(value[0, 0]) if value.ndim > 0 else float(value)
            return action[0]
        else:
            # Ação determinística: usar a média
            action = mean[0]
            action = action * self.action_scale + self.action_bias
            return np.clip(action, self.action_low, self.action_high)

    def step(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Armazena a transição no buffer. O update é chamado depois do rollout."""
        self.buffer.add(state, action, reward, done, self._current_value, self._current_log_prob)
        self.step_count += 1

    def should_update(self) -> bool:
        """Verifica se o buffer está cheio e deve fazer update."""
        return len(self.buffer) >= self.n_steps

    def _learn(self) -> Dict[str, float]:
        """Executa o update do PPO usando os dados do buffer de rollout."""
        # Calcular GAE e retornos
        # Usar o último valor estimado para bootstrap
        last_state = self.buffer.states[self.buffer.size - 1:self.buffer.size]
        if last_state.shape[0] > 0:
            last_value, _ = self._critic_forward(self.critic, last_state)
            last_value = float(last_value[0, 0])
        else:
            last_value = 0.0
        self.buffer.compute_gae(last_value)

        # Normalizar vantagens
        adv = self.buffer.advantages[:self.buffer.size]
        adv_mean = np.mean(adv)
        adv_std = np.std(adv) + 1e-8
        self.buffer.advantages[:self.buffer.size] = (adv - adv_mean) / adv_std

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            batches = self.buffer.get_batches(self.batch_size)
            for batch in batches:
                states, actions, old_log_probs, advantages, returns, _ = batch
                batch_size_actual = states.shape[0]

                # Forward pass do ator
                mean, actor_cache = self._actor_forward(self.actor, states)
                log_std = self.actor["log_std"]
                std = np.exp(log_std)
                noise = (np.arctanh(np.clip(actions / self.action_scale - self.action_bias / self.action_scale, -0.999, 0.999)) - mean) / (std + 1e-8)
                raw_action = mean + std * noise
                new_log_prob = -0.5 * (((raw_action - mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
                action_tanh = np.tanh(raw_action)
                new_log_prob -= np.log(1 - action_tanh ** 2 + 1e-6)
                new_log_prob = np.sum(new_log_prob, axis=-1)

                # Entropy
                entropy = np.sum(0.5 + 0.5 * np.log(2 * np.pi * np.e * std ** 2 + 1e-8), axis=-1)
                mean_entropy = np.mean(entropy)

                # PPO clip
                ratio = np.exp(new_log_prob - old_log_probs)
                surr1 = ratio * advantages
                surr2 = np.clip(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                actor_loss = -np.mean(np.minimum(surr1, surr2)) - self.ent_coef * mean_entropy

                # Backward pass do ator
                self._actor_backward(actor_loss, actor_cache, states, advantages, old_log_probs, log_std, std, noise, actions)

                # Forward pass do crítico
                values, critic_cache = self._critic_forward(self.critic, states)
                critic_loss = np.mean((values.squeeze(-1) - returns) ** 2)

                # Backward pass do crítico
                self._critic_backward(critic_loss, critic_cache, values, returns)

                total_actor_loss += float(actor_loss)
                total_critic_loss += float(critic_loss)
                total_entropy += float(mean_entropy)
                n_updates += 1

        # Limpar buffer
        self.buffer.reset()

        metrics = {
            "actor_loss": total_actor_loss / max(1, n_updates),
            "critic_loss": total_critic_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }
        return metrics

    def _actor_backward(self, actor_loss: float, cache: Tuple, states: np.ndarray,
                        advantages: np.ndarray, old_log_probs: np.ndarray,
                        log_std: np.ndarray, std: np.ndarray, noise: np.ndarray,
                        actions: np.ndarray) -> None:
        """Backward pass simplificado do ator usando gradientes numéricos/analíticos aproximados."""
        state, z1, a1, z2, a2, z3 = cache
        batch_size = states.shape[0]

        # Gradient clipping value
        clip_value = self.max_grad_norm

        # Aproximação do gradiente via diferenças finitas para o actor é custosa,
        # então usamos uma atualização de gradiente estimado.
        # Para uma implementação NumPy pura, approximar o gradiente da policy loss.
        # Aqui usamos uma simplificação: o gradiente do log_prob em relação aos pesos.
        # Na prática, para PPO puro em NumPy, o gradiente do ator é complexo.
        # Usamos uma abordagem de REINFORCE com baseline para estimar o gradiente.

        # Log prob gradient w.r.t. z3 (pre-tanh)
        # d(log_prob)/d(z3) via chain rule
        d_log_policy = advantages.reshape(-1, 1) * (1.0 / (std.reshape(1, -1) + 1e-8))
        d_log_policy = d_log_policy * noise  # Aproximação

        # Through tanh
        d_raw = d_log_policy * tanh_derivative(z3) * self.action_scale

        # Through linear layer 3
        dW3 = a2.T @ d_raw / batch_size
        db3 = np.mean(d_raw, axis=0, keepdims=True)
        d_a2 = d_raw @ self.actor["W3"].T

        # Through ReLU and linear layer 2
        d_z2 = d_a2 * relu_derivative(z2)
        dW2 = a1.T @ d_z2 / batch_size
        db2 = np.mean(d_z2, axis=0, keepdims=True)
        d_a1 = d_z2 @ self.actor["W2"].T

        # Through ReLU and linear layer 1
        d_z1 = d_a1 * relu_derivative(z1)
        dW1 = state.T @ d_z1 / batch_size
        db1 = np.mean(d_z1, axis=0, keepdims=True)

        # Clip gradients
        for dW in [dW1, dW2, dW3]:
            np.clip(dW, -clip_value, clip_value, out=dW)
        for db in [db1, db2, db3]:
            np.clip(db, -clip_value, clip_value, out=db)

        # Update weights (gradient ascent)
        self.actor["W1"] += self.actor_lr * dW1
        self.actor["b1"] += self.actor_lr * db1
        self.actor["W2"] += self.actor_lr * dW2
        self.actor["b2"] += self.actor_lr * db2
        self.actor["W3"] += self.actor_lr * dW3
        self.actor["b3"] += self.actor_lr * db3

    def _critic_backward(self, critic_loss: float, cache: Tuple,
                         values: np.ndarray, returns: np.ndarray) -> None:
        """Backward pass do crítico."""
        state, z1, a1, z2, a2 = cache
        batch_size = state.shape[0]

        # Gradiente da MSE loss w.r.t. output
        grad_output = 2.0 * (values - returns.reshape(-1, 1)) / batch_size

        clip_value = self.max_grad_norm

        # Through linear layer 3
        dW3 = a2.T @ grad_output
        db3 = np.mean(grad_output, axis=0, keepdims=True)
        d_a2 = grad_output @ self.critic["W3"].T

        # Through ReLU and linear layer 2
        d_z2 = d_a2 * relu_derivative(z2)
        dW2 = a1.T @ d_z2
        db2 = np.mean(d_z2, axis=0, keepdims=True)
        d_a1 = d_z2 @ self.critic["W2"].T

        # Through ReLU and linear layer 1
        d_z1 = d_a1 * relu_derivative(z1)
        dW1 = state.T @ d_z1
        db1 = np.mean(d_z1, axis=0, keepdims=True)

        # Clip gradients
        for dW in [dW1, dW2, dW3]:
            np.clip(dW, -clip_value, clip_value, out=dW)
        for db in [db1, db2, db3]:
            np.clip(db, -clip_value, clip_value, out=db)

        # Update weights (gradient descent)
        self.critic["W1"] -= self.actor_lr * dW1
        self.critic["b1"] -= self.actor_lr * db1
        self.critic["W2"] -= self.actor_lr * dW2
        self.critic["b2"] -= self.actor_lr * db2
        self.critic["W3"] -= self.actor_lr * dW3
        self.critic["b3"] -= self.actor_lr * db3

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            actor_W1=self.actor["W1"],
            actor_b1=self.actor["b1"],
            actor_W2=self.actor["W2"],
            actor_b2=self.actor["b2"],
            actor_W3=self.actor["W3"],
            actor_b3=self.actor["b3"],
            actor_log_std=self.actor["log_std"],
            critic_W1=self.critic["W1"],
            critic_b1=self.critic["b1"],
            critic_W2=self.critic["W2"],
            critic_b2=self.critic["b2"],
            critic_W3=self.critic["W3"],
            critic_b3=self.critic["b3"],
        )
        self.logger.info(f"Modelo PPO salvo em: {path}")

    def load(self, path: str) -> None:
        params = np.load(path)
        self.actor["W1"] = params["actor_W1"].astype(np.float32)
        self.actor["b1"] = params["actor_b1"].astype(np.float32)
        self.actor["W2"] = params["actor_W2"].astype(np.float32)
        self.actor["b2"] = params["actor_b2"].astype(np.float32)
        self.actor["W3"] = params["actor_W3"].astype(np.float32)
        self.actor["b3"] = params["actor_b3"].astype(np.float32)
        if "actor_log_std" in params:
            self.actor["log_std"] = params["actor_log_std"].astype(np.float32)
        self.critic["W1"] = params["critic_W1"].astype(np.float32)
        self.critic["b1"] = params["critic_b1"].astype(np.float32)
        self.critic["W2"] = params["critic_W2"].astype(np.float32)
        self.critic["b2"] = params["critic_b2"].astype(np.float32)
        self.critic["W3"] = params["critic_W3"].astype(np.float32)
        self.critic["b3"] = params["critic_b3"].astype(np.float32)
        self.logger.info(f"Modelo PPO carregado de: {path}")