"""Agente DDPG para controle contínuo em CARLA usando redes neurais simples em NumPy."""

import logging
import os
from typing import Dict, Tuple

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float32)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        index = self.position
        self.state[index] = state
        self.action[index] = action
        self.reward[index, 0] = reward
        self.next_state[index] = next_state
        self.done[index, 0] = 1.0 if done else 0.0
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_size = min(batch_size, self.size)
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.next_state[indices],
            self.done[indices],
        )

    def __len__(self) -> int:
        return self.size


class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.batch_size = config.get("batch_size", 64)
        self.warmup_steps = config.get("warmup_steps", 1000)
        self.buffer_size = config.get("buffer_size", 200000)
        self.actor_lr = config.get("actor_lr", 0.0005)
        self.critic_lr = config.get("critic_lr", 0.001)
        self.noise_std = config.get("noise_std", 0.2)
        self.noise_decay = config.get("noise_decay", 0.9995)
        self.min_noise_std = config.get("min_noise_std", 0.02)
        self.noise = self.noise_std
        self.update_every = config.get("update_every", 1)
        self.step_count = 0

        self.action_low = np.array(config.get("action_low", [0.0, -1.0]), dtype=np.float32)
        self.action_high = np.array(config.get("action_high", [0.7, 1.0]), dtype=np.float32)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0
        self.noise = self.noise_std

        self.memory = ReplayBuffer(state_dim, action_dim, self.buffer_size)
        self.actor = self._build_actor_network(state_dim, action_dim)
        self.actor_target = self._copy_network(self.actor)
        self.critic = self._build_critic_network(state_dim, action_dim)
        self.critic_target = self._copy_network(self.critic)

        self.model_path = config.get("model_path", "assets/ddpg_agent.npz")
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
        }

    def _build_critic_network(self, state_dim: int, action_dim: int) -> Dict[str, np.ndarray]:
        return {
            "W1": self._init_weights(state_dim + action_dim, 64),
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

    def _copy_network(self, network: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {name: value.copy() for name, value in network.items()}

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

    def choose_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state = state.reshape(1, -1).astype(np.float32)
        if explore and len(self.memory) < self.warmup_steps:
            return np.random.uniform(self.action_low, self.action_high).astype(np.float32)

        action, _ = self._actor_forward(self.actor, state)
        action = action[0]
        if explore:
            action = action + np.random.randn(self.action_dim).astype(np.float32) * self.noise
            action = np.clip(action, self.action_low, self.action_high)
        return action

    def step(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.add(state, action, reward, next_state, done)
        self.step_count += 1
        if (
            len(self.memory) >= self.batch_size
            and len(self.memory) >= self.warmup_steps
            and self.step_count % self.update_every == 0
        ):
            self._learn()
            self.noise = max(self.min_noise_std, self.noise * self.noise_decay)

    def _learn(self) -> None:
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        target_actions, _ = self._actor_forward(self.actor_target, next_states)
        q_next, _ = self._critic_forward(self.critic_target, next_states, target_actions)
        y = rewards + self.gamma * (1.0 - dones) * q_next

        q_values, critic_cache = self._critic_forward(self.critic, states, actions)
        td_error = q_values - y
        critic_grad = 2.0 * td_error / self.batch_size
        self._critic_backward(critic_grad, critic_cache)

        predicted_actions, actor_cache = self._actor_forward(self.actor, states)
        q_for_actor, critic_cache_for_actor = self._critic_forward(self.critic, states, predicted_actions)
        action_gradients = self._critic_action_gradient(critic_cache_for_actor)
        self._actor_backward(-action_gradients / self.batch_size, actor_cache)

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _actor_forward(self, network: Dict[str, np.ndarray], state: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        z1 = state @ network["W1"] + network["b1"]
        a1 = relu(z1)
        z2 = a1 @ network["W2"] + network["b2"]
        a2 = relu(z2)
        z3 = a2 @ network["W3"] + network["b3"]
        raw_action = np.tanh(z3)
        action = raw_action * self.action_scale + self.action_bias
        return action, (state, z1, a1, z2, a2, z3)

    def _critic_forward(self, network: Dict[str, np.ndarray], state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        x = np.concatenate([state, action], axis=-1)
        z1 = x @ network["W1"] + network["b1"]
        a1 = relu(z1)
        z2 = a1 @ network["W2"] + network["b2"]
        a2 = relu(z2)
        q = a2 @ network["W3"] + network["b3"]
        return q, (x, z1, a1, z2, a2)

    def _critic_action_gradient(self, critic_cache: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        x, z1, a1, z2, a2 = critic_cache
        delta = np.ones((x.shape[0], 1), dtype=np.float32)
        grad_a2 = delta @ self.critic["W3"].T
        grad_z2 = grad_a2 * relu_derivative(z2)
        grad_a1 = grad_z2 @ self.critic["W2"].T
        grad_z1 = grad_a1 * relu_derivative(z1)
        grad_x = grad_z1 @ self.critic["W1"].T
        return grad_x[:, self.state_dim:]

    def _critic_backward(self, grad_output: np.ndarray, cache: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        x, z1, a1, z2, a2 = cache
        dW3 = a2.T @ grad_output
        db3 = np.mean(grad_output, axis=0, keepdims=True)
        d_a2 = grad_output @ self.critic["W3"].T
        d_z2 = d_a2 * relu_derivative(z2)
        dW2 = a1.T @ d_z2
        db2 = np.mean(d_z2, axis=0, keepdims=True)
        d_a1 = d_z2 @ self.critic["W2"].T
        d_z1 = d_a1 * relu_derivative(z1)
        dW1 = x.T @ d_z1
        db1 = np.mean(d_z1, axis=0, keepdims=True)

        clip_value = 1.0
        np.clip(dW1, -clip_value, clip_value, out=dW1)
        np.clip(dW2, -clip_value, clip_value, out=dW2)
        np.clip(dW3, -clip_value, clip_value, out=dW3)
        np.clip(db1, -clip_value, clip_value, out=db1)
        np.clip(db2, -clip_value, clip_value, out=db2)
        np.clip(db3, -clip_value, clip_value, out=db3)

        self.critic["W1"] -= self.critic_lr * dW1
        self.critic["b1"] -= self.critic_lr * db1
        self.critic["W2"] -= self.critic_lr * dW2
        self.critic["b2"] -= self.critic_lr * db2
        self.critic["W3"] -= self.critic_lr * dW3
        self.critic["b3"] -= self.critic_lr * db3

    def _actor_backward(self, grad_action: np.ndarray, cache: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        state, z1, a1, z2, a2, z3 = cache
        d_raw = grad_action * tanh_derivative(z3) * self.action_scale
        dW3 = a2.T @ d_raw
        db3 = np.mean(d_raw, axis=0, keepdims=True)
        d_a2 = d_raw @ self.actor["W3"].T
        d_z2 = d_a2 * relu_derivative(z2)
        dW2 = a1.T @ d_z2
        db2 = np.mean(d_z2, axis=0, keepdims=True)
        d_a1 = d_z2 @ self.actor["W2"].T
        d_z1 = d_a1 * relu_derivative(z1)
        dW1 = state.T @ d_z1
        db1 = np.mean(d_z1, axis=0, keepdims=True)

        clip_value = 1.0
        np.clip(dW1, -clip_value, clip_value, out=dW1)
        np.clip(dW2, -clip_value, clip_value, out=dW2)
        np.clip(dW3, -clip_value, clip_value, out=dW3)
        np.clip(db1, -clip_value, clip_value, out=db1)
        np.clip(db2, -clip_value, clip_value, out=db2)
        np.clip(db3, -clip_value, clip_value, out=db3)

        self.actor["W1"] += self.actor_lr * dW1
        self.actor["b1"] += self.actor_lr * db1
        self.actor["W2"] += self.actor_lr * dW2
        self.actor["b2"] += self.actor_lr * db2
        self.actor["W3"] += self.actor_lr * dW3
        self.actor["b3"] += self.actor_lr * db3

    def _soft_update(self, source: Dict[str, np.ndarray], target: Dict[str, np.ndarray]) -> None:
        for name in source:
            target[name] = self.tau * source[name] + (1.0 - self.tau) * target[name]

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
            critic_W1=self.critic["W1"],
            critic_b1=self.critic["b1"],
            critic_W2=self.critic["W2"],
            critic_b2=self.critic["b2"],
            critic_W3=self.critic["W3"],
            critic_b3=self.critic["b3"],
        )
        self.logger.info(f"Modelo DDPG salvo em: {path}")

    def load(self, path: str) -> None:
        params = np.load(path)
        self.actor["W1"] = params["actor_W1"].astype(np.float32)
        self.actor["b1"] = params["actor_b1"].astype(np.float32)
        self.actor["W2"] = params["actor_W2"].astype(np.float32)
        self.actor["b2"] = params["actor_b2"].astype(np.float32)
        self.actor["W3"] = params["actor_W3"].astype(np.float32)
        self.actor["b3"] = params["actor_b3"].astype(np.float32)
        self.critic["W1"] = params["critic_W1"].astype(np.float32)
        self.critic["b1"] = params["critic_b1"].astype(np.float32)
        self.critic["W2"] = params["critic_W2"].astype(np.float32)
        self.critic["b2"] = params["critic_b2"].astype(np.float32)
        self.critic["W3"] = params["critic_W3"].astype(np.float32)
        self.critic["b3"] = params["critic_b3"].astype(np.float32)
        self.actor_target = self._copy_network(self.actor)
        self.critic_target = self._copy_network(self.critic)
        self.logger.info(f"Modelo DDPG carregado de: {path}")
