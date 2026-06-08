"""Módulo para logging de dados de treinamento em CSV."""

import csv
import os
from datetime import datetime
from typing import List, Optional

class DataLogger:
    """Logger para salvar dados de treinamento em CSV."""

    def __init__(self, log_dir: str = "logs", filename: Optional[str] = None):
        self.log_dir = log_dir
        
        # Se filename não fornecido, gerar automaticamente com timestamp
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.csv"
        
        self.filename = filename
        self.filepath = os.path.join(log_dir, filename)

        # Criar diretório se não existir
        os.makedirs(log_dir, exist_ok=True)

        # Inicializar CSV se não existir
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        """Inicializa o arquivo CSV com cabeçalhos se não existir."""
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'episode',
                    'total_reward',
                    'avg_lane_offset',
                    'avg_heading_error',
                    'avg_speed',
                    'max_speed',
                    'min_speed',
                    'distance_traveled',
                    'success'
                ])

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        lane_offsets: List[float],
        heading_errors: List[float],
        speeds: List[float],
        distance_traveled: float,
        success: bool
    ) -> None:
        """Registra dados de um episódio no CSV.

        Args:
            episode: Número do episódio
            total_reward: Recompensa total do episódio
            lane_offsets: Lista de offsets de lane durante o episódio
            heading_errors: Lista de erros de heading durante o episódio
            speeds: Lista de velocidades durante o episódio
            distance_traveled: Distância total percorrida
            success: Se o episódio terminou com sucesso
        """
        # Calcular médias e estatísticas
        avg_lane_offset = sum(lane_offsets) / len(lane_offsets) if lane_offsets else 0.0
        avg_heading_error = sum(heading_errors) / len(heading_errors) if heading_errors else 0.0
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0
        min_speed = min(speeds) if speeds else 0.0

        # Escrever no CSV
        with open(self.filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                episode,
                f"{total_reward:.4f}",
                f"{avg_lane_offset:.4f}",
                f"{avg_heading_error:.4f}",
                f"{avg_speed:.4f}",
                f"{max_speed:.4f}",
                f"{min_speed:.4f}",
                f"{distance_traveled:.4f}",
                success
            ])

    def get_summary(self) -> Optional[dict]:
        """Retorna um resumo dos dados de treinamento."""
        if not os.path.exists(self.filepath):
            return None

        episodes = []
        rewards = []
        successes = []

        with open(self.filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                episodes.append(int(row['episode']))
                rewards.append(float(row['total_reward']))
                successes.append(row['success'].lower() == 'true')

        if not episodes:
            return None

        return {
            'total_episodes': len(episodes),
            'avg_reward': sum(rewards) / len(rewards),
            'success_rate': sum(successes) / len(successes),
            'best_reward': max(rewards),
            'worst_reward': min(rewards)
        }