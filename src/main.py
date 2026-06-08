"""Ponto de entrada para treinar e avaliar o lane follower."""

from config import CONFIG
from training.train import train


def main() -> None:
    train(CONFIG)


if __name__ == "__main__":
    main()
