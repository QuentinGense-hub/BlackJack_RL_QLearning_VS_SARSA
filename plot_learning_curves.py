from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from blackjack_utils import (
    DEFAULT_EPISODES,
    DEFAULT_SMOOTHING_BLOCK_SIZE,
    QLEARNING_MODEL_PATH,
    SARSA_MODEL_PATH,
    save_q_table,
    smooth_rewards_by_block,
    train_qlearning,
    train_sarsa,
)

PLOT_DIR = Path(__file__).resolve().parent / "plots"
DEFAULT_PLOT_PATH = PLOT_DIR / "blackjack_learning_curves.png"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Genere une courbe d'apprentissage lissee par blocs de 100 episodes "
            "pour Q-learning et SARSA sur Blackjack."
        )
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Nombre total d'episodes d'entrainement pour chaque agent.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_SMOOTHING_BLOCK_SIZE,
        help="Taille des blocs utilises pour lisser la courbe.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Chemin du fichier image genere.",
    )
    return parser


def plot_learning_curves(episodes, qlearning_rewards, sarsa_rewards, block_size, output_path):
    qlearning_x, qlearning_y = smooth_rewards_by_block(qlearning_rewards, block_size)
    sarsa_x, sarsa_y = smooth_rewards_by_block(sarsa_rewards, block_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))
    plt.plot(qlearning_x, qlearning_y, label="Q-learning", linewidth=2.0, color="#1f77b4")
    plt.plot(sarsa_x, sarsa_y, label="SARSA", linewidth=2.0, color="#d62728")
    plt.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    plt.title("Courbes d'apprentissage lissees sur Blackjack")
    plt.xlabel(f"Episode (moyenne par blocs de {block_size})")
    plt.ylabel("Reward moyen")
    plt.xlim(0, episodes)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    args = build_parser().parse_args()

    print(f"Entrainement Q-learning sur {args.episodes} episodes...")
    qlearning_q, qlearning_rewards, _ = train_qlearning(n_episodes=args.episodes)
    save_q_table(qlearning_q, QLEARNING_MODEL_PATH)

    print(f"Entrainement SARSA sur {args.episodes} episodes...")
    sarsa_q, sarsa_rewards, _ = train_sarsa(n_episodes=args.episodes)
    save_q_table(sarsa_q, SARSA_MODEL_PATH)

    plot_learning_curves(
        args.episodes,
        qlearning_rewards,
        sarsa_rewards,
        args.block_size,
        args.output,
    )

    print(f"Graphe sauvegarde dans {args.output}")


if __name__ == "__main__":
    main()
