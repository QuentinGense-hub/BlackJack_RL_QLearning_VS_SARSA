from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from blackjack_utils import (
    DEFAULT_EPISODES,
    DEFAULT_SMOOTHING_BLOCK_SIZE,
    NO_DECAY_EPSILON_DECAY,
    QLEARNING_NO_DECAY_MODEL_PATH,
    QLEARNING_MODEL_PATH,
    SARSA_NO_DECAY_MODEL_PATH,
    SARSA_MODEL_PATH,
    save_q_table,
    smooth_rewards_by_block,
    smooth_rewards_by_window,
    train_qlearning,
    train_sarsa,
)

PLOT_DIR = Path(__file__).resolve().parent / "plots"
DEFAULT_PLOT_PATH = PLOT_DIR / "blackjack_learning_curves.png"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Genere une courbe d'apprentissage lissee pour Q-learning et SARSA, "
            "avec et sans decay epsilon, sur Blackjack."
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
        help="Taille du bloc ou de la fenetre utilisee pour lisser la courbe.",
    )
    parser.add_argument(
        "--smoothing",
        choices=("block", "window"),
        default="block",
        help=(
            "block: un point moyen par bloc, plus lisible ; "
            "window: un point par episode avec moyenne glissante."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Chemin du fichier image genere.",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Ne sauvegarde pas les tables Q entrainees.",
    )
    return parser


def plot_learning_curves(episodes, learning_runs, block_size, smoothing, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = {
        "Q-learning avec decay": {"color": "#1f77b4", "linestyle": "-"},
        "Q-learning sans decay": {"color": "#1f77b4", "linestyle": "--"},
        "SARSA avec decay": {"color": "#d62728", "linestyle": "-"},
        "SARSA sans decay": {"color": "#d62728", "linestyle": "--"},
    }

    plt.figure(figsize=(12, 7))

    for label, rewards in learning_runs:
        if smoothing == "window":
            x_values, y_values = smooth_rewards_by_window(rewards, block_size)
        else:
            x_values, y_values = smooth_rewards_by_block(rewards, block_size)

        style = styles[label]
        plt.plot(
            x_values,
            y_values,
            label=label,
            linewidth=2.0,
            color=style["color"],
            linestyle=style["linestyle"],
        )

    plt.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    plt.title("Courbes d'apprentissage lissees sur Blackjack")
    if smoothing == "window":
        plt.xlabel(f"Episode (moyenne glissante sur {block_size} episodes)")
    else:
        plt.xlabel(f"Episode (moyenne par blocs de {block_size})")
    plt.ylabel("Reward moyen")
    plt.xlim(0, episodes)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train_variant(label, train_fn, episodes, model_path, epsilon_decay, save_models):
    print(f"Entrainement {label} sur {episodes} episodes...")

    train_kwargs = {"n_episodes": episodes}
    if epsilon_decay is not None:
        train_kwargs["epsilon_decay"] = epsilon_decay

    q_table, rewards, _ = train_fn(**train_kwargs)

    if save_models:
        save_q_table(q_table, model_path)

    return label, rewards


def main():
    args = build_parser().parse_args()
    save_models = not args.no_save_models

    learning_runs = [
        train_variant(
            "Q-learning avec decay",
            train_qlearning,
            args.episodes,
            QLEARNING_MODEL_PATH,
            epsilon_decay=None,
            save_models=save_models,
        ),
        train_variant(
            "Q-learning sans decay",
            train_qlearning,
            args.episodes,
            QLEARNING_NO_DECAY_MODEL_PATH,
            epsilon_decay=NO_DECAY_EPSILON_DECAY,
            save_models=save_models,
        ),
        train_variant(
            "SARSA avec decay",
            train_sarsa,
            args.episodes,
            SARSA_MODEL_PATH,
            epsilon_decay=None,
            save_models=save_models,
        ),
        train_variant(
            "SARSA sans decay",
            train_sarsa,
            args.episodes,
            SARSA_NO_DECAY_MODEL_PATH,
            epsilon_decay=NO_DECAY_EPSILON_DECAY,
            save_models=save_models,
        ),
    ]

    plot_learning_curves(
        args.episodes,
        learning_runs,
        args.block_size,
        args.smoothing,
        args.output,
    )

    print(f"Graphe sauvegarde dans {args.output}")


if __name__ == "__main__":
    main()
