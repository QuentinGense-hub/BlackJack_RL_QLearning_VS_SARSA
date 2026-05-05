from blackjack_utils import (
    DEFAULT_RECENT_METRICS_GAMES,
    SARSA_MODEL_PATH,
    compute_recent_metrics,
    save_q_table,
    train_sarsa,
)


def main():
    q_table, rewards, _ = train_sarsa()
    metrics = compute_recent_metrics(rewards, DEFAULT_RECENT_METRICS_GAMES)

    save_q_table(q_table, SARSA_MODEL_PATH)

    print("Entrainement SARSA termine")
    print(f"Metriques sur les {metrics['sample_size']} derniers episodes :")
    print("Reward moyen :", metrics["mean_reward"])
    print(
        {
            "wins": metrics["wins"],
            "losses": metrics["losses"],
            "draws": metrics["draws"],
        }
    )
    print(f"Modele sauvegarde dans {SARSA_MODEL_PATH}")


if __name__ == "__main__":
    main()
