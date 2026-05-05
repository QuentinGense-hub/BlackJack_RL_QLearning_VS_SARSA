import numpy as np

from blackjack_utils import (
    QLEARNING_MODEL_PATH,
    evaluate_agent,
    make_env,
    save_q_table,
    train_qlearning,
)


def main():
    q_table, rewards, _ = train_qlearning()
    env = make_env()

    try:
        metrics = evaluate_agent(env, q_table)
    finally:
        env.close()

    save_q_table(q_table, QLEARNING_MODEL_PATH)

    print("Entrainement Q-learning termine")
    print("Reward moyen :", np.mean(rewards[-10_000:]))
    print(metrics)
    print(f"Modele sauvegarde dans {QLEARNING_MODEL_PATH}")


if __name__ == "__main__":
    main()
