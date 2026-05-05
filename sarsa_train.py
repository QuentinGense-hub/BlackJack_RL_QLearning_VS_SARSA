import numpy as np

from blackjack_utils import (
    SARSA_MODEL_PATH,
    evaluate_agent,
    make_env,
    save_q_table,
    train_sarsa,
)


def main():
    q_table, rewards, _ = train_sarsa()
    env = make_env()

    try:
        metrics = evaluate_agent(env, q_table)
    finally:
        env.close()

    save_q_table(q_table, SARSA_MODEL_PATH)

    print("Entrainement SARSA termine")
    print("Reward moyen :", np.mean(rewards[-10_000:]))
    print(metrics)
    print(f"Modele sauvegarde dans {SARSA_MODEL_PATH}")


if __name__ == "__main__":
    main()
