from __future__ import annotations

import argparse
import time

from blackjack_utils import (
    QLEARNING_MODEL_PATH,
    SARSA_MODEL_PATH,
    load_q_table,
    make_env,
    save_q_table,
    train_qlearning,
    train_sarsa,
)

ACTION_NAMES = {
    0: "stick",
    1: "hit",
}


def format_state(state):
    player_sum, dealer_card, usable_ace = state
    ace_text = "oui" if usable_ace else "non"
    return (
        f"somme joueur={player_sum}, carte visible croupier={dealer_card}, "
        f"as utilisable={ace_text}"
    )


def format_cards(cards):
    return "[" + ", ".join(str(card) for card in cards) + "]"


def reward_label(reward):
    if reward > 0:
        return "victoire"
    if reward < 0:
        return "defaite"
    return "match nul"


def print_recorded_episode(episode, agent_name, delay=0.0):
    print(f"\n=== {agent_name} | Episode {episode['episode_number']} ===")
    print(f"Etat initial: {format_state(episode['initial_state'])}")
    print(
        "Cartes initiales:"
        f" joueur={format_cards(episode['initial_player_cards'])}"
        f" | croupier={format_cards(episode['initial_dealer_cards'])}"
    )

    for step_number, step in enumerate(episode["steps"], start=1):
        print(f"Tour {step_number}: {ACTION_NAMES[step['action']]}")
        print(
            "Cartes:"
            f" joueur={format_cards(step['player_cards'])}"
            f" | croupier={format_cards(step['dealer_cards'])}"
        )

        if step["done"]:
            print(f"Resultat: {reward_label(step['reward'])} (reward={step['reward']})")
        else:
            print(f"Nouvel etat: {format_state(step['next_state'])}")

        if delay > 0:
            time.sleep(delay)

    print(f"Reward total de l'episode: {episode['total_reward']}")


def play_policy_episode(env, q_table, episode_number, agent_name, delay=0.0, seed=None):
    state, _ = env.reset(seed=seed)
    done = False
    step_number = 1

    print(f"\n=== {agent_name} | Partie finale {episode_number} ===")
    print(f"Etat initial: {format_state(state)}")
    print(
        "Cartes initiales:"
        f" joueur={format_cards(env.unwrapped.player)}"
        f" | croupier={format_cards(env.unwrapped.dealer)}"
    )

    while not done:
        action_values = q_table[state]
        action = int(action_values.argmax())
        print(f"Tour {step_number}: {ACTION_NAMES[action]}")

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        print(
            "Cartes:"
            f" joueur={format_cards(env.unwrapped.player)}"
            f" | croupier={format_cards(env.unwrapped.dealer)}"
        )

        if done:
            print(f"Resultat: {reward_label(reward)} (reward={reward})")
        else:
            print(f"Nouvel etat: {format_state(next_state)}")

        state = next_state
        step_number += 1

        if delay > 0:
            time.sleep(delay)


def play_policy_episode_human(
    env,
    q_table,
    episode_number,
    agent_name,
    delay=0.0,
    episode_pause=1.5,
    seed=None,
):
    state, _ = env.reset(seed=seed)
    done = False

    print(f"{agent_name} | Partie {episode_number} en cours...")

    while not done:
        action_values = q_table[state]
        action = int(action_values.argmax())
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state

        if delay > 0:
            time.sleep(delay)

    print(f"{agent_name} | Partie {episode_number} terminee | {reward_label(reward)}")

    if episode_pause > 0:
        time.sleep(episode_pause)


def train_and_record_agent(name, model_path, train_fn, episodes, last_episodes):
    print(
        f"{name}: entrainement sur {episodes} episodes "
        f"et enregistrement des {last_episodes} derniers."
    )
    q_table, _, recorded_episodes = train_fn(
        n_episodes=episodes,
        record_last_episodes=last_episodes,
    )
    save_q_table(q_table, model_path)
    print(f"{name}: modele final sauvegarde dans {model_path}.")
    return recorded_episodes


def load_or_train_policy_agent(name, model_path, train_fn, episodes, retrain):
    if model_path.exists() and not retrain:
        print(f"{name}: chargement du modele sauvegarde ({model_path.name}).")
        return load_q_table(model_path)

    print(f"{name}: entrainement en cours sur {episodes} episodes...")
    q_table, _, _ = train_fn(n_episodes=episodes)
    save_q_table(q_table, model_path)
    print(f"{name}: modele sauvegarde dans {model_path}.")
    return q_table


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Observe les 100 derniers episodes d'entrainement ou fait jouer "
            "les politiques finales de Q-learning et SARSA."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("training", "policy"),
        default="training",
        help=(
            "training: affiche les derniers episodes d'entrainement ; "
            "policy: affiche des parties jouees avec la politique finale."
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Nombre d'episodes/parties a afficher par agent.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500_000,
        help="Nombre total d'episodes d'entrainement.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Pause supplementaire en secondes entre deux actions.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("human", "text"),
        default="human",
        help=(
            "Mode d'affichage en mode policy. "
            "human ouvre la fenetre Gymnasium, text garde l'affichage terminal."
        ),
    )
    parser.add_argument(
        "--episode-pause",
        type=float,
        default=1.5,
        help="Pause en secondes a la fin de chaque partie en mode human.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed de base pour reproduire les parties en mode policy.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force un nouvel entrainement en mode policy meme si un modele existe deja.",
    )
    return parser


def run_training_mode(args):
    qlearning_episodes = train_and_record_agent(
        "Q-learning",
        QLEARNING_MODEL_PATH,
        train_qlearning,
        args.episodes,
        args.games,
    )
    sarsa_episodes = train_and_record_agent(
        "SARSA",
        SARSA_MODEL_PATH,
        train_sarsa,
        args.episodes,
        args.games,
    )

    for agent_name, episodes in (
        ("Q-learning", qlearning_episodes),
        ("SARSA", sarsa_episodes),
    ):
        for episode in episodes:
            print_recorded_episode(episode, agent_name, delay=args.delay)


def run_policy_mode(args):
    qlearning_q = load_or_train_policy_agent(
        "Q-learning",
        QLEARNING_MODEL_PATH,
        train_qlearning,
        args.episodes,
        args.retrain,
    )
    sarsa_q = load_or_train_policy_agent(
        "SARSA",
        SARSA_MODEL_PATH,
        train_sarsa,
        args.episodes,
        args.retrain,
    )

    for agent_index, (agent_name, q_table) in enumerate(
        (
            ("Q-learning", qlearning_q),
            ("SARSA", sarsa_q),
        )
    ):
        render_mode = "human" if args.render_mode == "human" else None
        env = make_env(render_mode=render_mode)

        try:
            for game_index in range(1, args.games + 1):
                episode_seed = None
                if args.seed is not None:
                    episode_seed = args.seed + (agent_index * 10_000) + game_index

                if args.render_mode == "human":
                    play_policy_episode_human(
                        env,
                        q_table,
                        game_index,
                        agent_name,
                        delay=args.delay,
                        episode_pause=args.episode_pause,
                        seed=episode_seed,
                    )
                else:
                    play_policy_episode(
                        env,
                        q_table,
                        game_index,
                        agent_name,
                        delay=args.delay,
                        seed=episode_seed,
                    )
        finally:
            env.close()


def main():
    args = build_parser().parse_args()

    if args.mode == "training":
        run_training_mode(args)
    else:
        run_policy_mode(args)


if __name__ == "__main__":
    main()
