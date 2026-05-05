from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np

DEFAULT_ALPHA = 0.01
DEFAULT_GAMMA = 0.95
DEFAULT_EPSILON = 1.0
DEFAULT_EPSILON_MIN = 0.05
DEFAULT_EPSILON_DECAY = 0.999995
DEFAULT_EPISODES = 1_000_000
DEFAULT_EVAL_GAMES = 100_000
DEFAULT_RECENT_METRICS_GAMES = 100
DEFAULT_SMOOTHING_BLOCK_SIZE = 10000
DEFAULT_NATURAL = True
ACTION_COUNT = 2

MODEL_DIR = Path(__file__).resolve().parent / "models"
QLEARNING_MODEL_PATH = MODEL_DIR / "blackjack_qlearning.pkl"
SARSA_MODEL_PATH = MODEL_DIR / "blackjack_sarsa.pkl"


def make_env(render_mode=None, natural=DEFAULT_NATURAL):
    return gym.make(
        "Blackjack-v1",
        sab=False,
        natural=natural,
        render_mode=render_mode,
    )


def create_q_table():
    return defaultdict(lambda: np.zeros(ACTION_COUNT, dtype=float))


def choose_action(env, q_table, state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return int(np.argmax(q_table[state]))


def snapshot_episode(env, episode_number, initial_state):
    return {
        "episode_number": episode_number,
        "initial_state": initial_state,
        "initial_player_cards": list(env.unwrapped.player),
        "initial_dealer_cards": list(env.unwrapped.dealer),
        "steps": [],
        "total_reward": 0.0,
    }


def train_qlearning(
    n_episodes=DEFAULT_EPISODES,
    alpha=DEFAULT_ALPHA,
    gamma=DEFAULT_GAMMA,
    epsilon=DEFAULT_EPSILON,
    epsilon_min=DEFAULT_EPSILON_MIN,
    epsilon_decay=DEFAULT_EPSILON_DECAY,
    record_last_episodes=0,
):
    env = make_env()
    q_table = create_q_table()
    rewards = []
    recorded_episodes = []

    for episode_index in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        should_record = episode_index >= max(0, n_episodes - record_last_episodes)
        episode_log = None

        if should_record:
            episode_log = snapshot_episode(env, episode_index + 1, state)

        while not done:
            action = choose_action(env, q_table, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next_value = 0.0 if done else float(np.max(q_table[next_state]))
            q_table[state][action] += alpha * (
                reward + gamma * best_next_value - q_table[state][action]
            )

            if episode_log is not None:
                episode_log["steps"].append(
                    {
                        "state": state,
                        "action": action,
                        "next_state": next_state,
                        "reward": reward,
                        "done": done,
                        "player_cards": list(env.unwrapped.player),
                        "dealer_cards": list(env.unwrapped.dealer),
                    }
                )

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)

        if episode_log is not None:
            episode_log["total_reward"] = total_reward
            recorded_episodes.append(episode_log)

    env.close()
    return q_table, rewards, recorded_episodes


def train_sarsa(
    n_episodes=DEFAULT_EPISODES,
    alpha=DEFAULT_ALPHA,
    gamma=DEFAULT_GAMMA,
    epsilon=DEFAULT_EPSILON,
    epsilon_min=DEFAULT_EPSILON_MIN,
    epsilon_decay=DEFAULT_EPSILON_DECAY,
    record_last_episodes=0,
):
    env = make_env()
    q_table = create_q_table()
    rewards = []
    recorded_episodes = []

    for episode_index in range(n_episodes):
        state, _ = env.reset()
        action = choose_action(env, q_table, state, epsilon)
        done = False
        total_reward = 0.0
        should_record = episode_index >= max(0, n_episodes - record_last_episodes)
        episode_log = None

        if should_record:
            episode_log = snapshot_episode(env, episode_index + 1, state)

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                target = reward
            else:
                next_action = choose_action(env, q_table, next_state, epsilon)
                target = reward + gamma * q_table[next_state][next_action]

            q_table[state][action] += alpha * (target - q_table[state][action])

            if episode_log is not None:
                episode_log["steps"].append(
                    {
                        "state": state,
                        "action": action,
                        "next_state": next_state,
                        "reward": reward,
                        "done": done,
                        "player_cards": list(env.unwrapped.player),
                        "dealer_cards": list(env.unwrapped.dealer),
                    }
                )

            if not done:
                state = next_state
                action = next_action

            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)

        if episode_log is not None:
            episode_log["total_reward"] = total_reward
            recorded_episodes.append(episode_log)

    env.close()
    return q_table, rewards, recorded_episodes


def evaluate_agent(env, q_table, n_games=DEFAULT_EVAL_GAMES):
    wins = losses = draws = 0

    for _ in range(n_games):
        state, _ = env.reset()
        done = False

        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins / n_games,
        "losses": losses / n_games,
        "draws": draws / n_games,
    }


def compute_recent_metrics(rewards, n_games=DEFAULT_RECENT_METRICS_GAMES):
    recent_rewards = rewards[-n_games:]
    sample_size = len(recent_rewards)

    if sample_size == 0:
        return {
            "sample_size": 0,
            "mean_reward": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "draws": 0.0,
        }

    wins = sum(reward > 0 for reward in recent_rewards)
    losses = sum(reward < 0 for reward in recent_rewards)
    draws = sum(reward == 0 for reward in recent_rewards)

    return {
        "sample_size": sample_size,
        "mean_reward": float(np.mean(recent_rewards)),
        "wins": wins / sample_size,
        "losses": losses / sample_size,
        "draws": draws / sample_size,
    }


def smooth_rewards_by_block(rewards, block_size=DEFAULT_SMOOTHING_BLOCK_SIZE):
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    smoothed_rewards = []
    episode_indices = []

    for start_index in range(0, len(rewards), block_size):
        reward_block = rewards[start_index : start_index + block_size]
        if not reward_block:
            continue

        smoothed_rewards.append(float(np.mean(reward_block)))
        episode_indices.append(start_index + len(reward_block))

    return episode_indices, smoothed_rewards


def save_q_table(q_table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as file:
        pickle.dump(dict(q_table), file)


def load_q_table(path):
    path = Path(path)

    with path.open("rb") as file:
        saved_q_table = pickle.load(file)

    q_table = create_q_table()
    for state, values in saved_q_table.items():
        q_table[state] = np.array(values, dtype=float)

    return q_table
