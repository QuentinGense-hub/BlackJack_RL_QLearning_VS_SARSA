import gymnasium as gym
import numpy as np
from collections import defaultdict

env = gym.make("Blackjack-v1", sab=True)

Q = defaultdict(lambda: np.zeros(env.action_space.n))

alpha = 0.01
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999995

n_episodes = 500_000

def choose_action(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])

rewards = []

for episode in range(n_episodes):
    state, info = env.reset()
    action = choose_action(state)

    done = False
    total_reward = 0

    while not done:
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            target = reward
        else:
            next_action = choose_action(next_state)
            target = reward + gamma * Q[next_state][next_action]

        Q[state][action] += alpha * (target - Q[state][action])

        if not done:
            state = next_state
            action = next_action

        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

def evaluate_agent(env, Q, n_games=100_000):
    wins = losses = draws = 0

    for _ in range(n_games):
        state, info = env.reset()
        done = False

        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins / n_games,
        "losses": losses / n_games,
        "draws": draws / n_games
    }

print("Entraînement SARSA terminé")
print("Reward moyen :", np.mean(rewards[-10_000:]))
print(evaluate_agent(env, Q))