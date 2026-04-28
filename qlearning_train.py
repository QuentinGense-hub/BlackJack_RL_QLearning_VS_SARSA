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
    done = False
    total_reward = 0

    while not done:
        action = choose_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        best_next_action = np.max(Q[next_state])

        Q[state][action] += alpha * (
            reward + gamma * best_next_action - Q[state][action]
        )

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

print("Entraînement terminé")
print("Reward moyen :", np.mean(rewards[-10_000:]))