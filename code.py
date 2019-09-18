import gym
import random
import numpy as np

# Load and check the environment from OpenAI Gym
env = gym.make("Taxi-v2")
env.render()

action_dim = env.action_space.n
state_dim = env.observation_space.n
print(f'Action Dimension: {action_dim} \n State Dimension: {state_dim}')

# Hyperparameters
learning_rate = 0.7
gamma = 0.6
decay_rate = 0.01


def q_training(total_episodes=50000, max_steps=99, epsilon=1.0):
    """Iteratively construct Q table (agent training)

    Args:
        total_episodes (int, optional): Number of episodes for training. Defaults to 50000.
        max_steps (int, optional): Number of steps per episode. Defaults to 99.
        epsilon (float, optional): Initial threshold for exploiting solution space. Defaults to 1.0.

    Returns:
        numpy.ndarray: Trained Q table for each (state, action) pair
    """

    # Initialize Q table with zeros
    q_table = np.zeros((state_dim, action_dim))

    max_epsilon, min_epislon = (1.0, 0.01)

    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False

        for step in range(max_steps):
            rand_value = random.uniform(0, 1)
            if rand_value > epsilon:
                action = np.argmax(q_table[state, :])
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
            state = new_state

            if done:
                break

            epsilon = min_epislon + \
                (max_epsilon - min_epislon) * np.exp(-decay_rate * episode)

    return q_table


def q_testing(q_table, total_test_episodes=100, max_steps=99):
    """Tests the performance of the agent following the policy given by the Q table

    Args:
        q_table (numpy.ndarray): Q table describing the optimal policy for the agent
        total_test_episodes (int, optional): Number of episodes to test on. Defaults to 100.
        max_steps (int, optional): Maximum steps in each episode. Defaults to 99.
    """

    env.reset()
    rewards = []
    frames = []

    for episode in range(total_test_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(q_table[state, :])

            new_state, reward, done, info = env.step(action)

            total_rewards += reward

            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
            })

            if done:
                rewards.append(total_rewards)
                # print ("Score", total_rewards)
                break
            state = new_state
    env.close()
    print("Mean Score: " + str(sum(rewards) / total_test_episodes))


qt = q_training()
q_testing(q_table=qt)
