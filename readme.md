# Reinforcement Learning Project

This project implements a reinforcement learning agent using Deep Q-Network (DQN) to navigate in a grid-based environment.

## Overview

The project consists of several Python files that contribute to the reinforcement learning process:

- `environment.py`: Defines the grid-based environment where the agent interacts.
- `agent.py`: Contains the implementation of the reinforcement learning agent using DQN.
- `experience_replay.py`: Implements the experience replay buffer for the agent's training.
- `test.py`: Orchestrates the training process of the agent in the defined environment.

## Files

### 1. environment.py

This file defines the environment in which the agent operates:

- Creates a grid-based environment with an agent and a goal.
- Allows the agent to move within the grid, collect rewards, and reach the goal.
- Provides methods to reset, render, and interact with the environment.

### 2. agent.py

Contains the definition of the reinforcement learning agent:

- Builds a neural network model (DQN) to approximate Q-values.
- Implements methods for action selection, learning from experiences, and saving/loading the model.

### 3. experience_replay.py

Implements the experience replay buffer:

- Stores experiences (state, action, reward, next_state, done) for training.
- Provides methods to add experiences, sample batches, and check availability for training.

### 4. test.py

Orchestrates the training process of the agent:

- Initializes the environment, agent, and experience replay.
- Runs a training loop for a specified number of episodes.
- Saves the trained model periodically during training.

## Usage

To run the project:

1. Ensure you have Python and Tensorflow installed.
2. Clone the repository and navigate to the project directory.
3. Run `python test.py` to start training the agent.

