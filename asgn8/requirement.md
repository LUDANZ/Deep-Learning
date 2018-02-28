# Assignment 8: Advantage Actor-Critic

## Goal:
We will be implementing the Advantage Actor-Critic (A2C) Algorithm for the Cartpole-v1 (note difference from HW 7) task in OpenAI Gym. For this assignment, you will need to install the OpenAI Gym library (locally, it's already installed in the course virtual environment).

We recommend that during development, you render the OpenAI gym environment, every time you perform an action. While this significantly slows down your train time, it is good to see how your model is learning to perform a given task. When you turn your assignment in however, turn rendering off.

## Data:
All data for this assignment will be collected interactively, via the CartPole-v1 environment of OpenAI Gym.

## A2C Steps:
Set up placeholders for inputs and outputs.
Initialize parameters for the model (however you like!)
Implement the forward pass for the model:
Feed-forward layers, into final policy over 2 actions.
Feed-forward layers, into final state value estimate (initial layers can be shared with policy network).
Train your model (details below).
Collect data interactively, by acting via your policy network.
Use a discount factor (gamma) of 0.99.
Use an Adam Optimizer with a learning rate of 0.001.
Update your parameters at the end of each game. (This instruction has been modified, if you update every 50 steps as previously advised no points will be deducted.)
Print the mean reward collected over the last 100 episodes, for each trial.

## Notes:
Run 3 different trials, each with 1000 episodes. On the last line of your input, print the mean reward collected for the last 100 episodes of each trial.
