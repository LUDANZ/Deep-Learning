# Assignment 7: REINFORCE

## Goal:
We will be implementing the REINFORCE Algorithm for the Cartpole-v0 task in OpenAI Gym. For this assignment, you will need to install the OpenAI Gym library (locally, it's already installed in the course virtual environment).

We recommend that during development, you render the OpenAI gym environment, every time you perform an action. While this significantly slows down your train time, it is good to see how your model is learning to perform a given task. When you turn your assignment in however, turn rendering off.

## Data:
All data for this assignment will be collected interactively, via the CartPole-v0 environment of OpenAI Gym.

## REINFORCE Steps:
Set up placeholders for inputs and outputs.
Initialize parameters for the model (however you like!)
Implement the forward pass for the model:
Feed-forward layers, into final policy over 2 actions.
Train your model (details below).
Collect data interactively, by acting via your policy network.
Use a discount factor (gamma) of 0.9999.
Use an Adam Optimizer with a learning rate of 0.005.
Print the mean reward collected over the last 100 episodes, for each trial.
Notes:
Run 3 different trials, each with 1000 episodes. On the last line of your input, print the mean reward collected for the last 100 episodes of each trial.
