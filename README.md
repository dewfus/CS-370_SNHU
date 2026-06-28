**CS-370: Current and Emerging Trends in Computer Science**

# Pirate Intelligent Agent (Deep Q-Learning)

This project builds an intelligent agent, a pirate NPC, that learns to find its
way through an 8x8 maze to a treasure cell using deep Q-learning. Instead of
following a hard-coded route, the agent figures out its own way through by
playing the game thousands of times and adjusting a neural network that
estimates how good each move is.

**Built with:** Python, TensorFlow, Keras, NumPy, Jupyter Notebook

## Project Overview

The setup is an 8x8 grid of open and blocked cells. The pirate starts in the
top-left corner, the treasure sits in the bottom-right, and the agent can move
left, up, right, or down. Every move pays out a reward: a small penalty for a
normal step, bigger penalties for hitting a wall or walking off the grid, and a
big reward for reaching the treasure. Train it long enough and it learns to
favor the moves that head toward the treasure. By the end it hits a 100% win
rate, which means it can solve the maze starting from any open cell.

## What I Was Given vs. What I Built

**Provided as starter code:**

- `TreasureMaze.py` — the environment, including the maze grid, state tracking,
  the reward rules, valid-move checks, and the reset/act/observe methods.
- `GameExperience.py` — the experience replay class that holds onto past moves
  so the agent can learn from them.
- Notebook helpers — `build_model`, which defines the neural network, plus
  `play_game`, `completion_check`, the maze itself, and the visualization
  function.

**What I built myself:**

- The **Q-Training Algorithm** (`qtrain`), which is the deep Q-learning loop and
  the heart of the project. Each epoch it drops the pirate on a random open
  cell, plays a full game with an epsilon-greedy policy (the balance between
  exploring and going with what it already knows), and saves every move into the
  replay buffer. Then it pulls a mini-batch of past moves, works out the target
  Q-values with the Bellman equation, and nudges the network with a custom
  training step.
- An **epsilon schedule** that starts at 1.0 (all exploration) and decays down
  to a floor of 0.05, so the agent commits to its strategy once it is actually
  good at the game.
- A **target network** that I sync every so often to keep training stable, so
  the learning targets are not moving on every single step.
- A **stopping rule** that ends training once the win rate and the completion
  check agree the agent can win from every open cell. Mine got there at epoch
  313.

## Reflection

### What do computer scientists do, and why does it matter?

Computer scientists take messy, open-ended real problems and turn them into
something a machine can actually solve. That means pinning down the problem,
picking the right algorithm, and building something that works reliably and
scales. Here, "find the treasure before the player does" became a reinforcement
learning problem with clear states, actions, and rewards. It matters because
that same way of thinking sits under a huge amount of the software people use
every day, from maps and recommendations to the automated decisions that quietly
run a lot of how organizations work.

### How do I approach a problem as a computer scientist?

I start by figuring out the problem before I write any code: what goes in, what
the goal is, what the limits are, and how I will know it is working. Then I break
it into smaller pieces and reach for techniques that already exist instead of
reinventing them, which is exactly why knowing patterns like Q-learning and
experience replay pays off. I build a little at a time and test constantly. This
project was a good example, going from a milestone version to a final one,
running the model, watching the win rate, and adjusting hyperparameters based on
what actually happened instead of what I assumed would.

### What are my ethical responsibilities to the end user and the organization?

For the end user, my job is to build something reliable, honest about what it
cannot do, and respectful of privacy, and to not cause harm through sloppy
design. That is a bigger deal with machine learning than people expect, because
an agent optimizes exactly what you reward it for, shortcuts and all, so how you
design the rewards and how you test are ethical questions, not just technical
ones. For the organization, I owe clean, well-documented work other people can
build on, honest claims about what a model does and does not do, and care with
how data gets handled. Being upfront about how a system actually behaves protects
both the people using it and the company putting it out there.
