
# Shrinking Fractional Dimensions With Reinforcement Learning

This repo contains code to accompany the CORL 2020 paper: Explicitly Encouraging Low Fractional Dimensional Trajectories Via Reinforcement Learning. The root directory contains the original manuscript as a pdf, a short video, and the source code to replicate the results found in the paper.  

In this paper, we introduce a method to incorporate a measure of fractal dimensionality into the reward function of a reinforcement learning agent. This allows us to use any on-policy RL algorithm to search for policies which induce trajectories with a small fractal dimension. Agents which produce lower dimensional trajectories are more amendable to so called mesh based analysis (see follow up work here: https://github.com/sgillen/fractal_mesh), which is a step towards making empirical guarantees for RL agents. We also observed the resulting agents were more robust to push disturbances and noise. 

The source code contains a modified implementation of Augmented Random Search which modifies rewards obtained from reinforcement learning environments in order to explicitly encourage agents to find policies which induce trajectories with a small fractional dimension. There are also notebooks which analyze the resulting policies.
