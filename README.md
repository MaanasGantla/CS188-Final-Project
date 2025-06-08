# Robosuite Square Assembly Imitation Learning

> Default‐project for **CS 188: Introduction to Robotics** (Spring 2025)  
> Imitation learning (Behavior Cloning) on the Robosuite *NutAssemblySquare* task.


## Overview

The goal is to learn a policy, using a low-dimensional state, that assembles a square nut onto a peg that 
can be in a variety of orientations and positions.

We gather 10D vectors containing the end effector's position (x, y, z), object's position (x,y,z), and the object's orientation
(quatx, quaty, quatz, quatw) for the N timesteps in the demo data. In addition, we gather the corresponding 7D action vectors for the N timesteps. This is done in extract_data.py.

Using two linear layers, we train our neural network with this data. This is done in deep_policy.py

Finally, the end effector position and object postiions/orientations are inputted to our model, outputting a 7D vector. This process is continued until actions are taken that lead to a success. This is done in deep_inference.py, and the test_deep.py queries this policy for a number of trials to measure success.

## Creating environment

```
conda env create -f environment.yml
conda activate robosuite_arm
```

## Train the Neural Network
```
python deep_policy.py
```


## Test the policy
```
python test_deep.py
```
