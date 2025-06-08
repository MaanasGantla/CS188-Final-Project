# Robosuite Square Assembly Imitation Learning

Default project for **CS 188: Introduction to Robotics** (Spring 2025)  
Imitation learning (Behavior Cloning) on the Robosuite **NutAssemblySquare** task.


## Table of Contents
- [Project Overview](#project-overview)
- [Creating Environment](#create-environment)
- [Training the Neural Network](#train-the-neural-network)
- [Test the Policy](#test-the-policy)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Project Overview

The goal is to learn a policy, using a low-dimensional state, that assembles a square nut onto a peg that 
can be in a variety of orientations and positions.

We gather 10D vectors containing the end effector's position (x, y, z), object's position (x,y,z), and the object's orientation
(quatx, quaty, quatz, quatw) for the N timesteps in the demo data. In addition, we gather the corresponding 7D action vectors for the N timesteps. This is done in extract_data.py.

Using two linear layers, we train our neural network with this data. This is done in deep_policy.py.

Finally, the end effector position and object positions/orientations are inputted into our model, outputting a 7D vector. This process is continued until actions are taken that lead to a success. This is done in deep_inference.py, and test_deep.py queries this policy for a number of trials to measure success.

## Creating Environment
1. Clone the repository.
2. Follow the [installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for conda.

Run the following commands from the root of the repository:
```
conda env create -f environment.yml
conda activate robosuite_arm
```

## Usage

### Train the Neural Network
```
python deep_policy.py
```


### Test the Policy
```
python test_deep.py
```


## Citation
If you find this work helpful and wish to cite it, here is the citation:

```bibtex
@inproceedings{
    maanas2025SquareNutAssemblyviaBehaviorCloning,
    title={Square Nut Assembly via Behavior Cloning},
    author={Maanas Gantla},
    year={2025},
    url={https://github.com/MaanasGantla/CS188-Final-Project/tree/main}
}
```



## License
This project is licensed under the MIT license. See the [LICENSE](License) file for details on the license.


## Contact
For any questions, please contact [Maanas Gantla](mailto:gantlamr@gmail.com)
