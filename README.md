# Statistical Reinforcement Learning Environments
Environments for Statistical Reinforcement Learning

## Installation

    pip install statisticalRL-environments


## Test
    python fulldemo.py

# List of environments:
    
    MABS:
        mab-bernoulli
    MDPs:
        random-rich
        ergodic-random-rich
        random-12
        random-small
        random-small-sparse
        random-100
        three-state
        nasty
        river-swim-6
        ergo-river-swim-6
        ergo-river-swim-25
        river-swim-25
    GRIDWORLD MDPs:
        grid-random-1616
        grid-random-1212
        grid-random-88
        grid-2-room
        grid-4-room

The library contains several stochastic MDPs, that is for which transition and reward functions are not deterministic.

## Randomly generated environments:
Furthermore, the library supports random generations of MDPs.
The environments containing "random" in their name are randomly generated MDPs, 
that is with randomly generated transition and reward (stochastic) functions.
For reproducibility reasons, they are generated with a given seed.

You can modify them in the registration list "registerStatisticalRLenvironments" available in the init file of lib.
In this list, you will find environments registered with specific parameters, including the (last) parameter "seed" which is used to generate the random transition and reward functions.
    
    "random-small" : lambda x: registerRandomMDP(nbStates=3, nbActions=4, maxProportionSupportTransition=0.5, maxProportionSupportReward=0.4, maxProportionSupportStart=0.1, minNonZeroProbability=0.15, minNonZeroReward=0.25, rewardStd=0.1,seed=5),

Note: fixing this seed  (here sedd=5) does not prevent the transitions/rewards to be stochastic, as they are using another random number generating process.
# Rendering

Each type of environment comes with different renderers, 
including the null renderer that displays nothing.


## Text rendering:

 This rendering is available for all environment types.


### MAB Text rendering:

![alt text](https://raw.githubusercontent.com/StatisticalRL/environments/main/media/screenshots/TextRendering.png)


### MDP Text rendering:

![alt text](https://raw.githubusercontent.com/StatisticalRL/environments/main/media/screenshots/TextRendering.png)

### Gridworld-MDP Text rendering
![alt text](https://raw.githubusercontent.com/StatisticalRL/environments/main/media/screenshots/TextRenderingGridWorld.png)

# Graph rendering:

This rendering is available for MDPs. 
On top of the visual display, it captures a screenshot png at each time step. 
This may be slow for large MDPs.

![alt text](https://raw.githubusercontent.com/StatisticalRL/environments/main/media/videos/3states.gif)

![alt text](https://raw.githubusercontent.com/StatisticalRL/environments/main/media/videos/riverswim2.gif)

![alt text](https://raw.githubusercontent.com/StatisticalRL/environments/main/media/videos/graph.gif)


### Grid-world rendering:

This rendering is available for Gridworld-MDPs.
On top of the visual display, it captures a screenshot png at each time step.

![alt text](https://raw.githubusercontent.com/StatisticalRL/environments/main/media/videos/gridworld.gif)


![alt text](https://raw.githubusercontent.com/StatisticalRL/environments/main/media/videos/gridworldbig.gif)

