import gymnasium
from gymnasium.envs.registration import  register
import numpy as np
import sys
INFINITY=sys.maxsize


"""
This module provides a structured and documented interface for registering
a collection of Multi-Armed Bandits (MABs), Markov Decision Processes (MDPs),
and benchmark reinforcement learning environments into Gymnasium.

It wraps `gymnasium.envs.registration.register` with higher-level, reproducible
APIs and ensures consistent naming, parameter handling, and configuration.

Name of src should not contain symbol '_'. 

"""
### MABS

def registerBernBandit(means,  max_steps=INFINITY,  reward_threshold=1.):
    name = 'MAB-Bernoulli-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.MABs.envs.StochasticBandits:BernoulliBandit',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'means': means, 'name':name }
    )
    return name


def registerGaussBandit(means,  vars, max_steps=INFINITY,  reward_threshold=1.):
    name = 'MAB-Gaussian-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.MABs.envs.StochasticBandits:GaussianBandit',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'means': means, 'vars': vars, 'name':name }
    )
    return name


def registerBinomialBandit(means,  repetitions, max_steps=INFINITY,  reward_threshold=1.):
    name = 'MAB-Binomial-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.MABs.envs.StochasticBandits:BinomialBandit',
        max_episode_steps=max_steps,
        reward_threshold=repetitions,
        kwargs={'means': means, 'repetitions': repetitions, 'name':name }
    )
    return name

def registerBatchQBinMAB(means,batchsize,quantization_range,repetitions):
    s="-".join(str(m) for m in means)
    name = f'Binomial200-batch{batchsize}-tol{quantization_range}-means-{s}-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.MABs.wrappers:BatchQBinMAB',
        max_episode_steps=INFINITY,
        reward_threshold=1.,
        kwargs={'probabilities':means,'batchsize':lambda x:batchsize,'quantization_range':quantization_range,'repetitions':repetitions,'name':name }
    )
    return name

### Batch Bandits
def registerBatchBandit(probabilities, batchsize_function, repetitions):
    action_names = ["a" + str(i) for i in range(len(probabilities))]
    s = "-".join(str(m) for m in probabilities)
    name = f'Batch-Binomial200-means-{s}-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.BatchMABs.envs.batchmabs:BatchBandit',
        max_episode_steps=INFINITY,
        reward_threshold=1.,
        kwargs={'action_names': action_names, 'probabilities': probabilities,
                'batchsize': batchsize_function, 'repetitions': repetitions, 'name': name}
    )
    return name


def registerBatchGBandit(probabilities, batchsize_function, variance):
    action_names = ["a" + str(i) for i in range(len(probabilities))]
    s = "-".join(str(m) for m in probabilities)
    name = f'Batch-Gaussian{variance}-means-{s}-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.BatchMABs.envs.batchmabs:BatchGBandit',
        max_episode_steps=INFINITY,
        reward_threshold=1.,
        kwargs={'action_names': action_names, 'probabilities': probabilities,
                'batchsize': batchsize_function, 'variance': variance, 'name': name}
    )
    return name


def registerBatchTruncGBandit(probabilities, batchsize_function, sigma=0.5, low=-1.0, high=1.0):
    """Register a bandit with TruncatedGaussian arms (support [low, high]).

    Parameters
    ----------
    probabilities : list of float
        Desired means of the *untruncated* Gaussians (location parameters).
        The true means of the truncated arms (stored in env.means) will differ
        slightly; use sigma small relative to the gap to [low, high].
    sigma : float
        Standard deviation of the underlying Gaussian (default 0.5).
    """
    action_names = ["a" + str(i) for i in range(len(probabilities))]
    s = "-".join(str(m) for m in probabilities)
    name = f'Batch-TruncGaussian-sigma{sigma}-means-{s}-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.BatchMABs.envs.batchmabs:BatchTruncGBandit',
        max_episode_steps=INFINITY,
        reward_threshold=1.,
        kwargs={'action_names': action_names, 'probabilities': probabilities,
                'batchsize': batchsize_function, 'sigma': sigma,
                'low': low, 'high': high, 'name': name}
    )
    return name


def registerBatchBernBandit(probabilities, batchsize_function):
    """Register a bandit with Bernoulli arms (support [0, 1])."""
    action_names = ["a" + str(i) for i in range(len(probabilities))]
    s = "-".join(str(m) for m in probabilities)
    name = f'Batch-Bernoulli-means-{s}-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.BatchMABs.envs.batchmabs:BatchBernBandit',
        max_episode_steps=INFINITY,
        reward_threshold=1.,
        kwargs={'action_names': action_names, 'probabilities': probabilities,
                'batchsize': batchsize_function, 'name': name}
    )
    return name

### MDPs
def registerRandomMDP(nbStates=5, nbActions=4, max_steps=INFINITY, reward_threshold=np.inf, maxProportionSupportTransition=0.5, maxProportionSupportReward=0.1, maxProportionSupportStart=0.2, minNonZeroProbability=0.2, minNonZeroReward=0.3, rewardStd=0.5, ergodic=0.,seed=None):
    name = 'RandomMDP-S'+str(nbStates)+'_A'+str(nbActions)+'_s'+str(seed)+'-v0'
    if(ergodic>0):
        name = 'ErgodicRandomMDP-S' + str(nbStates) + '_A' + str(nbActions) + '_s' + str(seed) + '-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.MDPs_discrete.envs.RandomMDP:RandomMDP',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'nbActions': nbActions, 'maxProportionSupportTransition': maxProportionSupportTransition, 'maxProportionSupportReward': maxProportionSupportReward,
                'maxProportionSupportStart': maxProportionSupportStart, 'minNonZeroProbability':minNonZeroProbability, 'minNonZeroReward':minNonZeroReward, 'rewardStd':rewardStd, 'ergodic':ergodic, 'seed':seed, 'name':name }
    )
    return name

def registerRiverSwim(nbStates=5, max_steps=INFINITY, reward_threshold=np.inf, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):
    name = 'RiverSwim-S'+str(nbStates)+'-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.MDPs_discrete.envs.RiverSwim:RiverSwim',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'rightProbaright': rightProbaright, 'rightProbaLeft': rightProbaLeft,
                'rewardL': rewardL, 'rewardR':rewardR, 'name':name }
    )
    return name


def registerErgodicRiverSwim(nbStates=5, max_steps=INFINITY, reward_threshold=np.inf, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1., ergodic=0.001):
    name = 'ErgodicRiverSwim-S'+str(nbStates)+'-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.MDPs_discrete.envs.RiverSwim:ErgodicRiverSwim',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'rightProbaright': rightProbaright, 'rightProbaLeft': rightProbaLeft,
                'rewardL': rewardL, 'rewardR':rewardR, 'ergodic': ergodic, 'name':name }
    )
    return name


def registerGridworld(sizeX=10, sizeY=10, map_name="4-room", rewardStd=0., initialSingleStateDistribution=False, max_steps=INFINITY, reward_threshold=np.inf, start=None, goal=None, seed=0):
    name ='Gridworld-'+map_name+'-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.MDPs_discrete.envs.GridWorld.gridworld:GridWorld',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'sizeX': sizeX,'sizeY':sizeY,'map_name':map_name,'rewardStd':rewardStd, 'initialSingleStateDistribution':initialSingleStateDistribution,'start':start, 'goal':goal, 'seed':seed, 'name':name}
    )
    return name


def registerRandomGridworld(sizeX=10, sizeY=10,rewardStd=0., initialSingleStateDistribution=False, max_steps=INFINITY, reward_threshold=np.inf, density=0.2, seed=0):
    name ='RandomGridworld-'+str(sizeX)+'x'+str(sizeY)+'_s'+str(seed)+'-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.MDPs_discrete.envs.GridWorld.gridworld:GridWorld',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'sizeX': sizeX,'sizeY':sizeY,'map_name':'random','rewardStd':rewardStd, 'initialSingleStateDistribution':initialSingleStateDistribution,'start':None, 'goal':None, 'name':name, 'seed':seed, 'density':density}
    )
    return name


def registerThreeState(delta = 0.005, max_steps=INFINITY, reward_threshold=np.inf, fixed_reward = True):
    name = 'ThreeState-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.MDPs_discrete.envs.ThreeStateMDP:ThreeState',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'delta': delta, 'fixed_reward': fixed_reward,'name':name }
    )
    return name


def registerNasty(delta = 0.005, epsilon=0.05, max_steps=INFINITY, reward_threshold=np.inf, fixed_reward = True):
    name = 'Nasty-v0'
    register(
        id=name,
        entry_point='statisticalrl_environments.MDPs_discrete.envs.ThreeStateMDP:Nasty',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'delta': delta, 'epsilon': epsilon, 'fixed_reward': fixed_reward,'name':name }
    )
    return name



import math
# ---------------------------------------------------------------------------
# Shared arm means: 4 arms, means from 0.1 to 0.9
# ---------------------------------------------------------------------------
#_MEANS = [0.1, 0.4, 0.7, 0.9]
_MEANS = [0.2, 0.6, 0.8, 0.8, 0.95, 0.9]

# Batch-size schedules — ell is the round index (0-based)
_B_CONST     = lambda ell: 10
_B_LINEAR    = lambda ell: int(ell + 1)
_B_QUADRATIC = lambda ell: int((ell + 1) ** 2)
_B_CUBIC     = lambda ell: int((ell + 1) ** 3)
_B_EXP       = lambda ell: int(2 ** ell)
_B_DOUBLE_EXP = lambda ell: int(math.exp(2 ** ell))
_B_ABRUPT = lambda ell: 100 if ell<2 else int((ell+1)**3)

def exotic_schedule1(t):
    schedule= {0:_B_CONST, 1: _B_LINEAR, 2: _B_CUBIC, 3: _B_EXP}
    return schedule[(t % 4)](t)
def exotic_schedule2(t):
    schedule= {0:_B_CONST, 1: _B_EXP, 2:_B_LINEAR, 3:_B_CUBIC}
    return schedule[(t % 4)](t)

_B_EXOTIC1 = exotic_schedule1
_B_EXOTIC2 = exotic_schedule2

registerStatisticalRLenvironments = {
    "mab-bernoulli": lambda x: registerBernBandit(means=[0.2, 0.8, 0.3, 0.7]),
    "mab-gaussian": lambda x: registerGaussBandit(means=[0.2, 0.8, 0.3, 0.7],vars=[1.,1.,1.,1.]),
    "mab-binomial": lambda x: registerBinomialBandit(means=[0.2, 0.8, 0.3, 0.7], repetitions=200),
   # "mab-batch-quantized": lambda x: registerBatchQBinMAB(means=[0.1, 0.15, 0.2, 0.75, 0.85, 0.9], batchsize=5,
   #                                                    quantization_range=15, repetitions=200),
    "random-rich": lambda x: registerRandomMDP(nbStates=10, nbActions=4, maxProportionSupportTransition=0.12,
                                            maxProportionSupportReward=0.8, maxProportionSupportStart=0.1,
                                            minNonZeroProbability=0.15, minNonZeroReward=0.4, rewardStd=0, seed=10),

    "ergodic-random-rich": lambda x: registerRandomMDP(nbStates=10, nbActions=4, maxProportionSupportTransition=0.12,
                                               maxProportionSupportReward=0.8, maxProportionSupportStart=0.1,
                                               minNonZeroProbability=0.15, minNonZeroReward=0.4, rewardStd=0, ergodic=0.01, seed=10),
    "random-12" : lambda x: registerRandomMDP(nbStates=12, nbActions=2, maxProportionSupportTransition=0.15, maxProportionSupportReward=0.25, maxProportionSupportStart=0.1, minNonZeroProbability=0.15, minNonZeroReward=0.3, rewardStd=0.1,seed=6),
    "random-small" : lambda x: registerRandomMDP(nbStates=3, nbActions=4, maxProportionSupportTransition=0.5, maxProportionSupportReward=0.4, maxProportionSupportStart=0.1, minNonZeroProbability=0.15, minNonZeroReward=0.25, rewardStd=0.1,seed=5),
    "random-small-sparse" : lambda x: registerRandomMDP(nbStates=4, nbActions=4, maxProportionSupportTransition=0.5, maxProportionSupportReward=0.08, maxProportionSupportStart=0.1, minNonZeroProbability=0.1, minNonZeroReward=0.2, rewardStd=0.1,seed=2),
    "random-100" : lambda x: registerRandomMDP(nbStates=100, nbActions=3, maxProportionSupportTransition=0.1, maxProportionSupportReward=0.1, maxProportionSupportStart=0.1, minNonZeroProbability=0.15, minNonZeroReward=0.3, rewardStd=0.1,seed=10),
    "three-state" : lambda x: registerThreeState(delta = 0.005),
    "nasty": lambda x: registerNasty(delta=0.005,epsilon=0.05),
    "river-swim-6" : lambda x: registerRiverSwim(nbStates=6, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.),
    "ergo-river-swim-6": lambda x: registerErgodicRiverSwim(nbStates=6, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005,
                                                rewardR=0.99, ergodic=0.001),
    "ergo-river-swim-25": lambda x: registerErgodicRiverSwim(nbStates=25, rightProbaright=0.4, rightProbaLeft=0.05,
                                                            rewardL=0.005,
                                                            rewardR=0.99, ergodic=0.001),
    "river-swim-25" : lambda x: registerRiverSwim(nbStates=25, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=0.99),
    "grid-random-1616" : lambda x: registerRandomGridworld(sizeX=16, sizeY=16, rewardStd=0.01, initialSingleStateDistribution=False, density=0.4, seed=7),
    "grid-random-1212" : lambda x: registerRandomGridworld(sizeX=12, sizeY=12, rewardStd=0.01, initialSingleStateDistribution=False, density=0.4, seed=5),
    "grid-random-88" : lambda x: registerRandomGridworld(sizeX=8, sizeY=8, rewardStd=0.01, initialSingleStateDistribution=False, density=0.4, seed=1),#16 16 7# 12 12 5 # 8 8 1
    "grid-2-room" : lambda x: registerGridworld(sizeX=9, sizeY=11, map_name="2-room", rewardStd=0.0, initialSingleStateDistribution=True,start=[1,1],goal=[7,9],seed=1),
    "grid-4-room" : lambda x: registerGridworld(sizeX=7, sizeY=7, map_name="4-room", rewardStd=0.0, initialSingleStateDistribution=True,start=[1,1],goal=[5,5], seed=1),


    # -- Gaussian (non-truncated), legacy --
    "constant-batch": lambda x: registerBatchGBandit([0.3, 0.5, 0.7, 0.9], lambda x: 5, variance=0.1),
    "linear-batch":   lambda x: registerBatchGBandit([0.3, 0.5, 0.7, 0.9], lambda x: int(2*(x+1)), variance=0.1),
    "cubic-batch":    lambda x: registerBatchGBandit([0.3, 0.5, 0.7, 0.9], lambda x: int(2*((x+1)**3)), variance=0.1),
    "exp-batch":      lambda x: registerBatchGBandit([0.3, 0.5, 0.7, 0.9], lambda x: int(2*(2**x)), variance=0.1),

    # -- TruncatedGaussian [-1, 1], sigma=0.5, means in [0.1, 0.9] --
    # upper bound for KLinf = 1.0 (high)
    "trunc-constant-batch":   lambda x: registerBatchTruncGBandit(_MEANS, _B_CONST),
    "trunc-linear-batch":     lambda x: registerBatchTruncGBandit(_MEANS, _B_LINEAR),
    "trunc-quadratic-batch":  lambda x: registerBatchTruncGBandit(_MEANS, _B_QUADRATIC),
    "trunc-cubic-batch":      lambda x: registerBatchTruncGBandit(_MEANS, _B_CUBIC),
    "trunc-exp-batch":        lambda x: registerBatchTruncGBandit(_MEANS, _B_EXP),
    "trunc-double-exp-batch": lambda x: registerBatchTruncGBandit(_MEANS, _B_DOUBLE_EXP),

    # -- Bernoulli, means in [0.1, 0.9], upper bound = 1.0 --
    "bern-constant-batch":   lambda x: registerBatchBernBandit(_MEANS, _B_CONST),
    "bern-linear-batch":     lambda x: registerBatchBernBandit(_MEANS, _B_LINEAR),
    "bern-quadratic-batch":  lambda x: registerBatchBernBandit(_MEANS, _B_QUADRATIC),
    "bern-cubic-batch":      lambda x: registerBatchBernBandit(_MEANS, _B_CUBIC),
    "bern-exp-batch":        lambda x: registerBatchBernBandit(_MEANS, _B_EXP),
    "bern-double-exp-batch": lambda x: registerBatchBernBandit(_MEANS, _B_DOUBLE_EXP),
    "bern-abrupt-batch": lambda x: registerBatchBernBandit(_MEANS, _B_ABRUPT),
    "bern-exotic1-batch": lambda x: registerBatchBernBandit(_MEANS, _B_EXOTIC1),
    "bern-exotic2-batch": lambda x: registerBatchBernBandit(_MEANS, _B_EXOTIC2)
}


def print_envlist():
    print("-"*30)
    print("List of registered environments: ")
    for k in registerStatisticalRLenvironments.keys():
        print("\t"+k)
    print("-"*30)

def register_env(envName):
    if (envName in registerStatisticalRLenvironments.keys()):
        regName = (registerStatisticalRLenvironments[envName])(0)
        print("[REGISTER.INFO] Environment " + envName + " registered as " + regName)
        return regName
    else:
        return envName

def makeWorld(registername):
    """

    :param registername: name of the environment to be registered into gym
    :return:  full name of the registered environment
    """
    return gymnasium.make(registername).unwrapped

def make(envName):
    return makeWorld(register_env(envName))