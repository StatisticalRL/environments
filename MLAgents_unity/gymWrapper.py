import gymnasium

from gymnasium import Env, spaces
from gymnasium.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
#from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.base_env import (
    BaseEnv,
    DecisionSteps,
    TerminalSteps,
    BehaviorSpec,
    ActionTuple,
    BehaviorName,
    AgentId,
    BehaviorMapping,
)



# Defining your own Environment imported from Unity
###################################################

class UnityMDP(Env):

    def __init__(self, path):
        unity_env = UnityEnvironment(path)
        self.env = UnityToGymWrapper(unity_env, 0)

        states = self.env.observation_space
        actions= self.env.action_space
        #print("s:", states)
        #print(states.shape)
        #print(states.shape[0])
        #print("a:", actions)
        #print(actions.n)

        self.action_space = self.action_space_wrapper(self.env.action_space)# spaces.Discrete(actions.n)
        self.observation_space = self.state_space_wrapper(self.env.observation_space)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.np_random, seed = seeding.np_random(seed)
        s, info = self.env.reset()
        self.current_state= self.state_wrapper(s)
        return self.current_state, info

    def step(self,a):
        state,reward,isfinal,istruncated,info=self.env.step(self.action_wrapper(a))
        # May need to convert from state to one-hot encoding: Currently MLAgents_unity only accepts one-hot encoding of states!
        s = self.state_wrapper(state)
        self.current_state=s
        return s,reward,isfinal,istruncated,info


    def action_space_wrapper(self,action_space):
        # Assumes action_space is discrete
        #return spaces.Discrete(action_space.n)
        return action_space

    def action_wrapper(self,action):
        # Maps a python action to a MLAgents_unity action
        return action

    def state_space_wrapper(self,state_space):
        # The following assumes one-hot encoding of states:
        #nS = state_space.shape[0]
        #self.nS=nS
        #return spaces.Discrete(nS)
        return state_space

    def state_wrapper(self,state):
        # Discrete conversion assuming one hot encoding:
        #for n in range(self.nS):
        #   if (state[n]==1.):
        #        return n
        return state


# Example:
##########

class PlantMDP(UnityMDP):

    def __init__(self):
        UnityMDP.__init__(self, "./MLAgents_unity/PlantSingleAgent")

    def state_space_wrapper(self, state_space):
        self.nS=5 # Hard coding number of states. Ideally this parameter should be retrieved from Unity.
        return spaces.Discrete(self.nS)

    def state_wrapper(self, state):
        return (int) (state[1])

def plantMDPtest():
    u=PlantMDP()
    # States = [a b c} with a=plant height, b=water height, c=max tank capacity
    u.reset()
    print("START")
    for i in range(10):
       print("s:",u.current_state)
       a = u.action_space.sample()
       print("a:",a)
       state, reward, isfinal,istruncated, info = u.step(a)
       print("s':",state,"r:",reward,"terminal:",isfinal,"info",info)






# Alternative approach:
#######################

def DiscreteStateDiscreteAction():
    unity_env = UnityEnvironment("./MLAgents_unity/PlantSingleAgent")
    env = UnityToGymWrapper(unity_env,0)

    env.reset()
    for i in range(10):
        states = env.observation_space
        print(states)
        actions= env.action_space
        print(actions)
        a = actions.sample()
        print("a:",a)
        #print(states)
        #s=states.sample()
        #print("s:", len(s))
        #a = input("action?")
        state,reward,isfinal,info=env.step(a)
        print("S:",state)
        print("R:",reward)
        print("F:",isfinal)
        print("I:",info)

def VectorStateDiscreteAction():
    unity_env = UnityEnvironment("./MLAgents_unity/PushBlockSingleAgent")
    env = UnityToGymWrapper(unity_env,0)

    env.reset()
    paststate=np.zeros(210)
    for i in range(30):
        actions= env.action_space
        print(actions)
        a = actions.sample()
        print("a:",a)
        states= env.observation_space
        print(states)
        #s=states.sample()
        #print("s:", len(s))
        #a = input("action?")
        state,reward,isfinal,info=env.step(a)
        print("S:",state)
        print("R:",reward)
        print("F:",isfinal)
        print("I:",info)
        deltastates=state-paststate
        print("D:",deltastates)
        paststate = state


def VectorStateVectorAction():
    unity_env = UnityEnvironment("./MLAgents_unity/3DBallSingleAgent")
    env = UnityToGymWrapper(unity_env,0)

    env.reset()
    for i in range(30):
        actions= env.action_space
        print(actions)
        a = actions.sample()
        print("a:",a)
        #a = input("action?")
        state,reward,isfinal,info=env.step(a)
        print("S:",state)
        print("R:",reward)
        print("F:",isfinal)
        print("I:",info)


def ImageStateDiscreteAction():
    unity_env = UnityEnvironment("./MLAgents_unity/gridworldsingleagent")
    env = UnityToGymWrapper(unity_env,0)

    env.reset()
    for i in range(20):
        actions= env.action_space
        print(actions)
        a = actions.sample()
        print("a:",a)
        a = input("action?")
        state,reward,isfinal,info=env.step(a)
        #We get the state as an image from the Unity camera !
        #print(len(state),len(state[0]),len(state[0][0]))
        # If output is image:
        plt.figure(1)
        plt.imshow(state)
        plt.show()
        plt.pause(0.01)

        print("R:",reward)
        print("F:",isfinal)
        print("I:",info)

        #env.render()

#Testing different environments:
################################
#DiscreteStateDiscreteAction()
VectorStateDiscreteAction()
#VectorStateVectorAction()
#ImageStateDiscreteAction()
#plantMDPtest()
