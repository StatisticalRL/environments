#from experiments.runExperiments import *
import src as bW
import numpy as np

#################################
# Running a single experiment:
#################################


def animate(env, learner, timeHorizon):
    print("Render mode:", str(env.rendermode))
    print("New initialization of ", learner.name())
    observation, info = env.reset()
    print("Initial state:" + str(observation))
    #env.render()
    learner.reset(observation)
    cumreward = 0.
    cumrewards = []
    for t in range(timeHorizon):
        state = observation
        env.render()
        action = learner.play(state)  # Get action
        observation, reward, done, truncated,  info = env.step(action)
        # print("S:"+str(state)+" A:"+str(action) + " R:"+str(reward)+" S:"+str(observation) +" done:"+str(done) +"\n")
        learner.update(state, action, reward, observation)  # Update learners
        cumreward += reward
        cumrewards.append(cumreward)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break


def demo_riverSwim():
     testName = 'ergo-river-swim-6'
     envName = (bW.registerStatisticalRLenvironments[testName])(0)
     env = bW.makeWorld(envName)
     rendermode = np.random.choice(list(env.renderers.keys()))
     print("Choice of renderer:", rendermode, " in ", list(env.renderers.keys()))
     env.change_rendermode(rendermode)
     learner = RandomAgent(env)
     #learner = lh.Human(env)
     # learner = le.UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
     #print(env.renderers.keys(),env.rendermode)
     animate(env, learner, 100)
     #animate(env, learner, 100, 'text')
     #animate(env, learner, 100, 'text')


def demo_randomGrid():
     testName = 'grid-random-88' #TODO: numpy randint has recent changed not yet propagated.
     #testName = 'grid-2-room'

     envName = (bW.registerStatisticalRLenvironments[testName])(0)
     env = bW.makeWorld(envName)
     rendermode = np.random.choice(list(env.renderers.keys()))
     print("Choice of renderer:", rendermode, " in ", list(env.renderers.keys()))
     #env.change_rendermode('gw-pyplot')
     env.change_rendermode(rendermode)
     learner = RandomAgent(env)
     #learner = bl.PSRL(env.observation_space.n, env.action_space.n, delta=0.05)
     #learner = lh.Human(env)
     # learner = le.UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
     #animate(env, learner, 100, 'text')
     animate(env, learner, 100)

def demo_randomMDP():
    testName = 'random-12'
    envName = (bW.registerStatisticalRLenvironments[testName])(0)
    env = bW.makeWorld(envName)
    rendermode = np.random.choice(list(env.renderers.keys()))
    print("Choice of renderer:", rendermode, " in ", list(env.renderers.keys()))
    #env.change_rendermode('networkx')
    #env.rendermode = 'networkx'
    #env.env.rendermode = 'text'
    #env.env.rendermode = 'networkx'
    #env.env.rendermode = 'pydot'
    env.change_rendermode(rendermode)
    learner = RandomAgent(env)
    # learner = le.UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
    animate(env, learner, 5)
    #animate(env, learner, 50, 'text')
    #
    #
    # testName = 'random100'
    # envName = (bW.registerWorlds[testName])(0)
    # env = bW.makeWorld(envName)
    # learner = lr.Random(env)
    # # learner = le.UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
    # animate(env, learner, 100, 'text')

#demo_riverSwim()
#demo_randomGrid()
demo_randomMDP()