
from lib import registerStatisticalRLenvironments,makeWorld,make
from demo import RandomAgent,animate
import numpy as np


def demo(envname):
     env = make(envname)
     print("-"*30+"\n"+env.name+"\n"+"-"*30)
     print("Available renderers: ",list(env.renderers.keys()))
     rendermode = np.random.choice(list(env.renderers.keys())[1:])
     if(envname=="random-100"):
         rendermode='text' #Force text rendering as others are farily slow.
     env.change_rendermode(rendermode)
     learner = RandomAgent(env)
     #learner = Human(env)
     #learner = UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
     animate(env, learner, 20)
     print("-"*30+"\n")

def print_registered_environments():
    print("-"*30+ "\nList of registered environments:\n"+ "-"*30)
    [print(k) for k in registerStatisticalRLenvironments.keys()]
    print("-"*30)

def random_environment():
    envname = np.random.choice(list(registerStatisticalRLenvironments.keys()))
    demo(envname)

def all_environments():
    for e in registerStatisticalRLenvironments:
        demo(e)

if __name__ == "__main__":

    print_registered_environments()
    random_environment()
    all_environments()



