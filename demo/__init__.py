
class RandomAgent:
    def __init__(self,env):
        self.env=env

    def name(self):
        return "Random Agent"

    def reset(self,inistate):
        ()

    def play(self,state):
        return self.env.action_space.sample()

    def update(self, state, action, reward, observation):
        ()



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



