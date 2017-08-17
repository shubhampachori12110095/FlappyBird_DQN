import DQN_Brain
import gym
import gym_ple

env = gym.make("FlappyBird-v0")
DQN_play = DQN_Brain.Experience_Replay(env = env)
DQN_play.start_the_trainning_process()


