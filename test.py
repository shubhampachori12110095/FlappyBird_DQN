import DQN_Brain
import gym
import gym_ple


env = gym.make("FlappyBird-v0")

DQN_play = DQN_Brain.Experience_Replay(env = env, INITIAL_RATIO = 0.00001, TEST_MODE = True)
DQN_play.load_the_models("MyModel/save_net-4520000")
DQN_play.start_the_trainning_process()


