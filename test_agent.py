from agent import Agent,sarsa,qlearning
from maze import make_maze
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--maze', type=int,
                      default=0,
                      help='index of maps: 0, 1')
parser.add_argument('--method', type=int,
                      default=0,
                      help='index of method: 0, 1')
FLAGS, unparsed = parser.parse_known_args()
#print(FLAGS)

env = make_maze("space%d" % FLAGS.maze)
agent = Agent(env)

np.random.seed(0)
num_episodes = 50
max_number_of_steps = 30

if FLAGS.method==0:
    qlearning(agent, env, num_episodes, max_number_of_steps)
    agent.save("q-learning%d" % FLAGS.maze)
else:
    sarsa(agent, env, num_episodes, max_number_of_steps)
    agent.save("sarsa%d" % FLAGS.maze)

