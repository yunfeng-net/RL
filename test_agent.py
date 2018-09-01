from agent import Agent,DAgent,sarsa,qlearning
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
parser.add_argument('--agent', type=int,
                      default=0,
                      help='index of agent: 0, 1')
FLAGS, unparsed = parser.parse_known_args()
FLAGS.maze = 2
FLAGS.agent= 1
FLAGS.method = 0
print(FLAGS)

env = make_maze("space%d" % FLAGS.maze)
if FLAGS.agent==0:
    agent = Agent(env)
else:
    agent = DAgent(env)

np.random.seed(0)
num_episodes = 50
max_number_of_steps = 30
if FLAGS.maze>=2:
    num_episodes *= 10
    max_number_of_steps *= 10

if FLAGS.method==0:
    qlearning(agent, env, num_episodes, max_number_of_steps)
    agent.save("q-learning%d" % FLAGS.maze)
else:
    sarsa(agent, env, num_episodes, max_number_of_steps)
    agent.save("sarsa%d" % FLAGS.maze)

