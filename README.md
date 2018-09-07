# Reinforcement learning

## Notice: some pieces of code comes from internet, thank them all. My work just focuses on the object interaction

## basic method
* maze.py: 2 maps and CartPole-v0, choose by maze=0,1,2
* agent.py: training for Q-learning and SARSA, choose by method=0,1
  + DAgent convert Q-table into Q-nn. it need model/target_model, replay and batch_size=32 to converge. choose by agent=1
  + the qlearn and sarsa code can be reused
* pg.py: GAgent by policy gradient. choose by agent=2
* test_agent.py: test utility controlled by arguments, \-\-help for info

## policy gradient
