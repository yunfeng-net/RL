# Reinforcement learning

## Notice: some pieces of code comes from internet, thank them all. My work just focuses on the class interaction

## basic method
* maze.py: 2 maps and CartPole-v0
* agent.py: training for Q-learning and SARSA
  + DAgent convert Q-table into Q-nn. it need model/target_model, replay and batch_size=32 to converge
  + the qlearn and sarsa code can be reused
* test_agent.py: test utility

## policy gradient
