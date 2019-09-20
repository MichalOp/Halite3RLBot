# Halite3RLbot

### RL bot for Halite 3 competition

The code in this repo is likely quite awful, as it wasn't really intended to be read by anyone.

How the code is supposed to be used
---
To train a network yourself, you'll need tensorflow 1.12 with CUDA support (and possibly some other libraries) and as good GPU as possible (I was using GTX 1080Ti). To start training, run `python3 trainer.py`. Tensorboard plots will be located in *training* folder. If you don't have enough VRAM, decrease batch_size in *trainer.py*.

Killing training process might not remove all of the child processes. To do so, run `pkill python3`.

*build_bot.sh* builds a bot ready for deployment using currently saved weights, and stores it in ready_bots folder. The folder currently contains my final version of the bot used in the competition - capable of achieving 60th place.

(in the description I assume the reader knows the rules of Halite III)

Project structure
---
 * *trainer.py* launches the workers and loads, runs and trains the network.
 * *train_func.py* contains tensorflow code for training the network.
 * *halite_network.py* contains network model. 
 * *envcontrol.py* is responsible for playing games and collecting trajectories.
 * *connector.py* is responsible for launching games with *PipeBot.py* agents.
 * *PipeBot.py* is a simple agent capable of comunicating through FIFO files to send observations to and execute orders from *envcontrol.py*.
 * *looptime.py* is an utility to quickly measure time spent by program in loops.
 * *halite_network_deploy.py* is a deployment version of the *halite_network.py*. The only real difference in it is lack of a *@tf.contrib.defun* decorator, as it seemed to cause troubles in tensorflow 1.11 that was used on Halite servers.
 * *DropBot.py* is the final bot code. The name comes from the fact that it builds dropoffs.

Idea of the project
---
It was quite hard for me to imagine how an effective Halite III strategy would look like, so I thought - let algorithm figure that for itself. I had several ideas how to accomplish that, but the simplest one - to consider every ship to be an individual, selfish agent - was what I ultimately went with. Despite its simplicity and obvious problems with cooperation, it turned out to be surprisingly effective.

The core algorithm
---
Training data is collected through self-play by several worker processes. Each of the workers is controlling agents playing a single game, and collecting trajectories for the training algorithm. There is also a single thread that is responsible for running and training the network - workers send input data for the network thread which then runs network on it and sends it back to the workers. That means there is only one network loaded on the GPU memory, which in turn allows me to using bigger batch sizes during training. 

I used Proximal Policy Optimization as a training algorithm due to its stability. I experimented with Deep Q-learning, but it wasn't as effective.

To reduce amount of compute needed, I used a fully convolutional architecture where single network computes policy distribution and value estimates for every ship on the map at the same time. The network is 'looped' at the edges, so it sees the situation behind the edge of the map. 

Rewards
---
At the beginning I simply rewarded agents for delivering halite to the dropoffs. That worked quite well, but the agents learned to trust their opponent that they will help them collect more from inspiration, and they often failed against agressive agents in 1v1 games. 

To counter that, I added a reward for killing an opponent equal to his own estimate of value in 1v1 games. That greatly improved performance in those games, as the agents were less trustful and more willing to sacrifice their lives. (Note that without that reward, killing yourself was always a bad idea for the agent, regardless of the situation.) Still, my agents remained more effective in 4-player games.

One can clearly see that there are many situations that my reward system doesn't cover. Still, it worked reasonably well and there were other problems that didn't seem to be caused by it, and I wasn't able to improve it.

Training and problems
---
In a final configuration, agents were trained on 32x32, 48x48 and 64x64 maps, with 3 workers responsible for 2-player games and 3 for 4-player games. Training was generally run for around 12-18 hours. During that time, performance of the agents quickly improved, but after that, something particularly interesting kept happening. Performance kept to suddenly collapse until it reached zero in several training iterations, with one ship blocking the dropoff. 

I was never able to pinpoint the reason for that, but my leading hypothesis is that as the training progressed, the randomness of the actions was reduced. Without randomness, agents were too dumb to avoid deadlocks. As number of deadlocks increased, it became more and more rewarding to return to the dropoff quicker, before someone blocks it. That in turn blocked the dropoffs more, which encouraged blocking behavior, leading to agents simply trying to return to base all the time. 

I think my hypothesis is also supported by the fact that my Q-learning agents (not included in this repo) failed completely in that game. It seems that my network architecture might have been too limited to successfully understand the game.

Scripted parts
---
Ships and dropoffs are built using a scripted policy. The amount of ships to build is based on the halite on the map divided by the number of players, and a dropoff is built at the ship that is currently furthest away from any other dropoff after every 15 ships are built. 

There is also a movement solver that prevents ships from colliding. It leverages the fact that every ship chooses a single direction it wants to move in, and then computes the longest chains of ships that can be moved by constructing a directed graph and searching for cycles and longest paths in it.

Final thoughts
---
This was my first time I participated in an AI competition, and I've learnt quite a lot during that time. I will definitely try my luck at the next Halite competion if it will be organised, and with the knowledge gained here, I hope to achieve higher score than this time. 

Honestly I was very surprised that my fairly simple (I think) RL solution worked so well. I will definitely try to use RL in other competitions as well.
