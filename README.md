# Halite3RLbot

### RL bot for Halite 3 competition

The code in this repo is likely quite awful, as it wasn't really intended to be read by anyone. But if any part of it is of any use to you, feel free to copy it.

To train a network yourself, you'll need tensorflow 1.12 with CUDA support (and possibly some other libraries) and as good GPU as possible (I was using GTX 1080Ti). To start training, run *trainer.py*. Tensorboard plots will be located in *training* folder. If you don't have enough VRAM, decrease batch_size in *trainer.py*.

Killing training process might not remove all of the child processes. To do so, run *pkill python3*.

*build_bot.sh* builds a bot ready for deployment using currently saved weights, and stores it in ready_bots folder. The folder currently contains my final version of the bot used in the competition.

Description of what on Earth is going on here might be added in near future.
