#!/bin/bash

today=`date '+%Y_%m_%d__%H_%M_%S'`;

rm MyBot.py
cp DropBot.py MyBot.py
zip -r bot_$today.zip MyBot.py halite_network_deploy.py weights hlt
mv -f bot_$today.zip ready_bots/bot_$today.zip
