#!/bin/bash

today=`date '+%Y_%m_%d__%H_%M_%S'`;
zip -r weights_$today.zip weights
mv -f weights_$today.zip saved_weights/weights_$today.zip
