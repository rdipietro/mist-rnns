#!/bin/bash

learning_rate="$(python3 print_learning_rate.py random -4.0 1.0)"

python3 ../mnist_train.py \
  --permute 1 \
  --layer_type MISTLayer \
  --num_hidden_units 139 \
  --learning_rate $learning_rate
