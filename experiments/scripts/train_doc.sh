#!/bin/bash -e
python main.py --dataset doc --train
convert samples/train_*.jpg demo/doc.gif
