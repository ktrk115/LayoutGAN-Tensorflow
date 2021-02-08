#!/bin/bash -e
python main.py --dataset mobile --train --sample_dir samples_mobile
convert samples_mobile/train_*.jpg demo/mobile.gif
