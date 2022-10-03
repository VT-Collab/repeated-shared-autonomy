#!/bin/bash

# find * -size -4M -type f -print0 | xargs -0 git add
# git commit -m "running ours with 20 models"
# git push origin main
python train_cae.py --n-tasks 3
python train_classifier.py --n-tasks 3