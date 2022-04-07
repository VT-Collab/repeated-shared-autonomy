#!/bin/bash

python train_class_old.py
wait
find * -size -4M -type f -print0 | xargs -0 git add
wait
git commit -m "trained 20 class models per tasklist"
wait 
git push origin main