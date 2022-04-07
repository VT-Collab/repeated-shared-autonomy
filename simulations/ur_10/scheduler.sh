#!/bin/bash

python train_cae_old.py
wait
find * -size -4M -type f -print0 | xargs -0 git add
wait
git commit -m "trained 20 cae models per tasklist"
wait 
git push origin main