#!/bin/bash
python train_classifier_old.py &
python train_cae_old.py &
wait
python collect_data.py
wait
find * -size -4M -type f -print0 | xargs -0 git add
wait
git commit -m "trained individual models and collected data for regret computation"
wait 
git push origin main