#!/bin/bash
# python run_files.py 20
# python sim_play.py 20
# python plot.py 20
python eval_policy.py
find * -size -4M -type f -print0 | xargs -0 git add
git commit -m "Fix bug with our method to run the reqd no. of models"
git push origin main