#!/bin/bash
# python run_files.py 20
# python sim_play.py 20
# python plot.py 20
python eval_policy.py
find * -size -4M -type f -print0 | xargs -0 git add
git commit -m "runs for 20 goals with 20 models per goal and 5 runs"
git push origin main