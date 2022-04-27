#!/bin/bash
# python run_files.py 20
# python sim_play.py 20
# python plot.py 20
python eval_ensemble.py
find * -size -4M -type f -print0 | xargs -0 git add
git commit -m "100 ensemble runs with 20 models per run"
git push origin main