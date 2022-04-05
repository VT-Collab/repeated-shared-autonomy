#!/bin/bash
# python run_files.py 20
python sim_play.py 20
python plot.py 20
find * -size -4M -type f -print0 | xargs -0 git add
git commit -m "ran for 20 goals with small latent"
git push origin main