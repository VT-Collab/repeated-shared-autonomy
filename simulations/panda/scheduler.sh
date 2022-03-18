#!/bin/bash
python sim_play.py 20
wait
python plot.py 20
find * -size -4M -type f -print0 | xargs -0 git add
git commit -m "ran for 20 goals"
git push origin main