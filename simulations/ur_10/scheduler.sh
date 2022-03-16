#!/bin/bash

python collect_data.py
wait
find * -size -4M -type f -print0 | xargs -0 git add
wait
git commit -m "ran models with 8 latent z"
wait 
git push origin main