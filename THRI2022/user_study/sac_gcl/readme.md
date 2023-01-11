### Testing SAC Method 

To run all intents, run `python test_sac_method.py`. 

To run only lemon intent (intent0), run `python test_sac_method.py --num_intents 1`.

To train, run `python learn_intent.py`. To save to a specific filename, use `--intentFolder <filename>`

## Tabulated hyperparameters

| Lemon Intent Run | Batch Size | Memory Size | 
| --- | --- | --- | 
| 1 | 1000 | 10000 |  
| 2 | 10000 | 50000 | 

Final run:
Intent0: batch size 10000
Intent1: batch size 2500
Intent2: batch size 2500


# TODO
- Add return to home
- Add limits for table etc
- Write wrapper to switch b/n ours and CASA
