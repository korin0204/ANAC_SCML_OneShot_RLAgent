# RLAgent in ANAC2025 SCML OneShot
## How to setup
This project requires python version "3.11.0" <= "3.11.11".

To protect local env, I recommend using "pyenv" and "venv".

## How to train
If you want to train a model, set flag `test_only = False` in `train.py`.

If you want to use existed model to train a model, set flag `add_learn = True` in `train.py`.

## How to test your trained model
To test world using a context,
set flag `test_only = True` in `train.py`. And
```
python3 -m RLAgent.train
```

To full test world,
```
python3 -m RLAgent.myagent
```