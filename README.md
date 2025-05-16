# RLAgent in ANAC2025 SCML OneShot
## How to setup
This project requires python version "3.11.0" <= "3.11.11".

To protect local env, I recommend using "pyenv" and "venv".

Creating and activating virtual environment can below code.
```
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## How to train
If you want to train a model, set flag `test_only = False` in `train.py` and run it.

If you want to use existed model to additional training, set flag `add_learn = True` in `train.py` and run it.

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