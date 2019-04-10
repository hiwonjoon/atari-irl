# Atari-IRL

## System Configuration

```
git clone git@github.com:hiwonjoon/atari-irl.git --recursive
```

```
conda env create -f environment.yml
conda activate atari-irl
pip install -r requirements.txt
```

Open `rllab/sampler/stateful_pool.py` and fix the typo.
`from joblib.pool import MemmapingPool` -> `from joblib.pool import MemmappingPool`

Open `rllab/setup.py` and change the line `if package.startswith('rllab')],` to `if package.startswith('rllab') or package.startswith('sandbox')],`


```
cd rllab
pip install -e .
```

```
cd baselines
```

check whether branch is correct. commit hash must be `24fe3d6`. If not, please run `git checkout 24fe3d6`. Then,

```
pip install -e .
```

```
cd inverse_rl
```

create setup.py and put the code below.
```
# setup.py
from setuptools import setup,find_packages

setup(
    name='inverse_rl',
    packages=[package for package in find_packages()
                if package.startswith('inverse_rl')],
    version='0.1.0',
)
```

```
pip install -e .
```

## Train

1. Train an Expert

```
python -m scripts.train_expert --env PongNoFrameskip-v4 --expert_path ./experts/pong --no-one_hot_code --num_envs 8 --n_cpu 8
```

2. Generate Trajectory

```
python -m scripts.generate_trajectories --env PongNoFrameskip-v4 --expert_path ./experts/pong/checkpoints/update-100 --n_cpu 1 --num_envs 1 --no-render --num_trajectories 8
```

- If you want to check the generate trajectories quality, run

    ```
    python -m scripts.trajectory_to_gif --trajectories_file ./experts/pong/checkpoints/update-100/trajectories.pkl
    ```

3. Train Auto-encoder

```
python -m scripts.train_ae --env PongNoFrameskip-v4 --num_envs 8 --encoder_type pixel_class --log_path ./autoencoder/pong
```

4. Run AIRL algorithm

  - With trained AE

    ```
    python -m scripts.train_airl --env PongNoFrameskip-v4 --n_cpu 1 --num_envs 1 --trajectories_file ./experts/pong/checkpoints/update-100/trajectories.pkl --no-state_only --encoder ./autoencoder/pong/vae_100.pkl --log_path ./airl/pong --reward_type mlp
    ```

  - With raw ob

    ```
    python -m scripts.train_airl --env PongNoFrameskip-v4 --n_cpu 1 --num_envs 1 --trajectories_file ./experts/pong/checkpoints/update-100/trajectories.pkl --no-state_only --log_path ./airl/pong
    ```

