# AI Arena: PPO standing

## Train a standing agent

Run this command from the project root:

```bash
python -m ppo_standing.train \
  --seed 35 --timesteps 204800 --num-envs 4 \
  --episode-seconds 10 --env-version 5 \
  --static-mu 0.2 --kinetic-mu 0.1 \
  --policy bounded --n-steps 512 --batch-size 256 \
  --n-epochs 8 --gamma 0.997 --gae-lambda 0.99 \
  --learning-rate 0.0003 --ent-coef 0.01 \
  --eval-every 8192 --save-every 8192 \
  --eval-seed-start 90000 --watch-eval
```

Training flags:

- `--seed 35` selects the random starting point for the experiment. Use another
  integer to run a separate experiment.
- `--timesteps 204800` is the total number of agent-environment interactions
  used for training. Larger values take longer.
- `--eval-every 8192` evaluates the current model after every 8,192
  interactions and saves a checkpoint for that evaluation.
- `--watch-eval` opens a viewer after each evaluation. Press `Esc` in the
  viewer to resume training.

The terminal prints the directory for the new run, for example
`ppo_runs/seed_35_YYYYMMDD_HHMMSS`. It contains `best.zip` (the best evaluated
model), `last.zip` (the final model), and a checkpoint after every evaluation
in `checkpoints/`.

## Continue a training run

Use the run directory to continue from its latest `last.zip` checkpoint. The
new steps, evaluations, and checkpoints are saved in the same directory:

```bash
python -m ppo_standing.train \
  --resume ppo_runs/RUN_NAME --timesteps 102400
```

## View a trained model

Replace `RUN_NAME` with the directory printed during training:

```bash
python -m ppo_standing.watch \
  --model ppo_runs/RUN_NAME/best.zip --seed 100000
```

In the viewer, `R` repeats the same pose, `N` selects the next pose, `M`
compares the model with zero actions, and `Esc` closes the window.
