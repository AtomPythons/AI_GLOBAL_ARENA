# AI Arena

A Pygame humanoid physics sandbox with a Q-learning experiment.
Install `pygame` before running the scripts.

## PPO standing training

Install the dependencies in your activated virtual environment:

```bash
python -m pip install -r requirements-ppo.txt
```

The command below trains an agent to stand for 10 seconds without a spring
controller. The terminal prints a new run directory such as
`ppo_runs/seed_35_YYYYMMDD_HHMMSS`.

```bash
python -m ppo_standing.train \
  --seed 35 --timesteps 204800 --num-envs 4 \
  --episode-seconds 10 --env-version 5 \
  --static-mu 0.2 --kinetic-mu 0.1 \
  --policy bounded --n-steps 512 --batch-size 256 \
  --n-epochs 8 --gamma 0.997 --gae-lambda 0.99 \
  --learning-rate 0.0003 --ent-coef 0.01 \
  --eval-every 8192 --save-every 8192 \
  --eval-seed-start 90000
```

`--seed 35` sets the initial random-number-generator state. With the same
code and settings, it makes the initial poses and training process more
repeatable. Replace `35` with another integer to create a separate experiment.

`--timesteps 204800` is the number of agent-environment interactions, not a
duration in seconds. Larger values take longer to train. With four parallel
environments, the counter advances by four on each shared step.

After training, substitute the printed run directory in the commands below.
`best.zip` is the best model on independent evaluation poses; `last.zip` is
the state at the final training step. Every evaluation also saves the current
model as `checkpoints/step_XXXXXXXXX.zip` inside the run directory. Add
`--watch-eval` to the training command to open a viewer after every evaluation;
press `Esc` in that window to continue training.

```bash
python -m ppo_standing.train \
  --evaluate ppo_runs/seed_35_YYYYMMDD_HHMMSS/best.zip \
  --eval-episode-seconds 10 --eval-seed-start 100000 --eval-episodes 100

python -m ppo_standing.watch \
  --model ppo_runs/seed_35_YYYYMMDD_HHMMSS/best.zip --seed 100000
```

In the viewer, `R` repeats the same initial pose, `N` switches to the next
one, `M` compares the model with zero actions, and `Esc` closes the window.

## Training

```bash
python train_q_learning.py
python train_q_learning.py --headless --batches 1
```

Training runs in batches of 10,000 episodes by default. The agent is evaluated
every 500 episodes. Each run writes two files beside the scripts:
`ai_model_<timestamp>_best_model_of_session.json` (best mean standing time)
and `ai_model_<timestamp>_last_model_of_session.json` (state at exit). The
episode count carries over when training resumes. Use `--batch-size N` to
change the batch size. `--headless` skips the window, and `--batches N` sets
how many batches to run in that mode. To continue training a model, run:

```bash
python train_q_learning.py --model ai_model_YYYYMMDD_HHMMSS_ffffff_last_model_of_session.json
```

After a batch in windowed mode, press `R` to replay the model or `C` to train
another batch.

## Viewing a saved model

```bash
python watch_ai_model.py --model ai_model_YYYYMMDD_HHMMSS_ffffff_best_model_of_session.json
```

Omit `--model` to open the latest session's best checkpoint. In the viewer, `R`
replays the same start, `N` starts a new pose, and `M` toggles model control.
`P` toggles posture assistance and `L` toggles point labels. The arrow keys or
mouse move the selected point; use `1`-`0` or `-` to select a point.

The default `spring` agent chooses any combination of five feedback force
groups (32 actions): pelvis, chest, head, left leg and right leg. Each group
uses the current pose and velocity to correct the body, so this is assisted
balance rather than learning raw motor forces from scratch. The physics and
rendering are unchanged. Use `--agent direct` to start the earlier 13-force
agent with all 8192 raw-force combinations. Use `--agent pose` for a separate
seven-group controller (128 actions) that adds left and right arm forces and
uses mirrored knee targets. The `--pose` switch in `balance_controller.py`
shows that complete pose controller without training. The pose agent still
learns only to avoid falling; it does not yet reward naturalness. These agents
use Q-learning and
reward frames without a fall while penalizing fallen frames. The old
posture-based return is logged separately for diagnosis. The state includes
lean, angular motion, body height, movement, head alignment and foot contact.
Q-values are shared between related poses through small feature groups.

Earlier models still work in the viewer. The `--agent` option is only for a
fresh run; checkpoint files from different versions are not interchangeable
for continued training.

For a non-learning balance-controller demo, run:

```bash
python balance_controller.py --episodes 200 --show
python balance_controller.py --episodes 10 --pose --show
```

On Windows, activate the existing virtual environment in PowerShell with:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
