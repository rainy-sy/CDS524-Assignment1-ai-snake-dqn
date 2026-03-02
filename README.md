# CDS524 Assignment 1 — Reinforcement Learning Game Design

**Student:** WANG Shiyu  
**Game:** AI Snake (Deep Q-Learning / DQN)  

This folder contains all deliverables needed for submission.

## Links (Deliverables)
- GitHub repo: https://github.com/rainy-sy/CDS524-Assignment1-ai-snake-dqn
- Google Colab notebook (or equivalent): `CDS524_Assignment1_Snake_DQN.ipynb`
- YouTube demo video: <PASTE_YOUTUBE_LINK_HERE>

## 1) How to run (local Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python snake_dqn.py
```

### Controls
- `T`: cycle mode (`manual` → `random_ai` → `test`) (training is started via `F3`)
- `F1/F2/F3/F4`: jump to mode (`manual/random_ai/training/test`)
- `SPACE`: pause/resume
- `R`: reset current game
- Arrow keys: manual control (manual starts only after first arrow key)
- `1/2/3`: manual speed level
- `ENTER`: play again (on GAME OVER)
- `Q`: quit

## 2) Modes

- **manual**: human plays
- **random_ai**: random action agent
- **training**: runs 300 episodes with experience replay; saves `model.pth`; exports `training_curve.png`
- **test**: loads `model.pth` and plays with `epsilon=0` (pure exploitation)

## 3) Headless training (for Colab / notebook)

If your environment cannot open a Pygame window (e.g., Google Colab), you can run training headlessly:

```bash
HEADLESS=1 python snake_dqn.py
```

In headless mode, audio and training screenshots are disabled.

## 4) Files produced

- `snake_dqn.py`: full game + DQN agent (single-file)
- `model.pth`: saved DQN model (created during training)
- `training_curve.png`: training curve (score vs episode)
- `best_score.txt`: best score record
- `training_ep_*.png`: optional training screenshots (every 50 episodes; not in headless)

## 5) Rubric mapping (quick)

- **Game Design**: objective/rules, defined state/action spaces, reward function
- **Q-Learning**: implemented as Deep Q-Learning (DQN) with epsilon-greedy, learning rate, discount factor
- **Game Interaction**: Pygame UI shows state/action/reward and supports multiple modes
- **Documentation**: report + demo checklist included in `REPORT.md`

## 6) GitHub repo setup

This folder is ready to be a GitHub repository (it includes a `.gitignore`).

### Option A: GitHub Desktop (easiest on Windows)
1. Install GitHub Desktop
2. **File → Add Local Repository...** and select this folder
3. Publish to GitHub

### Option B: Git for Windows (command line)
1. Install Git for Windows
2. In PowerShell, run:

```powershell
git init
git add .
git commit -m "Initial commit"
```

Then create a repo on GitHub and follow the shown `git remote add origin ...` commands.
