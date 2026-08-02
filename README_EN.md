# EP Base Running Intelligence

EP (Emerson Performance) project — base running analytics using 100% real,
public Baseball Savant data, 2026 season. Same methodology as
[EP Swing Intelligence](https://github.com/ejimenezperformance/ep-swing-intelligence),
applied to a different skill: speed and decision-making on the bases.

*[Versión en español](README.md)*

## Core question

How well does **raw physical speed** (Sprint Speed, Home-to-First time)
predict the **real base running value** Savant already calculates? And,
more interesting: does speed predict different base-running skills
equally well, or are there large differences between them?

## Data

Five public Baseball Savant downloads ("Running" leaderboards), 2026
season, no player selection on our part — full qualified population:

| File | n | Content |
|---|---|---|
| `data/sprint_speed.csv` | 513 | Sprint Speed (ft/sec), Home-to-First (sec) — physical variables |
| `data/baserunning_run_value.csv` | 228 | Savant's real Baserunning Run Value (overall) |
| `data/base_running.csv` | 298 | Value of taking extra bases (1st→3rd, etc.) |
| `data/basestealing_running_game.csv` | 420 | Stolen base value, primary/secondary leads |
| `data/running_splits.csv` | 482 | Time splits every 5 feet from contact to 90 feet |

All merged by `player_id`, no manual transcription — direct CSV download
from Savant.

## Result (chapter 1)

Linear regression with 10-fold cross-validation (`sprint_speed` +
`hp_to_1b` → real value, normalized by opportunities to avoid confounding
with playing time):

| Skill | n | R² |
|---|---|---|
| Baserunning Run Value (overall) | 227 | **0.41** |
| Extra bases taken (1st→3rd, etc.) | 298 | **0.48** |
| Successful stolen base | 406 | **0.04** |

![Comparison](report/chart_baserunning_comparativo.png)

### The honest read

Raw speed explains nearly half the variance in **taking extra bases** —
makes sense, it's a direct physical decision: the runner sees the ball,
runs, and speed dominates the outcome.

But for **successful stolen bases**, speed explains almost nothing
(R²=0.04). This isn't a model error — it's a real finding: stealing a base
depends on reading the "jump" against the pitcher, count tendencies, the
opposing catcher's arm, and timing — decision and anticipation variables
that sprint speed doesn't capture, even though intuition says "the fastest
runner steals the most bases."

**For coaching context:** this separates two skills that get trained
differently. Sprint speed (linear mechanics, strength, acceleration
technique) is classic Strength & Conditioning territory. Successful base
stealing depends more on game reading and timing — scouting and
situational-repetition work, not just running faster.

## Result (chapter 2)

Chapter 1 left an open question: if top speed doesn't predict successful
stolen bases, is it because no part of the sprint matters, or specifically
because stealing depends on the **burst** ("the jump") rather than cruise
speed?

We used `running_splits.csv` (time at every 5 feet from contact) to break
the sprint into three phases and measure average velocity (ft/sec) in
each:

- **Burst (0-10 ft):** first steps, reaction/start
- **Acceleration (10-45 ft):** acceleration phase
- **Top speed (45-90 ft):** sustained cruise speed

Linear regression, 10-fold cross-validation, each phase tested alone and
all three together (n=179, players with all three data sources merged
and a minimum of 3 stolen-base attempts):

| Skill | Burst (0-10ft) | Acceleration (10-45ft) | Top speed (45-90ft) | All 3 phases |
|---|---|---|---|---|
| Baserunning Run Value (total) | 0.12 | **0.32** | 0.29 | 0.33 |
| Extra bases (success rate) | 0.01 | 0.10 | **0.11** | 0.10 |
| Stolen bases (success rate) | 0.01 | 0.01 | 0.00 | -0.01 |

![Phases](report/chart_baserunning_ch2_phases.png)

### The honest read

This chapter's hypothesis was that stolen-base success would be explained
by the **burst** (the initial "jump"), even if not by top speed. **That
hypothesis did not hold.** No phase of the sprint — not burst, not
acceleration, not cruise speed — predicts stolen-base success. R² stays
essentially at zero across all three phases and even turns negative when
combined.

This is a stronger finding than chapter 1's, not a weaker one: it's not
that "top speed alone falls short of explaining it" — it's that **no
speed metric, measured at any point in the sprint, explains successful
base stealing**. It reinforces that this is a reading-and-decision skill
(timing against the pitcher, count tendencies, catcher's arm) almost
entirely independent of how fast the runner actually is.

For overall base-running value and for extra bases, the **acceleration
phase (10-45 ft)** explains as much or more than top speed — suggesting
much of the real value comes from how quickly a runner reaches useful
speed, not just their maximum ceiling.

**For coaching context:** if the goal is improving overall base running
or extra-base advancement, training the acceleration phase (the first
10-45 feet) has as much or more impact than chasing pure top speed. For
stolen bases, no speed work is going to move the needle — that's scouting
and situational-repetition territory.

## How to reproduce

```bash
pip install pandas scikit-learn matplotlib
python3 model/train_baserunning_model.py       # chapter 1
python3 model/train_baserunning_model_ch2.py   # chapter 2
```

Chapter 1 generates `model/baserunning_model_metrics.json`,
`report/chart_baserunning_scatter.png`, and
`data/merged_baserunning_dataset_2026.csv`.

Chapter 2 generates `model/baserunning_model_ch2_metrics.json`,
`report/chart_baserunning_ch2_phases.png`, and
`data/merged_baserunning_dataset_ch2_2026.csv`.

## Next steps (not included in these chapters)

- Split by position (infielders vs. outfielders vs. catchers)
- Isolate elite runners ("bolts," sprints above a threshold) from the rest
- Triangulate with sprint biomechanics literature (linear acceleration vs.
  agility/change of direction)

## About EP

Part of the **Emerson Performance (EP)** portfolio — baseball performance
analytics methodology combining public Savant data, biomechanics, and
field work as a Strength & Conditioning Coach.
