# STEPH Score — Documentation

**Shooting Transformation Elevation Per Hundred**

> A lineup-based metric measuring how much a player's on-court presence
> elevates his teammates' shooting efficiency, independent of his own shooting.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Core Metric — STEPH_PCT](#2-core-metric--steph_pct)
3. [Teammate Quality Factor & STEPH_ADJ](#3-teammate-quality-factor--steph_adj)
4. [Total Offensive Efficiency (TOE)](#4-total-offensive-efficiency-toe)
5. [Statistical Significance & Bootstrap CI](#5-statistical-significance--bootstrap-ci)
6. [Data Pipeline](#6-data-pipeline)
7. [Methodological Decisions](#7-methodological-decisions)
8. [Career Averages — Regularization](#8-career-averages--regularization)
9. [Reliability Tiers](#9-reliability-tiers)
10. [Available Outputs](#10-available-outputs)
11. [Interactive Explorer](#11-interactive-explorer)
12. [Limitations & Future Work](#12-limitations--future-work)

---

## 1. Motivation

Traditional on/off metrics measure how a **team** performs with a player on or
off the floor. STEPH Score isolates a specific mechanism: **does a player make
his teammates shoot more efficiently?**

This captures:

- **Gravity** — defenders collapse on a scorer, freeing teammates for open shots
- **Playmaking** — creating higher-quality looks for others
- **Spacing** — floor spacing that improves shot quality for everyone else
- **Screening** — off-ball movement and screens that generate open attempts

It explicitly *excludes* the player's own shooting so that a 70% TS% center
doesn't score highly just by shooting efficiently himself.

---

## 2. Core Metric — STEPH_PCT

### Formula

```
STEPH_PCT(P) = ( TS%_teammates_ON − TS%_teammates_OFF ) / TS%_teammates_OFF × 100
```

Where:

- **TS%_teammates_ON** — minute-weighted True Shooting % of P's teammates,
  computed across all 5-man lineups that include P, with P's estimated shot
  contribution subtracted from each lineup's raw totals.
- **TS%_teammates_OFF** — same calculation for lineups that *exclude* P but
  *include* at least one of P's core teammates (see §7.2).

### Why percentage-based, not absolute points?

The absolute version (`STEPH_ABS = (ts_on − ts_off) × 100`) is kept for
reference, but the percentage form is more meaningful for comparisons:

- Improving teammates from **55% → 57%** is a **+3.6% relative gain**
- Improving teammates from **62% → 64%** is only a **+3.2% relative gain**

The absolute form treats both equally. The percentage form correctly weights
that the second scenario happened against a higher baseline.

### True Shooting %

```
TS% = PTS / ( 2 × (FGA + 0.44 × FTA) )
```

The 0.44 factor accounts for the fact that not all free throw attempts come
from two-shot fouls (and-one, technical fouls, etc.).

### Subtracting P's own shots

`LeagueDashLineups` returns 5-man totals — FGA, FTA, PTS for the entire
lineup, not split by player. To isolate teammate shooting, we estimate P's
contribution per lineup:

```
P_FGA_in_lineup ≈ P_season_FGA_per_min × lineup_MIN
P_FTA_in_lineup ≈ P_season_FTA_per_min × lineup_MIN
P_PTS_in_lineup ≈ P_season_PTS_per_min × lineup_MIN
```

Then:

```
tm_FGA = lineup_FGA − P_FGA_in_lineup   (clipped to ≥ 0)
tm_FTA = lineup_FTA − P_FTA_in_lineup
tm_PTS = lineup_PTS − P_PTS_in_lineup
```

This is an approximation (P's rate varies by lineup context) but removes the
most significant source of self-inflation.

---

## 3. Teammate Quality Factor & STEPH_ADJ

### Motivation

It's harder to improve teammates who already shoot poorly — there are fewer
tactical adjustments that make a 47% TS% player suddenly become a 52% TS%
player. Improving efficient teammates is a sign of genuine gravity, but
*creating efficiency where there was none* is rarer and arguably more
impressive.

### Formula

```
TQF(P) = league_avg_TS% / ( TS%_teammates_OFF × 100 )
```

- If P's teammates shoot **below** league average when P is off the floor →
  `TQF > 1` → STEPH_ADJ **amplified** (harder lift, more meaningful)
- If P's teammates shoot **above** league average when P is off the floor →
  `TQF < 1` → STEPH_ADJ **discounted** (teammates were already good regardless)

```
STEPH_ADJ(P) = STEPH_PCT(P) × TQF(P)
```

---

## 4. Total Offensive Efficiency (TOE)

TOE integrates two things: how well a player shoots **himself**, and how much
he improves **his teammates** — weighted by how much of the team's scoring
each component actually accounts for.

### Step 1 — Own shooting premium

```
own_premium = player_TS% × 100 − league_avg_TS%
```

A player at 68% TS% vs a 57% league average contributes +11 percentage points
of premium. A player at 52% contributes −5 points.

### Step 2 — Shot volume weights

These are derived from actual scoring volumes, not manually chosen:

```
player_volume    = player_PTS  (season total)
teammate_volume  = TS%_ON × player_MIN  (estimated teammate pts while P on floor)
total_volume     = player_volume + teammate_volume

player_weight    = player_volume  / total_volume
teammate_weight  = teammate_volume / total_volume
```

A ball-dominant scorer taking 60% of team shots → `player_weight ≈ 0.60`.
A spacer/screener taking 12% of team shots → `player_weight ≈ 0.12`, so his
TOE is almost entirely driven by his impact on the other four players.

### Step 3 — Asymmetric penalty

Below-average performance is penalized 1.5× harder than equivalent above-average
performance, because actively hurting offensive efficiency is worse than being
neutral:

```
own_comp = own_premium      if own_premium ≥ 0
         = own_premium × 1.5  if own_premium < 0

adj_comp = STEPH_ADJ        if STEPH_ADJ ≥ 0
         = STEPH_ADJ × 1.5    if STEPH_ADJ < 0
```

### Step 4 — Combine

```
TOE = own_comp × player_weight + adj_comp × teammate_weight
```

### Interpretation

| Scenario | Expected TOE |
|---|---|
| Elite scorer + strong gravity (Jokić) | High positive |
| Poor scorer but high gravity (Draymond) | Moderate positive (teammate weight dominates) |
| Elite scorer, zero gravity | Moderate positive (own weight dominates) |
| Poor scorer, negative gravity | Strongly negative |

---

## 5. Statistical Significance & Bootstrap CI

### The problem

A player who appeared in 8 five-man lineups could show STEPH_PCT = +4.2% by
pure sampling noise. A player with 200 lineups at +3.1% is far more trustworthy.
Bootstrap significance quantifies this.

### Bootstrap procedure

For each player:

1. Take their `on_rows` (ON lineups) and `off_rows` (OFF lineups).
2. Resample both with replacement 500 times (each resample picks `len(on_rows)`
   lineups randomly, duplicates allowed).
3. For each resample, compute STEPH_PCT.
4. This produces a distribution of 500 plausible STEPH values.

### Outputs

| Column | Meaning |
|---|---|
| `CI_LO` | 2.5th percentile of bootstrap distribution (lower 95% bound) |
| `CI_HI` | 97.5th percentile (upper 95% bound) |
| `P_VALUE` | Two-sided t-test: is the mean significantly different from 0? |
| `SIG` | Stars: `***` p<0.001, `**` p<0.01, `*` p<0.05, blank = n.s. |

### Interpretation

- **CI crosses zero** (e.g. `[-0.8, +4.1]`) → effect not statistically
  distinguishable from noise at 95% confidence.
- **CI entirely positive** (e.g. `[+1.1, +5.3]`) → player reliably elevates
  teammates with high confidence.
- **`***`** on the leaderboard means the effect was found in 99.9% of bootstrap
  resamples to be positive — very unlikely to be a fluke.

---

## 6. Data Pipeline

### API calls per season (2 total)

| Endpoint | Purpose |
|---|---|
| `LeagueDashLineups` | All 5-man lineup combinations with FGA, FTA, PTS, MIN |
| `LeagueDashPlayerStats` | Individual season totals (PTS, FGA, FTA, AST, MIN, GP) |

### Processing

```
For each qualifying player P:
  1. Convert P's name to "F. LastName" (ascii-normalized) to match GROUP_NAME
  2. Filter lineups to P's team(s)
  3. Split ON lineups (GROUP_NAME contains P) vs OFF (does not)
  4. Identify "core teammates" — players sharing ≥ MIN_SHARED_MIN lineup-minutes
  5. Restrict OFF sample to lineups containing at least one core teammate
  6. Subtract P's estimated shots from ON lineup totals
  7. Compute minute-weighted TS% for both splits
  8. Compute STEPH_PCT, TQF, STEPH_ADJ, TOE, bootstrap CI
```

### Unicode normalization

NBA API player name encoding is inconsistent across endpoints. `Jokić` in
player stats and `Jokic` in lineup `GROUP_NAME` would fail to match. All names
are normalized with:

```python
unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("ascii")
```

---

## 7. Methodological Decisions

### 7.1 Minutes qualifier

Only players in the **top 20% of minutes played** (`MIN_MIN_PCT = 0.80`) qualify
for the per-season leaderboard. This is season-specific — the threshold is
computed fresh each season from that season's player pool.

Players who don't qualify due to injury or limited role still get a STEPH score
computed during the career pass (see §8), as long as they have sufficient lineup
data.

### 7.2 Matched OFF sample

A naive OFF sample (all lineups without P) includes garbage-time combinations
involving players who never shared the court with P. This introduces noise.

The matched approach: P's OFF sample is restricted to lineups that contain at
least one player who shared ≥ `MIN_SHARED_MIN = 150` minutes of lineup time
with P.

**Example:** Steph Curry's OFF sample = Warriors lineups containing
Draymond Green, Kuminga, etc., but not Steph. This compares the same
surrounding cast with and without Steph.

### 7.3 Career pass threshold

The career uncapped pass uses `MIN_SHARED_CAREER = 50` (vs 150 for the season
pass) to capture bench-role seasons and early-career data where a player didn't
accumulate 150 shared lineup minutes with anyone.

---

## 8. Career Averages — Regularization

### Problem

A player with one exceptional season (e.g. STEPH_PCT = +8.0%) should not rank
above a player with five consistently great seasons (avg +5.5%). Without
regularization, one-season wonders pollute the career leaderboard.

### Bayesian regularization

Career average uses a **regularized minute-weighted mean** that adds one
synthetic "prior season" at the league average:

```
prior_weight = REGULARIZATION × avg_season_minutes   (= 1 × ~2,200 = 2,200 min)

AVG_STEPH_PCT = ( Σ STEPH_PCT_i × MIN_i + league_avg × prior_weight )
               / ( Σ MIN_i + prior_weight )
```

Effect by number of seasons:

| Seasons | Prior influence |
|---|---|
| 1 season | Prior ≈ 50% of the weight |
| 3 seasons | Prior ≈ 25% |
| 7 seasons | Prior ≈ 13% |
| 9 seasons | Prior ≈ 11% |

A player with 9 seasons of data is essentially unaffected. A one-season player
is pulled meaningfully toward the mean.

### Season filter

Seasons where the player played fewer than `MIN_CAREER_MIN = 700` minutes are
excluded from the career average (injury seasons, partial trades, etc.).

---

## 9. Reliability Tiers

| Tier | Seasons | Reliability Score | Interpretation |
|---|---|---|---|
| Established | 3+ | 1.0 | Sample confirmed — trust career average |
| Developing | 2 | 0.67 | Promising but needs another season |
| Emerging | 1 | 0.33 | High score possible, could be noise |

`RELIABILITY = min(SEASONS / 3, 1.0)`

The `career_steph_reliability.png` chart shows Established players separately
from Emerging ones to prevent one-season standouts from dominating the career
narrative.

---

## 10. Available Outputs

### CSV files

| File | Contents |
|---|---|
| `steph_all_seasons.csv` | Every player-season with all metrics |
| `steph_career_averages.csv` | Career averages per player in the steph_leaders pool |

### CSV column reference — `steph_all_seasons.csv`

| Column | Description |
|---|---|
| `STEPH_PCT` | Primary metric — relative % improvement in teammate TS% |
| `STEPH_ADJ` | Quality-adjusted STEPH (× teammate quality factor) |
| `STEPH_ABS` | Absolute teammate TS% pts gained (legacy, kept for reference) |
| `TQF` | Teammate Quality Factor (`league_avg_ts / tm_ts_off`) |
| `TM_TS_ON` | Teammate TS% when P is on the floor (P's shots excluded) |
| `TM_TS_OFF` | Teammate TS% when P is off the floor (matched sample) |
| `OWN_TS` | Player's own True Shooting % |
| `LEAGUE_AVG_TS` | League average TS% that season |
| `TOE` | Total Offensive Efficiency |
| `P_SHOT_WEIGHT` | Fraction of on-floor scoring attributed to P |
| `TM_SHOT_WEIGHT` | Fraction attributed to teammates (= 1 − P_SHOT_WEIGHT) |
| `CI_LO` | Bootstrap 95% CI lower bound for STEPH_PCT |
| `CI_HI` | Bootstrap 95% CI upper bound for STEPH_PCT |
| `P_VALUE` | Bootstrap t-test p-value (null: STEPH_PCT = 0) |
| `SIG` | Significance stars (`***` `**` `*` or blank) |
| `PPG` | Points per game |
| `APG` | Assists per game |
| `N_CORE_TM` | Number of core teammates (≥ MIN_SHARED_MIN shared minutes) |
| `N_ON` | Number of 5-man lineup rows in P's ON sample |
| `N_OFF` | Number of 5-man lineup rows in P's matched OFF sample |

### Plots

| File | Description |
|---|---|
| `scatter_{season}.png` | Per-season OWN_TS vs STEPH_PCT scatter |
| `career_steph_scatter.png` | Career avg STEPH_PCT vs seasons calculated |
| `career_steph_reliability.png` | Established vs Emerging bar chart |
| `career_toe_bar.png` | Career TOE leaderboard |
| `career_toe_scatter.png` | Career TOE vs STEPH_ADJ scatter |
| `top10_single_season.png` | Top 10 single-season STEPH_PCT scores |
| `career_steph_adj_bar.png` | Career quality-adjusted STEPH leaderboard |

---

## 11. Interactive Explorer

### Setup

```bash
pip install streamlit plotly pandas
streamlit run steph_explorer.py
```

<<<<<<< HEAD
Place `steph_all_seasons.csv` in the same directory as `steph_explorer.py`.

### Modes

On launch the app shows three mode options and nothing else — no default is
selected. Pick a mode from the sidebar radio:

| Mode | Purpose |
|---|---|
| **Graph** | Scatter or bar chart. Highlight specific players. Choose season scope. |
| **Player Stats** | Full stat table for one or more players. No chart. |
| **Compare** | Two players side by side. No chart. |

---

### Graph mode

#### Season scope

The scope dropdown offers three types at the top, followed by individual seasons:

| Option | What it shows |
|---|---|
| **Career Average** | One dot per player — minute-weighted average across all their seasons in the dataset. Hover shows player name, teams, and number of seasons. |
| **All Seasons** | Every player-season as its own dot. A player like LeBron James has one dot per season he appears in the data. |
| **YYYY-YY** | A specific individual season. |

Seasons available: **2007-08 onwards**, excluding the COVID-shortened **2019-20**
season which is omitted from the dataset entirely due to its non-representative
nature.

#### Highlight player(s)

Selected players are drawn as fully opaque, bordered dots with name and team
annotated directly on the chart. All other players remain as faint background
dots. In *All Seasons* scope the annotation also shows the short season (e.g.
"LeBron James · CLE (12-13)"). In *Career Average* scope it shows the number
of seasons.

#### Chart types

**Scatter** — any stat on X, any stat on Y. Optional league average crosshairs
(horizontal dashed line for Y, vertical dotted line for X). Optional dot sizing
by minutes played.

**Bar** — single stat, top N players, optional league average vertical line.
Bars are gold if above league average, red if below. Highlighted players are
drawn in blue regardless of value.

---

### Player Stats mode

Select one or more players, then choose either a specific season or **Career
Average**. All available stats are displayed in a table — there is no stat
picker; everything is shown with a plain-English description alongside each
value. No chart is generated.

**Career Average** in this mode shows the minute-weighted mean across all
qualifying seasons in the dataset, plus total minutes and the number of seasons
used in the calculation.

---

### Compare mode

Select two players independently. Each has its own season/career scope selector,
so you can mix periods freely — for example Jokić 2023-24 vs Kobe Bryant
2008-09, or LeBron James career vs Michael Jordan career.

Stats are displayed in three columns: **Player 1 value | Stat name | Player 2
value**. The better value for each stat is highlighted in gold. For `P_VALUE`
the comparison is correctly reversed — a lower p-value is better, and is
highlighted gold accordingly.

No chart is generated in this mode.

---
=======
Place `steph_all_seasons.csv` in the same directory.

### Features

| Control | Description |
|---|---|
| **Season(s)** | Multi-select — combine seasons or view individually |
| **Highlight Player(s)** | Named players are annotated; others shown as background dots |
| **Chart Type** | Scatter or Bar |
| **X / Y Axis** (scatter) | Any of the 7 available stats |
| **Stat** (bar) | Any of the 7 available stats |
| **League Average Lines** | Toggle horizontal/vertical league average reference |
| **Size by Minutes** | Dot size proportional to minutes played (scatter only) |
| **Top N** (bar) | How many players to show |
>>>>>>> origin/main

### Available stats in the explorer

| Stat | Column | Description |
|---|---|---|
| STEPH Absolute | `STEPH_ABS` | Teammate TS% pts gained (absolute) |
<<<<<<< HEAD
| STEPH % | `STEPH_PCT` | Relative teammate TS% improvement — primary metric |
| STEPH Adjusted % | `STEPH_ADJ` | Quality-adjusted gravity (× TQF) |
| Own True Shooting % | `OWN_TS` | Player's own TS% |
| Points Per Game | `PPG` | Season PTS / GP |
| Assists Per Game | `APG` | Season AST / GP |
| Total Offensive Efficiency | `TOE` | Combined own efficiency + teammate gravity |

These seven stats are available in Graph mode dropdowns. Player Stats and
Compare modes show all columns from the CSV (see §10 for full column reference).

---

### Example use cases

**"Where does Klay Thompson sit on the TOE vs TS% scatter in 2022-23?"**
→ Mode: Graph · Scope: 2022-23 · Chart: Scatter · X: Own TS% · Y: TOE
· Highlight: Klay Thompson

**"Show every LeBron season as its own data point on the STEPH % leaderboard"**
→ Mode: Graph · Scope: All Seasons · Chart: Bar · Stat: STEPH %
· Highlight: LeBron James

**"Who had the highest career average gravity score among players active since 2007?"**
→ Mode: Graph · Scope: Career Average · Chart: Bar · Stat: STEPH %

**"How does Jokić's 2023-24 season compare to Kobe's 2008-09?"**
→ Mode: Compare · Player 1: Nikola Jokić / 2023-24 · Player 2: Kobe Bryant / 2008-09

**"What were all of Draymond Green's stats in 2020-21?"**
→ Mode: Player Stats · Player: Draymond Green · Season: 2020-21
=======
| STEPH % | `STEPH_PCT` | Relative teammate TS% improvement (primary) |
| STEPH Adjusted % | `STEPH_ADJ` | Quality-adjusted gravity |
| Own True Shooting % | `OWN_TS` | Player's own TS% |
| Points Per Game | `PPG` | Derived from season PTS / GP |
| Assists Per Game | `APG` | Derived from season AST / GP |
| Total Offensive Efficiency | `TOE` | Combined own + teammate impact |

### Example use cases

**"Show me where Klay Thompson sits on the TOE vs TS% scatter for 2022-23"**
→ Season: 2022-23 · Chart: Scatter · X: Own TS% · Y: TOE · Highlight: Klay Thompson

**"Compare Steph Curry and Draymond Green's STEPH % across all seasons"**
→ Seasons: all · Chart: Bar · Stat: STEPH % · Highlight: both players

**"Who led the league in TOE in 2024-25?"**
→ Season: 2024-25 · Chart: Bar · Stat: TOE · no highlights
>>>>>>> origin/main

---

## 12. Limitations & Future Work

### Current limitations

**Per-minute rate assumption.** Subtracting P's shots uses season-wide
per-minute averages. In reality, P's usage varies by lineup. A star in
lineups with other stars shoots less; in lineups with bench players, more.
This introduces systematic error for high-usage players in varied lineups.

**Lineup cap.** `LeagueDashLineups` returns the top 2,000 lineups by minutes.
Rare combinations (< ~10 min) are excluded. This mostly affects bench rotations
and shouldn't impact qualifying players significantly.

**Same-team only.** Traded players get their scores computed per team and
minute-weighted. The OFF sample for a traded player is restricted to the same
team's lineups, so the comparison is internally consistent but doesn't capture
cross-team effects.

**Coaching and system effects.** A player on a pass-heavy team will mechanically
show higher teammate TS% because the system creates better shots. STEPH_ADJ
partially addresses this via TQF but doesn't fully remove system effects.

**No opponent adjustment.** Playing exclusively against weak defenses inflates
both ON and OFF TS%, but should largely cancel out in the delta.

### Possible extensions

- **Play-by-play version** — compute for each possession who shot and who was
  on the floor, eliminating the per-minute rate approximation entirely.
- **Positional groupings** — compute separately for wing vs. big teammate
  subsets to understand where gravity manifests.
- **Rolling window STEPH** — month-by-month STEPH to track how a player's
  gravity changes across a season (injury recovery, system adjustments).
- **Opponent-adjusted STEPH** — weight each lineup's TS% delta by defensive
  quality of the opposing team.

---

<<<<<<< HEAD
*Last updated: 2026 · Built with nba_api, pandas, scipy, matplotlib, plotly, streamlit*
=======
*Last updated: 2025 · Built with nba_api, pandas, scipy, matplotlib, plotly, streamlit*
>>>>>>> origin/main
