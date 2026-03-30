# steph_score.py  — Shooting Transformation Elevation Per Hundred
# =============================================================================
# Primary metric:  STEPH_PCT  — percentage-based relative improvement in
#                               teammate True Shooting %
#   STEPH_PCT   = (ts_on − ts_off) / ts_off × 100
#   STEPH_ADJ   = STEPH_PCT × (ts_off / league_avg_ts)   [quality-adjusted]
#
# TOE (Total Offensive Efficiency):
#   Integrates own shooting premium and teammate impact, weighted by
#   each player's actual shot-volume share on the floor.
#   TOE = own_premium × player_shot_weight
#       + STEPH_ADJ  × teammate_shot_weight
#
# Pipeline: 2 API calls per season → in-memory computation → 12+ charts + CSVs
# =============================================================================

import time
import warnings
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from scipy import stats

from nba_api.stats.endpoints import LeagueDashLineups, LeagueDashPlayerStats
from nba_api.stats.library import http as nba_http

warnings.filterwarnings("ignore")

# ── NBA API headers ────────────────────────────────────────────────────────────
nba_http.STATS_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}

# =============================================================================
# CONFIG
# =============================================================================
SEASONS = [f"{y}-{str(y+1)[-2:]}" for y in range(2007, 2025)]
MIN_MIN_PCT       = 0.80    # top 20% of minutes → qualifying per season
MIN_SHARED_MIN    = 150     # shared lineup-minutes to be a "core teammate" (per-season pass)
MIN_SHARED_CAREER = 50      # looser threshold for the career uncapped pass
MIN_CAREER_MIN    = 700     # exclude seasons < 700 min from career average
REGULARIZATION    = 1       # Bayesian prior seasons added for career average shrinkage
TOP_N             = 20
SLEEP_SEC         = 1.5
SLEEP_SEASON      = 8
TIMEOUT           = 60
BOOTSTRAP_N       = 500     # resample iterations for significance testing

# =============================================================================
# COLOUR PALETTE  (publisher-grade dark theme)
# =============================================================================
BG       = "#0D1117"   # page background
PANEL    = "#161B22"   # plot panel
GRID     = "#21262D"   # subtle gridlines
TEXT     = "#E6EDF3"   # primary text
MUTED    = "#8B949E"   # secondary / axis labels
GOLD     = "#F0B429"   # positive / high values
RED      = "#E05C5C"   # negative / low values
TEAL     = "#26C6A0"   # tertiary accent (TOE)
BLUE     = "#58A6FF"   # reliability / significance
PURPLE   = "#BC8CFF"   # emerging players

CMAP_DIV = "RdYlGn"   # diverging colormap for quality-adjusted plots

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "text.color"        : TEXT,
    "axes.labelcolor"   : MUTED,
    "xtick.color"       : MUTED,
    "ytick.color"       : MUTED,
    "figure.facecolor"  : BG,
    "axes.facecolor"    : PANEL,
    "axes.edgecolor"    : GRID,
    "grid.color"        : GRID,
    "grid.linewidth"    : 0.5,
    "axes.titlecolor"   : TEXT,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "legend.facecolor"  : PANEL,
    "legend.edgecolor"  : GRID,
    "legend.labelcolor" : TEXT,
})

# =============================================================================
# HELPERS
# =============================================================================

def true_shooting(pts, fga, fta):
    denom = 2 * (fga + 0.44 * fta)
    return np.where(denom > 0, pts / denom, np.nan)


def ascii_name(name):
    """Strip diacritics: Jokić → Jokic, Dončić → Doncic."""
    return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("ascii")


def to_lineup_name(full_name):
    """'Shai Gilgeous-Alexander' → 'S. Gilgeous-Alexander' (ascii-safe)."""
    n = ascii_name(full_name)
    parts = n.strip().split(" ", 1)
    return f"{parts[0][0]}. {parts[1]}" if len(parts) == 2 else n


def short_season(season):
    return season[2:]   # "2022-23" → "22-23"


def teammate_ts_raw(df, fga_rate, fta_rate, pts_rate):
    """Minute-weighted teammate TS% with P's shots subtracted."""
    tm_fga = (df["FGA"] - fga_rate * df["MIN"]).clip(lower=0)
    tm_fta = (df["FTA"] - fta_rate * df["MIN"]).clip(lower=0)
    tm_pts = (df["PTS"] - pts_rate * df["MIN"]).clip(lower=0)
    total  = df["MIN"].sum()
    if total == 0:
        return np.nan
    w  = df["MIN"] / total
    ts = true_shooting(tm_pts, tm_fga, tm_fta)
    return float(np.nansum(w * ts))


def get_core_teammates(on_rows, pname, threshold):
    mins = {}
    for _, row in on_rows.iterrows():
        for nm in [n.strip() for n in str(row["GROUP_NAME"]).split(" - ")]:
            if nm != pname:
                mins[nm] = mins.get(nm, 0) + row["MIN"]
    return {n for n, m in mins.items() if m >= threshold}


def lineup_has_core(group_name, core):
    if not group_name or not isinstance(group_name, str):
        return False
    return bool(set(n.strip() for n in group_name.split(" - ")) & core)


def dark_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def shadow():
    return [pe.withStroke(linewidth=2.5, foreground=BG)]


def sig_stars(p):
    """Return significance stars from p-value."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def bootstrap_steph(on_df, off_df, rates, n=BOOTSTRAP_N, seed=42):
    """
    Bootstrap confidence interval for STEPH_PCT.
    Resamples ON and OFF lineup sets with replacement N times.
    Returns (mean, lower_95, upper_95, p_value).
    """
    rng     = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        on_s  = on_df.sample(len(on_df),  replace=True, random_state=rng.integers(1e9))
        off_s = off_df.sample(len(off_df), replace=True, random_state=rng.integers(1e9))
        ts_on  = teammate_ts_raw(on_s,  *rates)
        ts_off = teammate_ts_raw(off_s, 0, 0, 0)
        if not np.isnan(ts_on) and not np.isnan(ts_off) and ts_off > 0:
            samples.append((ts_on - ts_off) / ts_off * 100)
    if len(samples) < 10:
        return np.nan, np.nan, np.nan, np.nan
    arr = np.array(samples)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    # two-sided t-test against null of 0
    t_stat, p = stats.ttest_1samp(arr, 0)
    return float(np.mean(arr)), float(lo), float(hi), float(p)


def no_overlap_offsets(xs, ys, base_offset=(8, 12), min_dist=0.035, max_iterations=50):
    """
    Improved anti-overlap for scatter annotations using force-directed placement.
    Returns list of (dx, dy) offsets per point.
    """
    if len(xs) == 0:
        return []

    # Normalize coordinates to 0-1 scale for consistent distance calculations
    x_range = max(xs) - min(xs) if max(xs) != min(xs) else 1
    y_range = max(ys) - min(ys) if max(ys) != min(ys) else 1

    xs_norm = [(x - min(xs)) / x_range for x in xs]
    ys_norm = [(y - min(ys)) / y_range for y in ys]

    # Initial positions: alternate above/below to reduce initial collisions
    offsets = []
    for i in range(len(xs)):
        # Stagger initial positions more aggressively
        dy = base_offset[1] if i % 2 == 0 else -base_offset[1] * 1.5
        dx = base_offset[0] if xs_norm[i] < 0.5 else -base_offset[0]
        offsets.append([dx, dy])

    # Force-directed adjustment to resolve overlaps
    for iteration in range(max_iterations):
        moved = False

        for i in range(len(xs)):
            xi, yi = xs_norm[i] + offsets[i][0]/100, ys_norm[i] + offsets[i][1]/100

            for j in range(i + 1, len(xs)):
                xj, yj = xs_norm[j] + offsets[j][0]/100, ys_norm[j] + offsets[j][1]/100

                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy

                if dist_sq < min_dist * min_dist and dist_sq > 0:
                    # Push apart
                    dist = dist_sq ** 0.5
                    force = (min_dist - dist) / dist * 0.5

                    offsets[i][0] += dx * force * 15
                    offsets[i][1] += dy * force * 15
                    offsets[j][0] -= dx * force * 15
                    offsets[j][1] -= dy * force * 15
                    moved = True

        if not moved:
            break

    # Clamp to reasonable bounds
    for i in range(len(offsets)):
        offsets[i][0] = max(-25, min(25, offsets[i][0]))
        offsets[i][1] = max(-30, min(35, offsets[i][1]))

    return [(int(o[0]), int(o[1])) for o in offsets]


# =============================================================================
# CORE COMPUTATION
# =============================================================================

def compute_season(season, lineups_df, player_df, league_avg_ts, threshold=None):
    """
    Compute STEPH_PCT, STEPH_ADJ, TOE, CI, significance for all qualifying players.
    threshold: override MIN_SHARED_MIN (used for career uncapped pass).
    """
    t = threshold if threshold is not None else MIN_SHARED_MIN
    pdf = player_df.copy()
    pdf["OWN_TS"]      = true_shooting(pdf["PTS"], pdf["FGA"], pdf["FTA"])
    pdf["FGA_PER_MIN"] = pdf["FGA"] / pdf["MIN"].replace(0, np.nan)
    pdf["FTA_PER_MIN"] = pdf["FTA"] / pdf["MIN"].replace(0, np.nan)
    pdf["PTS_PER_MIN"] = pdf["PTS"] / pdf["MIN"].replace(0, np.nan)

    min_thr   = pdf["MIN"].quantile(MIN_MIN_PCT) if threshold is None else 0
    qualified = pdf[pdf["MIN"] >= min_thr].copy()
    qualified["LINEUP_NAME"] = qualified["PLAYER_NAME"].apply(to_lineup_name)

    rows = []

    for _, player in qualified.iterrows():
        pname    = player["LINEUP_NAME"]
        team_ids = pdf.loc[pdf["PLAYER_ID"] == player["PLAYER_ID"], "TEAM_ID"].tolist()
        rates    = (player["FGA_PER_MIN"], player["FTA_PER_MIN"], player["PTS_PER_MIN"])

        if any(pd.isna(r) for r in rates):
            continue

        tl = lineups_df[lineups_df["TEAM_ID"].isin(team_ids)]
        if tl.empty:
            continue

        on_mask  = tl["GROUP_NAME"].str.contains(pname, regex=False, na=False)
        on_rows  = tl[on_mask]
        if on_rows.empty:
            continue

        core = get_core_teammates(on_rows, pname, t)
        if not core:
            continue

        off_mask = (~on_mask) & tl["GROUP_NAME"].apply(lambda g: lineup_has_core(g, core))
        off_rows = tl[off_mask]
        if off_rows.empty:
            continue

        ts_on  = teammate_ts_raw(on_rows,  *rates)
        ts_off = teammate_ts_raw(off_rows, 0, 0, 0)

        if np.isnan(ts_on) or np.isnan(ts_off) or ts_off <= 0:
            continue

        # ── Primary STEPH metrics ──────────────────────────────────────────
        steph_abs = (ts_on - ts_off) * 100          # absolute pts difference
        steph_pct = (ts_on - ts_off) / ts_off * 100  # percentage improvement (primary)

        # Quality factor: making inefficient teammates efficient is harder
        tqf       = league_avg_ts/(ts_off*100) if league_avg_ts > 0 else 1.0
        steph_adj = steph_pct * tqf                  # quality-adjusted %

        # ── Statistical significance via bootstrap ─────────────────────────
        bs_mean, ci_lo, ci_hi, p_val = bootstrap_steph(on_rows, off_rows, rates)
        stars = sig_stars(p_val) if not np.isnan(p_val) else ""

        # ── TOE: Total Offensive Efficiency ───────────────────────────────
        # Own premium over league average (pts scale, matching steph_adj scale)
        own_premium = (player["OWN_TS"] * 100) - league_avg_ts
        own_comp    = own_premium if own_premium >= 0 else own_premium * 1.5
        adj_comp    = steph_adj  if steph_adj   >= 0 else steph_adj   * 1.5

        # Volume weights: what share of on-floor scoring is P vs teammates?
        p_pts       = player["PTS"]
        tm_pts_est  = ts_on * player["MIN"]            # estimated teammate pts while P on
        total_vol   = p_pts + tm_pts_est
        p_w         = p_pts     / total_vol if total_vol > 0 else 0.5
        tm_w        = tm_pts_est / total_vol if total_vol > 0 else 0.5

        TOE = own_comp * p_w + adj_comp * tm_w

        rows.append({
            "PLAYER_ID"          : player["PLAYER_ID"],
            "PLAYER_NAME"        : player["PLAYER_NAME"],
            "SEASON"             : season,
            "STEPH_ABS"          : round(steph_abs, 3),   # absolute pts (legacy)
            "STEPH_PCT"          : round(steph_pct, 3),   # % improvement (primary)
            "STEPH_ADJ"          : round(steph_adj, 3),   # quality-adjusted %
            "TQF"                : round(tqf, 3),
            "TM_TS_ON"           : round(ts_on  * 100, 2),
            "TM_TS_OFF"          : round(ts_off * 100, 2),
            "OWN_TS"             : round(player["OWN_TS"] * 100, 2),
            "LEAGUE_AVG_TS"      : round(league_avg_ts, 2),
            "TOE"                : round(TOE, 3),
            "P_SHOT_WEIGHT"      : round(p_w,  3),
            "TM_SHOT_WEIGHT"     : round(tm_w, 3),
            "CI_LO"              : round(ci_lo, 3) if not np.isnan(ci_lo) else np.nan,
            "CI_HI"              : round(ci_hi, 3) if not np.isnan(ci_hi) else np.nan,
            "P_VALUE"            : round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "SIG"                : stars,
            "MIN"                : player["MIN"],
            "GP"                 : player["GP"],
            "PPG"                : round(player["PTS"] / player["GP"], 2) if player["GP"] > 0 else np.nan,
            "APG"                : round(player["AST"] / player["GP"], 2) if player["GP"] > 0 else np.nan,
            "N_CORE_TM"          : len(core),
            "N_ON"               : len(on_rows),
            "N_OFF"              : len(off_rows),
            "TEAM"               : player["TEAM_ABBREVIATION"],
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("STEPH_PCT", ascending=False).reset_index(drop=True)


def compute_for_player(pid, season, lineups_df, player_df, league_avg_ts):
    """Single-player uncapped computation for the career pass."""
    pdf = player_df.copy()
    pdf["OWN_TS"]      = true_shooting(pdf["PTS"], pdf["FGA"], pdf["FTA"])
    pdf["FGA_PER_MIN"] = pdf["FGA"] / pdf["MIN"].replace(0, np.nan)
    pdf["FTA_PER_MIN"] = pdf["FTA"] / pdf["MIN"].replace(0, np.nan)
    pdf["PTS_PER_MIN"] = pdf["PTS"] / pdf["MIN"].replace(0, np.nan)

    pr = pdf[pdf["PLAYER_ID"] == pid]
    if pr.empty:
        return None

    player = pr.iloc[0]
    pname  = to_lineup_name(player["PLAYER_NAME"])
    rates  = (player["FGA_PER_MIN"], player["FTA_PER_MIN"], player["PTS_PER_MIN"])
    if any(pd.isna(r) for r in rates):
        return None

    team_ids = pdf.loc[pdf["PLAYER_ID"] == pid, "TEAM_ID"].tolist()
    tl       = lineups_df[lineups_df["TEAM_ID"].isin(team_ids)]
    if tl.empty:
        return None

    on_mask  = tl["GROUP_NAME"].str.contains(pname, regex=False, na=False)
    on_rows  = tl[on_mask]
    if on_rows.empty:
        return None

    core = get_core_teammates(on_rows, pname, MIN_SHARED_CAREER)
    if not core:
        return None

    off_mask = (~on_mask) & tl["GROUP_NAME"].apply(lambda g: lineup_has_core(g, core))
    off_rows = tl[off_mask]
    if off_rows.empty:
        return None

    ts_on  = teammate_ts_raw(on_rows,  *rates)
    ts_off = teammate_ts_raw(off_rows, 0, 0, 0)
    if np.isnan(ts_on) or np.isnan(ts_off) or ts_off <= 0:
        return None

    steph_pct = (ts_on - ts_off) / ts_off * 100
    tqf = league_avg_ts / (ts_off * 100) if league_avg_ts > 0 else 1.0
    steph_adj = steph_pct * tqf

    own_premium = (player["OWN_TS"] * 100) - league_avg_ts
    own_comp    = own_premium if own_premium >= 0 else own_premium * 1.5
    adj_comp    = steph_adj  if steph_adj   >= 0 else steph_adj   * 1.5
    p_pts       = player["PTS"]
    tm_pts_est  = ts_on * player["MIN"]
    total_vol   = p_pts + tm_pts_est
    p_w         = p_pts / total_vol if total_vol > 0 else 0.5
    tm_w        = 1 - p_w
    TOE         = own_comp * p_w + adj_comp * tm_w

    return {
        "PLAYER_ID"    : pid,
        "PLAYER_NAME"  : player["PLAYER_NAME"],
        "SEASON"       : season,
        "STEPH_ABS"    : round((ts_on - ts_off) * 100, 3),
        "STEPH_PCT"    : round(steph_pct, 3),
        "STEPH_ADJ"    : round(steph_adj, 3),
        "TQF"          : round(tqf, 3),
        "TM_TS_ON"     : round(ts_on  * 100, 2),
        "TM_TS_OFF"    : round(ts_off * 100, 2),
        "OWN_TS"       : round(player["OWN_TS"] * 100, 2),
        "LEAGUE_AVG_TS": round(league_avg_ts, 2),
        "TOE"          : round(TOE, 3),
        "P_SHOT_WEIGHT": round(p_w,  3),
        "TM_SHOT_WEIGHT": round(tm_w, 3),
        "CI_LO"        : np.nan,
        "CI_HI"        : np.nan,
        "P_VALUE"      : np.nan,
        "SIG"          : "",
        "MIN"          : player["MIN"],
        "GP"           : player["GP"],
        "PPG"          : round(player["PTS"] / player["GP"], 2) if player["GP"] > 0 else np.nan,
        "APG"          : round(player["AST"] / player["GP"], 2) if player["GP"] > 0 else np.nan,
        "N_CORE_TM"    : len(core),
        "N_ON"         : len(on_rows),
        "N_OFF"        : len(off_rows),
        "TEAM"         : player["TEAM_ABBREVIATION"],
        "QUALIFIED"    : False,
    }


# =============================================================================
# FETCH
# =============================================================================

def fetch_season(season, retries=3):
    for attempt in range(retries):
        try:
            print(f"  [{season}] lineups …", end=" ", flush=True)
            time.sleep(SLEEP_SEC)
            lin = LeagueDashLineups(
                season=season,
                measure_type_detailed_defense="Base",
                per_mode_detailed="Totals",
                group_quantity=5,
                last_n_games=0, month=0, opponent_team_id=0,
                pace_adjust="N",
                season_type_all_star="Regular Season",
                timeout=TIMEOUT,
            ).get_data_frames()[0]
            lin["GROUP_NAME"] = lin["GROUP_NAME"].apply(
                lambda g: ascii_name(g) if isinstance(g, str) else g)
            print(f"{len(lin):,}", end=" | ", flush=True)

            print("players …", end=" ", flush=True)
            time.sleep(SLEEP_SEC)
            pla = LeagueDashPlayerStats(
                season=season,
                measure_type_detailed_defense="Base",
                per_mode_detailed="Totals",
                season_type_all_star="Regular Season",
                timeout=TIMEOUT,
            ).get_data_frames()[0]

            pla["TS_PCT"] = true_shooting(pla["PTS"], pla["FGA"], pla["FTA"])
            avg_ts = float(pla["TS_PCT"].mean() * 100)
            print(f"{len(pla):,} | league avg TS={avg_ts:.1f}% ✓")
            return lin, pla, avg_ts

        except Exception as e:
            w = 2 ** attempt
            print(f"\n    attempt {attempt+1} failed: {e} — retry {w}s")
            time.sleep(w)

    print(f"  ✗ gave up on {season}")
    return None, None, None


# =============================================================================
# MAIN LOOP
# =============================================================================

print("=" * 65)
print("STEPH SCORE  ·  Shooting Transformation Elevation Per Hundred")
print("=" * 65)

all_results  = {}   # season → DataFrame (qualified players)
raw_data     = {}   # season → (lineups, players, league_avg_ts)
steph_leaders = {201939}   # Steph Curry seeded

for season in SEASONS:
    print(f"\n── {season} " + "─" * 40)
    lin, pla, avg_ts = fetch_season(season)
    if lin is None:
        print(f"  Skipping {season}")
        continue

    raw_data[season] = (lin, pla, avg_ts)
    df = compute_season(season, lin, pla, avg_ts)
    if df.empty:
        print(f"  No results for {season}")
        continue

    all_results[season] = df
    top5 = df[["PLAYER_NAME","STEPH_PCT","STEPH_ADJ","TOE","OWN_TS","SIG"]].head(5)
    print(f"  Scored {len(df)} players. Top 5:")
    print(top5.to_string(index=False))

    steph_leaders.update(df.head(20)["PLAYER_ID"].tolist())
    time.sleep(SLEEP_SEASON)

print(f"\nLeaders pool: {len(steph_leaders)} players")

# =============================================================================
# MASTER TABLE — all leaders × all seasons (uncapped career pass)
# =============================================================================

print("\nBuilding master table (uncapped career pass) …")

pid_to_name = {}
for df in all_results.values():
    for _, r in df.iterrows():
        pid_to_name.setdefault(r["PLAYER_ID"], r["PLAYER_NAME"])

master_rows = []
for season, (lin, pla, avg_ts) in raw_data.items():
    sdf = all_results.get(season, pd.DataFrame())
    for pid in steph_leaders:
        match = sdf[sdf["PLAYER_ID"] == pid] if not sdf.empty else pd.DataFrame()
        if not match.empty:
            row = match.iloc[0].to_dict()
            row.setdefault("QUALIFIED", True)
        else:
            row = compute_for_player(pid, season, lin, pla, avg_ts)
            if row is None:
                row = {
                    "PLAYER_ID": pid, "PLAYER_NAME": pid_to_name.get(pid, f"ID_{pid}"),
                    "SEASON": season, "STEPH_PCT": np.nan, "STEPH_ADJ": np.nan,
                    "STEPH_ABS": np.nan, "TOE": np.nan, "OWN_TS": np.nan,
                    "MIN": np.nan, "QUALIFIED": False,
                }
        master_rows.append(row)

master = pd.DataFrame(master_rows)

# =============================================================================
# CAREER AVERAGES  (regularized, minute-weighted, ≥ MIN_CAREER_MIN filter)
# =============================================================================

eligible = master[
    master["STEPH_PCT"].notna() &
    (master["MIN"].fillna(0) >= MIN_CAREER_MIN)
].copy()

# League-wide reference values across all eligible rows
LG_STEPH_PCT  = eligible["STEPH_PCT"].mean()
LG_STEPH_ADJ  = eligible["STEPH_ADJ"].mean()
LG_TOE        = eligible["TOE"].mean()
LG_OWN_TS     = eligible["OWN_TS"].mean()
AVG_SEASON_MIN = eligible["MIN"].mean()

print(f"\nLeague averages over eligible seasons:")
print(f"  STEPH_PCT: {LG_STEPH_PCT:.3f}%  |  STEPH_ADJ: {LG_STEPH_ADJ:.3f}  |  TOE: {LG_TOE:.3f}")


def reg_wavg(g, col, prior_val, prior_weight):
    """Bayesian-regularized minute-weighted average."""
    ws = (g[col] * g["MIN"]).sum() + prior_val * prior_weight
    wt = g["MIN"].sum() + prior_weight
    return ws / wt if wt > 0 else np.nan


seasons_count = (
    eligible.groupby(["PLAYER_ID", "PLAYER_NAME"])["SEASON"]
    .count().reset_index().rename(columns={"SEASON": "SEASONS"})
)

pw = REGULARIZATION * AVG_SEASON_MIN
career = (
    eligible.groupby(["PLAYER_ID", "PLAYER_NAME"])
    .apply(lambda g: pd.Series({
        "AVG_STEPH_PCT" : reg_wavg(g, "STEPH_PCT", LG_STEPH_PCT, pw),
        "AVG_STEPH_ADJ" : reg_wavg(g, "STEPH_ADJ", LG_STEPH_ADJ, pw),
        "AVG_TOE"       : reg_wavg(g, "TOE",       LG_TOE,       pw),
        "AVG_OWN_TS"    : reg_wavg(g, "OWN_TS",    LG_OWN_TS,    pw),
        "TOTAL_MIN"     : g["MIN"].sum(),
    }))
    .reset_index()
    .merge(seasons_count, on=["PLAYER_ID","PLAYER_NAME"], how="left")
    .sort_values("AVG_STEPH_PCT", ascending=False)
    .reset_index(drop=True)
)
career.index += 1

# Reliability: 3+ seasons = 1.0, scales linearly below
career["RELIABILITY"] = (career["SEASONS"] / 3).clip(upper=1.0)
career["TIER"] = career["SEASONS"].apply(
    lambda s: "Established (3+ seasons)" if s >= 3
              else ("Developing (2 seasons)" if s == 2
                    else "Emerging (1 season)")
)

print(f"\n── Career STEPH_PCT Leaderboard ──")
print(career[["PLAYER_NAME","AVG_STEPH_PCT","AVG_STEPH_ADJ","AVG_TOE","SEASONS","TIER"]].head(20).to_string())

# =============================================================================
# VISUALISATION HELPERS
# =============================================================================

def fig_ax(w=14, h=8):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    dark_ax(ax)
    return fig, ax


def hline(ax, val, label, color=MUTED, lw=1.2, ls="--"):
    ax.axhline(val, color=color, linewidth=lw, linestyle=ls, alpha=0.7, zorder=1)
    ax.text(ax.get_xlim()[1] * 0.98 if ax.get_xlim()[1] else 0.98,
            val, f" {label}", va="bottom", ha="right",
            color=color, fontsize=8, alpha=0.8)


def vline(ax, val, label, color=MUTED, lw=1.0, ls=":"):
    ax.axvline(val, color=color, linewidth=lw, linestyle=ls, alpha=0.7, zorder=1)


def annotate_scatter(ax, xs, ys, labels, colors=None, fontsize=9, top_n=10):
    """
    Annotate top_n points in a scatter with anti-overlap offsets.
    labels: list of strings, colors: list or single color.
    """
    if colors is None:
        colors = [TEXT] * len(labels)
    elif isinstance(colors, str):
        colors = [colors] * len(labels)

    # sort by Y descending, annotate top_n
    order = np.argsort(ys)[::-1][:top_n]
    sel_x  = np.array(xs)[order]
    sel_y  = np.array(ys)[order]
    sel_l  = np.array(labels)[order]
    sel_c  = np.array(colors)[order]

    offsets = no_overlap_offsets(sel_x, sel_y)

    for x, y, lbl, col, (dx, dy) in zip(sel_x, sel_y, sel_l, sel_c, offsets):
        ax.annotate(
            lbl, (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            color=col, fontsize=fontsize, fontweight="bold",
            path_effects=shadow(),
            bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none", alpha=0.75),
            ha="left", va="bottom",
        )


def bar_chart(ax, names, values, bar_colors, bar_labels, lg_val, lg_label,
              xlabel, score_fmt="{:.2f}"):
    y = np.arange(len(names))
    bars = ax.barh(y, values, color=bar_colors, edgecolor="none", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=TEXT, fontsize=9)
    ax.axvline(lg_val, color=MUTED, lw=1.2, ls="--", alpha=0.7, label=f"Lg avg ({lg_val:.2f})")
    # score labels on bars
    for i, (v, lbl) in enumerate(zip(values, bar_labels)):
        xpos = v + abs(values.max() - values.min()) * 0.01
        ax.text(xpos, i, lbl, va="center", ha="left",
                color=TEXT, fontsize=8.5, fontweight="bold", path_effects=shadow())
    ax.set_xlabel(xlabel)
    ax.legend(loc="lower right")
    # x-padding
    xmax = max(values.max(), lg_val)
    xmin = min(values.min(), 0)
    ax.set_xlim(xmin - abs(xmax - xmin) * 0.05, xmax + abs(xmax - xmin) * 0.22)


# =============================================================================
# PLOT 1 — Per-season scatter:  OWN_TS (x)  vs  STEPH_PCT (y)
# =============================================================================

for season, df_s in all_results.items():
    fig, ax = fig_ax(13, 8)

    avg_s  = df_s["STEPH_PCT"].mean()
    avg_ts = df_s["OWN_TS"].mean()

    dot_c = np.where(df_s["STEPH_PCT"] >= avg_s, GOLD, RED)
    sizes = (df_s["MIN"] / df_s["MIN"].max() * 200 + 30).values

    sc = ax.scatter(df_s["OWN_TS"], df_s["STEPH_PCT"],
                    c=dot_c, s=sizes, alpha=0.80,
                    edgecolors=BG, linewidths=0.6, zorder=3)

    ax.axhline(avg_s,  color=MUTED, lw=1.0, ls="--", alpha=0.7,
               label=f"Lg avg STEPH_PCT  ({avg_s:.2f}%)")
    ax.axvline(avg_ts, color=MUTED, lw=1.0, ls=":",  alpha=0.7,
               label=f"Lg avg TS%  ({avg_ts:.1f}%)")

    # Y-axis padding
    yr = df_s["STEPH_PCT"].max() - df_s["STEPH_PCT"].min()
    ax.set_ylim(df_s["STEPH_PCT"].min() - yr*0.15, df_s["STEPH_PCT"].max() + yr*0.50)

    top10 = df_s.head(10)
    # MODIFIED: Include team in label
    labels = [f"{r['PLAYER_NAME']} ({r['TEAM']})  {r['STEPH_PCT']:+.2f}%{r['SIG']}"
              for _, r in top10.iterrows()]
    annotate_scatter(ax, top10["OWN_TS"].values, top10["STEPH_PCT"].values,
                     labels, colors=GOLD, fontsize=8, top_n=10)

    ax.set_xlabel("Player Own True Shooting %  (personal scoring efficiency)")
    ax.set_ylabel("STEPH Score — Relative Teammate TS% Improvement  (%)")
    ax.set_title(
        f"{season}  ·  Offensive Gravity vs Personal Efficiency\n"
        f"Size = minutes played  ·  Gold = above-avg gravity  ·  "
        f"*** p<0.001  ** p<0.01  * p<0.05",
        pad=12
    )
    legend_els = [
        Line2D([0],[0], marker="o", color="none", markerfacecolor=GOLD,
               markersize=8, label="Above avg gravity"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=RED,
               markersize=8, label="Below avg gravity"),
    ]
    ax.legend(handles=legend_els + [
        Line2D([0],[0], color=MUTED, lw=1, ls="--", label=f"Avg STEPH_PCT {avg_s:.2f}%"),
        Line2D([0],[0], color=MUTED, lw=1, ls=":",  label=f"Avg TS% {avg_ts:.1f}%"),
    ], loc="upper left", framealpha=0.85)

    plt.tight_layout()
    fname = f"scatter_{season.replace('-','_')}.png"
    plt.savefig(fname, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"✓  {fname}")


# =============================================================================
# PLOT 2 — Career STEPH_PCT scatter:  seasons (x)  vs  avg STEPH_PCT (y)
# =============================================================================

fig, ax = fig_ax(13, 8)

dot_c = np.where(career["AVG_STEPH_PCT"] >= LG_STEPH_PCT, GOLD, RED)
sizes = (career["TOTAL_MIN"] / career["TOTAL_MIN"].max() * 280 + 40).values

ax.scatter(career["SEASONS"], career["AVG_STEPH_PCT"],
           c=dot_c, s=sizes, alpha=0.82, edgecolors=BG, linewidths=0.6, zorder=3)

ax.axhline(LG_STEPH_PCT, color=MUTED, lw=1.2, ls="--", alpha=0.7,
           label=f"League avg STEPH_PCT  ({LG_STEPH_PCT:.2f}%)")

yr = career["AVG_STEPH_PCT"].max() - career["AVG_STEPH_PCT"].min()
ax.set_ylim(career["AVG_STEPH_PCT"].min() - yr*0.18, career["AVG_STEPH_PCT"].max() + yr*0.38)
ax.set_xticks(range(1, int(career["SEASONS"].max()) + 1))

top10_c = career.head(10)
labels  = [f"{r['PLAYER_NAME']}  {r['AVG_STEPH_PCT']:+.2f}%  ({int(r['SEASONS'])}s)"
           for _, r in top10_c.iterrows()]
annotate_scatter(ax, top10_c["SEASONS"].values, top10_c["AVG_STEPH_PCT"].values,
                 labels, colors=GOLD, fontsize=8.5, top_n=10)

ax.set_xlabel("Seasons With Sufficient Data (reliability proxy — more = more trustworthy)")
ax.set_ylabel("Career Avg STEPH Score  — Relative Teammate TS% Improvement  (%)")
ax.set_title(
    f"Career Gravity Score  ·  {SEASONS[0]}–{SEASONS[-1]}\n"
    f"Regularized minute-weighted avg  ·  Size = total minutes  ·  "
    f"Dashed = league average",
    pad=12
)
ax.legend(loc="upper left", framealpha=0.85)
plt.tight_layout()
plt.savefig("career_steph_scatter.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓  career_steph_scatter.png")


# =============================================================================
# PLOT 3 — Career STEPH_PCT bar chart — Established vs Emerging split
# =============================================================================

established = career[career["SEASONS"] >= 3].head(15).sort_values("AVG_STEPH_PCT")
emerging    = career[career["SEASONS"] < 3].head(10).sort_values("AVG_STEPH_PCT")

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 9))
fig.patch.set_facecolor(BG)
for ax in (ax_l, ax_r): dark_ax(ax)

# Left — Established
bar_chart(
    ax_l,
    names      = established["PLAYER_NAME"].values,
    values     = established["AVG_STEPH_PCT"].values,
    bar_colors = [GOLD if v >= LG_STEPH_PCT else RED for v in established["AVG_STEPH_PCT"]],
    bar_labels = [f"{v:+.2f}%  ({int(s)}s)" for v, s in
                  zip(established["AVG_STEPH_PCT"], established["SEASONS"])],
    lg_val     = LG_STEPH_PCT,
    lg_label   = f"Lg avg ({LG_STEPH_PCT:.2f}%)",
    xlabel     = "Career Avg Relative Teammate TS% Improvement  (STEPH_PCT, %)",
)
ax_l.set_title("Established Gravity  ·  3+ Seasons\n(High reliability — sample size confirmed)", pad=10)

# Right — Emerging
bar_chart(
    ax_r,
    names      = emerging["PLAYER_NAME"].values,
    values     = emerging["AVG_STEPH_PCT"].values,
    bar_colors = [PURPLE if s == 1 else TEAL for s in emerging["SEASONS"]],
    bar_labels = [f"{v:+.2f}%  ({int(s)}s)" for v, s in
                  zip(emerging["AVG_STEPH_PCT"], emerging["SEASONS"])],
    lg_val     = LG_STEPH_PCT,
    lg_label   = f"Lg avg ({LG_STEPH_PCT:.2f}%)",
    xlabel     = "Career Avg Relative Teammate TS% Improvement  (STEPH_PCT, %)",
)
ax_r.set_title("Emerging Gravity  ·  1–2 Seasons\n(Purple = 1 season — interpret with caution)", pad=10)

fig.suptitle(
    f"Career STEPH Score  ·  Established vs Emerging  ·  {SEASONS[0]}–{SEASONS[-1]}\n"
    f"Scores show relative % improvement in teammate TS% when player is on the floor",
    fontsize=13, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("career_steph_reliability.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓  career_steph_reliability.png")


# =============================================================================
# PLOT 4 — TOE Career Bar Chart
# =============================================================================

toe_top = career.nlargest(20, "AVG_TOE").sort_values("AVG_TOE")

fig, ax = fig_ax(13, 10)
bar_chart(
    ax,
    names      = toe_top["PLAYER_NAME"].values,
    values     = toe_top["AVG_TOE"].values,
    bar_colors = [TEAL if v >= LG_TOE else RED for v in toe_top["AVG_TOE"]],
    bar_labels = [f"{v:+.2f}  ({int(s)}s)" for v, s in
                  zip(toe_top["AVG_TOE"], toe_top["SEASONS"])],
    lg_val     = LG_TOE,
    lg_label   = f"Lg avg ({LG_TOE:.2f})",
    xlabel     = "Total Offensive Efficiency  (own TS% premium + teammate gravity, volume-weighted)",
)
ax.set_title(
    f"Career Total Offensive Efficiency (TOE)  ·  Top 20  ·  {SEASONS[0]}–{SEASONS[-1]}\n"
    "Own shooting premium + teammate gravity weighted by shot-volume share",
    pad=12
)
plt.tight_layout()
plt.savefig("career_toe_bar.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓  career_toe_bar.png")


# =============================================================================
# PLOT 5 — Career TOE scatter:  STEPH_ADJ (x)  vs  TOE (y)
# =============================================================================

fig, ax = fig_ax(13, 8)

dot_c = np.where(career["AVG_TOE"] >= LG_TOE, TEAL, RED)
sizes = (career["TOTAL_MIN"] / career["TOTAL_MIN"].max() * 250 + 40).values

ax.scatter(career["AVG_STEPH_ADJ"], career["AVG_TOE"],
           c=dot_c, s=sizes, alpha=0.82, edgecolors=BG, linewidths=0.6, zorder=3)

ax.axhline(LG_TOE,       color=MUTED, lw=1.0, ls="--", alpha=0.7,
           label=f"Lg avg TOE ({LG_TOE:.2f})")
ax.axvline(LG_STEPH_ADJ, color=MUTED, lw=1.0, ls=":",  alpha=0.7,
           label=f"Lg avg STEPH_ADJ ({LG_STEPH_ADJ:.2f})")

yr = career["AVG_TOE"].max() - career["AVG_TOE"].min()
ax.set_ylim(career["AVG_TOE"].min() - yr*0.12, career["AVG_TOE"].max() + yr*0.38)

top10_t = career.nlargest(10, "AVG_TOE")
labels  = [f"{r['PLAYER_NAME']}  TOE {r['AVG_TOE']:+.2f}"
           for _, r in top10_t.iterrows()]
annotate_scatter(ax, top10_t["AVG_STEPH_ADJ"].values, top10_t["AVG_TOE"].values,
                 labels, colors=TEAL, fontsize=8.5, top_n=10)

ax.set_xlabel("Career Avg Quality-Adjusted STEPH  (teammate gravity, baseline-adjusted)")
ax.set_ylabel("Career Avg TOE  (combined own efficiency + teammate gravity)")
ax.set_title(
    f"Career TOE vs Quality-Adjusted Gravity  ·  {SEASONS[0]}–{SEASONS[-1]}\n"
    "Teal = above avg TOE  ·  Size = total minutes  ·  "
    "Top-right = elite all-around offensive contributors",
    pad=12
)
ax.legend(loc="upper left", framealpha=0.85)
plt.tight_layout()
plt.savefig("career_toe_scatter.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓  career_toe_scatter.png")


# =============================================================================
# PLOT 6 — Top 10 single-season STEPH_PCT
# =============================================================================

all_df  = pd.concat(all_results.values(), ignore_index=True)
top10ss = all_df.nlargest(10, "STEPH_PCT").reset_index(drop=True)
top10ss["RANK"] = range(1, 11)

# MODIFIED: Include team in label
def ss_label(r):
    p  = r["PLAYER_NAME"].split(" ", 1)
    nm = f"{p[0][0]}.{p[1]}" if len(p) == 2 else p[0]
    return f"{nm} ({r['TEAM']})\n{short_season(r['SEASON'])}\n{r['STEPH_PCT']:+.2f}%{r['SIG']}"

top10ss["LABEL"] = top10ss.apply(ss_label, axis=1)

unique_p  = top10ss["PLAYER_NAME"].unique()
pal = [GOLD, TEAL, RED, PURPLE, BLUE, "#FF9F1C", "#2EC4B6", "#E71D36", "#F5C842", "#A8DADC"]
pcolor = {p: pal[i % len(pal)] for i, p in enumerate(unique_p)}

fig, ax = fig_ax(13, 8)

# MODIFIED: Better vertical spacing for labels
for i, row in top10ss.iterrows():
    col  = pcolor[row["PLAYER_NAME"]]
    ax.scatter(row["RANK"], row["STEPH_PCT"],
               color=col, s=200, zorder=4, edgecolors=TEXT, linewidths=1.2)

    # Stagger vertical offsets more aggressively
    tier = i // 3  # Group into tiers of 3
    oy = 22 + (tier * 8) if i % 2 == 0 else -32 - (tier * 8)

    ax.annotate(
        row["LABEL"],
        (row["RANK"], row["STEPH_PCT"]),
        textcoords="offset points", xytext=(0, oy),
        ha="center", va="bottom" if oy > 0 else "top",
        color=col, fontsize=8.5, fontweight="bold",
        path_effects=shadow(),
        bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec="none", alpha=0.78),
    )

ax.axhline(LG_STEPH_PCT, color=MUTED, lw=1.2, ls="--", alpha=0.7,
           label=f"Lg avg STEPH_PCT ({LG_STEPH_PCT:.2f}%)")

yr = top10ss["STEPH_PCT"].max() - top10ss["STEPH_PCT"].min()
ax.set_ylim(top10ss["STEPH_PCT"].min() - yr*0.6, top10ss["STEPH_PCT"].max() + yr*0.9)
ax.set_xticks(top10ss["RANK"])
ax.set_xticklabels([f"#{r}" for r in top10ss["RANK"]])

ax.set_xlabel("Season Rank  (#1 = Highest Single-Season STEPH Score in Sample)")
ax.set_ylabel("STEPH Score — Relative Teammate TS% Improvement  (%)")
ax.set_title(
    f"Top 10 Single-Season STEPH Scores  ·  {SEASONS[0]}–{SEASONS[-1]}\n"
    "Same player may appear multiple times  ·  Significance: *** p<0.001  ** p<0.01  * p<0.05",
    pad=12
)
ax.legend(loc="lower right", framealpha=0.85)
plt.tight_layout()
plt.savefig("top10_single_season.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓  top10_single_season.png")


# =============================================================================
# PLOT 7 — Quality-Adjusted STEPH_ADJ career bar chart
# =============================================================================

adj_top = career.nlargest(20, "AVG_STEPH_ADJ").sort_values("AVG_STEPH_ADJ")

fig, ax = fig_ax(13, 10)
bar_chart(
    ax,
    names      = adj_top["PLAYER_NAME"].values,
    values     = adj_top["AVG_STEPH_ADJ"].values,
    bar_colors = [TEAL if v >= LG_STEPH_ADJ else RED for v in adj_top["AVG_STEPH_ADJ"]],
    bar_labels = [f"{v:+.2f}  (TQF×{t:.2f}, {int(s)}s)" for v, t, s in
                  zip(adj_top["AVG_STEPH_ADJ"],
                      eligible.groupby("PLAYER_ID")["TQF"].mean().reindex(adj_top["PLAYER_ID"]).fillna(1).values,
                      adj_top["SEASONS"])],
    lg_val     = LG_STEPH_ADJ,
    lg_label   = f"Lg avg ({LG_STEPH_ADJ:.2f})",
    xlabel     = "Quality-Adjusted STEPH  (teammate % improvement × teammate-quality factor)",
)
ax.set_title(
    f"Career Quality-Adjusted Gravity  ·  Top 20  ·  {SEASONS[0]}–{SEASONS[-1]}\n"
    "Adjusts for difficulty: improving already-efficient teammates scores lower",
    pad=12
)
plt.tight_layout()
plt.savefig("career_steph_adj_bar.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓  career_steph_adj_bar.png")


# =============================================================================
# EXPORT CSVs
# =============================================================================

master_csv_cols = [
    "PLAYER_NAME","PLAYER_ID","SEASON","TEAM",
    "STEPH_PCT","STEPH_ADJ","STEPH_ABS","TQF",
    "TM_TS_ON","TM_TS_OFF","OWN_TS","LEAGUE_AVG_TS",
    "TOE","P_SHOT_WEIGHT","TM_SHOT_WEIGHT",
    "CI_LO","CI_HI","P_VALUE","SIG",
    "MIN","GP","PPG","APG","N_CORE_TM","N_ON","N_OFF",
]
master[[c for c in master_csv_cols if c in master.columns]].sort_values(
    ["PLAYER_NAME","SEASON"]
).to_csv("steph_all_seasons.csv", index=False)

career_csv_cols = [
    "PLAYER_NAME","PLAYER_ID","SEASONS","TIER","RELIABILITY",
    "AVG_STEPH_PCT","AVG_STEPH_ADJ","AVG_TOE","AVG_OWN_TS","TOTAL_MIN",
]
career[[c for c in career_csv_cols if c in career.columns]].to_csv(
    "steph_career_averages.csv", index_label="RANK")

print("\n✅  All done.")
print("   Plots : scatter_<season>.png          (per-season scatter)")
print("           career_steph_scatter.png       (career gravity scatter)")
print("           career_steph_reliability.png   (established vs emerging bar)")
print("           career_toe_bar.png             (TOE leaderboard bar)")
print("           career_toe_scatter.png         (TOE vs STEPH_ADJ career scatter)")
print("           top10_single_season.png        (top 10 individual season scores)")
print("           career_steph_adj_bar.png       (quality-adjusted gravity bar)")
print("   CSVs  : steph_all_seasons.csv")
print("           steph_career_averages.csv")