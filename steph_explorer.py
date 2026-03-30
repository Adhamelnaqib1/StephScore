# steph_explorer.py
# =============================================================================
# STEPH Score Explorer — Streamlit app
# STEPH Score Explorer — Interactive Streamlit App
#
# Run:   streamlit run steph_explorer.py
# Needs: pip install streamlit plotly pandas
# Data:  steph_all_seasons.csv  (produced by steph_score.py)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STEPH Score Explorer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark theme CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp, [data-testid="stAppViewContainer"] { background: #0D1117; color: #E6EDF3; }
  [data-testid="stSidebar"]                   { background: #161B22; }
  [data-testid="stSidebar"] *                 { color: #E6EDF3 !important; }
  h1, h2, h3                                  { color: #F0B429 !important; }
  .metric-card { background:#161B22; border-radius:8px; padding:14px 16px; margin-bottom:6px; }
  .metric-label { font-size:11px; color:#8B949E; margin-bottom:2px; }
  .metric-value { font-size:24px; font-weight:700; color:#E6EDF3; }
  .card { background:#161B22; border-radius:8px; padding:14px 16px; margin-bottom:8px; }
  .card-label { font-size:11px; color:#8B949E; margin-bottom:2px; }
  .card-value { font-size:22px; font-weight:700; color:#E6EDF3; }
  .stat-row { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #21262D; font-size:13px; }
  .stat-name { color:#8B949E; }
  .stat-val { color:#E6EDF3; font-weight:600; }
  .compare-header { font-size:15px; font-weight:700; color:#F0B429; padding:10px 0 6px; border-bottom:2px solid #F0B429; margin-bottom:8px; }
  .stMultiSelect span[data-baseweb="tag"] { background:#21262D !important; color:#E6EDF3 !important; }
  div[data-testid="stSelectbox"] label,
  div[data-testid="stMultiSelect"] label { color:#8B949E !important; font-size:12px; }
  .stRadio label { color: #E6EDF3 !important; }
  hr { border-color: #21262D; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS & STAT DEFINITIONS
# =============================================================================

STATS = {
    "STEPH_ABS"  : "STEPH Absolute  (teammate TS% pts gained)",
    "STEPH_PCT"  : "STEPH %  (relative teammate TS% improvement)",
    "STEPH_ADJ"  : "STEPH Adjusted %  (quality-adjusted gravity)",
    "OWN_TS"     : "Own True Shooting %",
    "PPG"        : "Points Per Game",
    "APG"        : "Assists Per Game",
    "TOE"        : "Total Offensive Efficiency (TOE)",
}

STAT_SHORT = {
    "STEPH_ABS" : "STEPH Abs",
    "STEPH_PCT" : "STEPH %",
    "STEPH_ADJ" : "STEPH Adj %",
    "OWN_TS"    : "TS%",
    "PPG"       : "PPG",
    "APG"       : "APG",
    "TOE"       : "TOE",
}

# All stats shown in Player Stats / Compare modes
ALL_STAT_COLS = [
    "SEASON","TEAM","STEPH_PCT","STEPH_ADJ","STEPH_ABS",
    "OWN_TS","TOE","PPG","APG","TQF",
    "TM_TS_ON","TM_TS_OFF","LEAGUE_AVG_TS",
    "P_SHOT_WEIGHT","TM_SHOT_WEIGHT",
    "CI_LO","CI_HI","P_VALUE","SIG",
    "MIN","GP","N_CORE_TM","N_ON","N_OFF",
]

STAT_DESC = {
    "STEPH_PCT"       : "Relative teammate TS% improvement (%)",
    "STEPH_ADJ"       : "Quality-adjusted teammate improvement",
    "STEPH_ABS"       : "Absolute teammate TS% pts gained",
    "OWN_TS"          : "Player's own True Shooting %",
    "TOE"             : "Total Offensive Efficiency (own + gravity)",
    "PPG"             : "Points per game",
    "APG"             : "Assists per game",
    "TQF"             : "Teammate Quality Factor",
    "TM_TS_ON"        : "Teammate TS% when player is ON",
    "TM_TS_OFF"       : "Teammate TS% when player is OFF",
    "LEAGUE_AVG_TS"   : "League average TS% that season",
    "P_SHOT_WEIGHT"   : "Player's shot volume share",
    "TM_SHOT_WEIGHT"  : "Teammates' shot volume share",
    "CI_LO"           : "Bootstrap 95% CI lower bound",
    "CI_HI"           : "Bootstrap 95% CI upper bound",
    "P_VALUE"         : "Bootstrap p-value (H0: STEPH=0)",
    "SIG"             : "Significance stars",
    "MIN"             : "Minutes played",
    "GP"              : "Games played",
    "N_CORE_TM"       : "Number of core teammates",
    "N_ON"            : "Lineup rows in ON sample",
    "N_OFF"           : "Lineup rows in OFF sample",
}

# Seasons in scope — 2007-08 onwards, no COVID year
VALID_SEASONS = [
    "2007-08","2008-09","2009-10","2010-11","2011-12","2012-13",
    "2013-14","2014-15","2015-16","2016-17","2017-18","2018-19",
    "2020-21","2021-22","2022-23","2023-24","2024-25",
]

# Plotly dark theme base
PLOTLY_THEME = dict(
    plot_bgcolor  = "#161B22",
    paper_bgcolor = "#0D1117",
    font          = dict(family="Inter, system-ui, sans-serif", color="#E6EDF3", size=12),
    xaxis         = dict(gridcolor="#21262D", zerolinecolor="#30363D",
                         title_font=dict(color="#8B949E", size=12),
                         tickfont=dict(color="#8B949E")),
    yaxis         = dict(gridcolor="#21262D", zerolinecolor="#30363D",
                         title_font=dict(color="#8B949E", size=12),
                         tickfont=dict(color="#8B949E")),
    legend        = dict(bgcolor="#161B22", bordercolor="#21262D", borderwidth=1,
                         font=dict(color="#E6EDF3", size=10)),
    hoverlabel    = dict(bgcolor="#21262D", bordercolor="#30363D",
                         font=dict(color="#E6EDF3", size=12)),
    margin        = dict(l=60, r=40, t=90, b=60),
)

GOLD   = "#F0B429"
RED    = "#E05C5C"
TEAL   = "#26C6A0"
BLUE   = "#58A6FF"
MUTED  = "#8B949E"
PANEL  = "#161B22"
PURPLE = "#BC8CFF"

PLAYER_PALETTE = [
    "#F0B429","#26C6A0","#58A6FF","#BC8CFF","#FF6B6B",
    "#FF9F1C","#2EC4B6","#E71D36","#F5C842","#A8DADC",
    "#FFB347","#87CEEB","#DDA0DD","#98FB98","#F08080",
]

# =============================================================================
# DATA LOADING
# =============================================================================

REQUIRED = ["PLAYER_NAME","PLAYER_ID","SEASON","TEAM",
            "STEPH_ABS","STEPH_PCT","STEPH_ADJ","OWN_TS","TOE","GP","MIN"]

@st.cache_data
def load(path="steph_all_seasons.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None, f"`steph_all_seasons.csv` not found. Run `steph_score.py` first."

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, f"Missing columns: {missing}. Re-run `steph_score.py`."

    df["SEASON"] = df["SEASON"].astype(str)

    # Derive PPG/APG if missing (older CSV)
    if "PPG" not in df.columns:
        df["PPG"] = np.nan
    if "APG" not in df.columns:
        df["APG"] = np.nan

    return df, None


@st.cache_data
def build_career(df):
    """One row per player — minute-weighted average across all seasons in df."""
    num_cols = [c for c in STATS if c in df.columns]
    grp = df.groupby(["PLAYER_NAME","PLAYER_ID"])

    rows = []
    for (pname, pid), g in grp:
        w = g["MIN"].fillna(0)
        total_min = w.sum()
        if total_min == 0:
            continue
        row = {"PLAYER_NAME": pname, "PLAYER_ID": pid,
               "SEASONS": g.loc[g["MIN"].fillna(0) > 0, "SEASON"].nunique(), "TOTAL_MIN": round(total_min, 0),
               "TEAMS": ", ".join(g["TEAM"].dropna().unique())}
        for col in num_cols:
            if col in g.columns and g[col].notna().any():
                valid = g[col].notna()
                row[col] = round(float(np.average(g.loc[valid, col], weights=w[valid])), 3)
            else:
                row[col] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values("PLAYER_NAME").reset_index(drop=True)


df_full, err = load()
career_df = None if df_full is None else build_career(df_full)

# =============================================================================
# SIDEBAR — MODE SELECTOR
# =============================================================================

with st.sidebar:
    st.markdown("## 🏀 STEPH Explorer")
    st.caption("Shooting Transformation Elevation Per Hundred")
    st.markdown("---")

    if err:
        st.error(err)
        st.stop()

    mode = st.radio(
        "Mode",
        ["Graph", "Player Stats", "Compare"],
        index=0,
        help="Graph: charts across stats. Player Stats: lookup tables. Compare: side-by-side.",
    )

# =============================================================================
# MODE: GRAPH
# =============================================================================

if mode == "Graph":

    with st.sidebar:
        st.markdown("---")

        # Season scope
        SCOPE_OPTIONS = ["Career Average", "All Seasons"] + sorted(
            [s for s in VALID_SEASONS if s in df_full["SEASON"].unique()],
            reverse=True,
        )
        scope = st.selectbox(
            "Season Scope",
            SCOPE_OPTIONS,
            help=(
                "'Career Average' → one dot per player, averaged across all their seasons.\n"
                "'All Seasons' → every player-season as its own dot.\n"
                "Or pick a specific season."
            ),
        )

        # Determine working dataframe
        if scope == "Career Average":
            df_plot_base = career_df.copy()
            is_career = True
        elif scope == "All Seasons":
            df_plot_base = df_full.copy()
            is_career = False
        else:
            df_plot_base = df_full[df_full["SEASON"] == scope].copy()
            is_career = False

        # Player highlight
        player_options = sorted(df_plot_base["PLAYER_NAME"].dropna().unique())
        selected_players = st.multiselect(
            "Highlight Player(s)",
            player_options,
            default=[],
            placeholder="Type a name…",
        )

        st.markdown("---")

        chart_type = st.radio("Chart Type", ["Scatter", "Bar"], horizontal=True)

        st.markdown("---")

        stat_keys = list(STATS.keys())
        avail = [k for k in stat_keys if k in df_plot_base.columns]

        if chart_type == "Scatter":
            x_stat = st.selectbox("X-Axis", avail,
                                  index=avail.index("OWN_TS") if "OWN_TS" in avail else 0,
                                  format_func=lambda k: STATS[k])
            y_stat = st.selectbox("Y-Axis", avail,
                                  index=avail.index("STEPH_PCT") if "STEPH_PCT" in avail else 1,
                                  format_func=lambda k: STATS[k])
            show_lg   = st.toggle("Show league average lines", value=True)
            size_mins = st.toggle("Size dots by minutes", value=True)
        else:
            bar_stat  = st.selectbox("Stat", avail,
                                     index=avail.index("TOE") if "TOE" in avail else 0,
                                     format_func=lambda k: STATS[k])
            top_n     = st.slider("Top N", 5, 30, 20)
            sort_asc  = st.radio("Sort", ["Descending", "Ascending"], horizontal=True) == "Ascending"
            show_lg   = st.toggle("Show league average line", value=True)

    # ── Header ────────────────────────────────────────────────────────────────
    scope_label = scope if scope not in ("Career Average","All Seasons") else scope
    st.markdown(f"## Graph  ·  {scope_label}")

    # ── Summary cards ─────────────────────────────────────────────────────────
    n_players = df_plot_base["PLAYER_NAME"].nunique()
    n_rows    = len(df_plot_base)
    avg_steph = df_plot_base["STEPH_PCT"].mean() if "STEPH_PCT" in df_plot_base.columns else np.nan
    avg_toe   = df_plot_base["TOE"].mean() if "TOE" in df_plot_base.columns else np.nan

    c1,c2,c3,c4 = st.columns(4)
    for col, lbl, val in zip(
        [c1,c2,c3,c4],
        ["Players","Data points","Avg STEPH %","Avg TOE"],
        [f"{n_players:,}", f"{n_rows:,}",
         f"{avg_steph:.2f}%" if not np.isnan(avg_steph) else "—",
         f"{avg_toe:.2f}"   if not np.isnan(avg_toe)   else "—"],
    ):
        col.markdown(
            f"<div class='card'><div class='card-label'>{lbl}</div>"
            f"<div class='card-value'>{val}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── SCATTER ───────────────────────────────────────────────────────────────
    if chart_type == "Scatter":
        df_sc = df_plot_base.dropna(subset=[x_stat, y_stat]).copy()
        if df_sc.empty:
            st.warning("No data for the chosen axes and scope.")
            st.stop()

        lg_x = df_sc[x_stat].mean()
        lg_y = df_sc[y_stat].mean()
        max_min = df_sc["MIN"].max() if ("MIN" in df_sc.columns and size_mins) else 1

        is_hi = df_sc["PLAYER_NAME"].isin(selected_players)
        df_bg = df_sc[~is_hi]
        df_hi = df_sc[is_hi]

        def sz(sub, scale=20, base=5):
            if size_mins and "MIN" in sub.columns:
                return ((sub["MIN"].fillna(0) / max_min * scale) + base).values
            return np.full(len(sub), 8)

        fig = go.Figure()

        # Background dots
        bg_c = np.where(df_bg[y_stat] >= lg_y, GOLD, RED)

        # Build hover for background (career vs per-season differ)
        if is_career:
            bg_custom = np.stack([
                df_bg["PLAYER_NAME"],
                df_bg["TEAMS"].fillna(""),
                df_bg["SEASONS"].astype(str),
                df_bg[x_stat].round(2).astype(str),
                df_bg[y_stat].round(2).astype(str),
            ], axis=-1)
            bg_hover = (
                "<b>%{customdata[0]}</b><br>"
                "Teams: %{customdata[1]}<br>"
                "Seasons in data: %{customdata[2]}<br>"
                f"{STAT_SHORT[x_stat]}: %{{customdata[3]}}<br>"
                f"{STAT_SHORT[y_stat]}: %{{customdata[4]}}"
                "<extra></extra>"
            )
        else:
            season_col = df_bg["SEASON"] if "SEASON" in df_bg.columns else pd.Series([""] * len(df_bg))
            bg_custom = np.stack([
                df_bg["PLAYER_NAME"],
                df_bg["TEAM"].fillna(""),
                season_col,
                df_bg[x_stat].round(2).astype(str),
                df_bg[y_stat].round(2).astype(str),
            ], axis=-1)
            bg_hover = (
                "<b>%{customdata[0]}</b>  %{customdata[1]}<br>"
                "Season: %{customdata[2]}<br>"
                f"{STAT_SHORT[x_stat]}: %{{customdata[3]}}<br>"
                f"{STAT_SHORT[y_stat]}: %{{customdata[4]}}"
                "<extra></extra>"
            )

        fig.add_trace(go.Scatter(
            x=df_bg[x_stat], y=df_bg[y_stat],
            mode="markers", name="All players",
            marker=dict(color=bg_c.tolist(), size=sz(df_bg), opacity=0.35, line=dict(width=0)),
            customdata=bg_custom, hovertemplate=bg_hover,
        ))

        # Highlighted players
        for i, pname in enumerate(selected_players):
            sub = df_hi[df_hi["PLAYER_NAME"] == pname]
            if sub.empty:
                st.sidebar.caption(f"⚠ '{pname}' not found in this scope.")
                continue

            col = PLAYER_PALETTE[i % len(PLAYER_PALETTE)]

            if is_career:
                ann = sub.apply(
                    lambda r: f"<b>{r['PLAYER_NAME']}</b><br>{int(r['SEASONS'])} seasons", axis=1
                ).tolist()
                hi_custom = np.stack([
                    sub["PLAYER_NAME"],
                    sub["TEAMS"].fillna(""),
                    sub["SEASONS"].astype(str),
                    sub[x_stat].round(2).astype(str),
                    sub[y_stat].round(2).astype(str),
                ], axis=-1)
                hi_hover = (
                    "<b>%{customdata[0]}</b><br>"
                    "Teams: %{customdata[1]}<br>"
                    "Seasons: %{customdata[2]}<br>"
                    f"{STAT_SHORT[x_stat]}: %{{customdata[3]}}<br>"
                    f"{STAT_SHORT[y_stat]}: %{{customdata[4]}}"
                    "<extra></extra>"
                )
            else:
                ann = sub.apply(
                    lambda r: (
                        f"<b>{r['PLAYER_NAME']}</b><br>{r['TEAM']}"
                        + (f" ({r['SEASON'][2:]})" if scope == "All Seasons" else "")
                    ),
                    axis=1,
                ).tolist()
                season_col2 = sub["SEASON"] if "SEASON" in sub.columns else pd.Series([""] * len(sub))
                hi_custom = np.stack([
                    sub["PLAYER_NAME"],
                    sub["TEAM"].fillna(""),
                    season_col2,
                    sub[x_stat].round(2).astype(str),
                    sub[y_stat].round(2).astype(str),
                ], axis=-1)
                hi_hover = (
                    "<b>%{customdata[0]}</b>  %{customdata[1]}<br>"
                    "Season: %{customdata[2]}<br>"
                    f"{STAT_SHORT[x_stat]}: %{{customdata[3]}}<br>"
                    f"{STAT_SHORT[y_stat]}: %{{customdata[4]}}"
                    "<extra></extra>"
                )

            fig.add_trace(go.Scatter(
                x=sub[x_stat], y=sub[y_stat],
                mode="markers+text", name=pname,
                marker=dict(color=col, size=sz(sub, scale=28, base=12),
                            opacity=1.0, line=dict(width=1.5, color="#E6EDF3")),
                text=ann, textposition="top right",
                textfont=dict(color=col, size=11),
                customdata=hi_custom, hovertemplate=hi_hover,
            ))

        if show_lg:
            fig.add_hline(y=lg_y, line_dash="dash", line_color=MUTED, line_width=1.2, opacity=0.7,
                          annotation_text=f"Lg avg {STAT_SHORT[y_stat]} = {lg_y:.2f}",
                          annotation_font=dict(color=MUTED, size=10),
                          annotation_position="bottom right")
            fig.add_vline(x=lg_x, line_dash="dot", line_color=MUTED, line_width=1.0, opacity=0.7,
                          annotation_text=f"Lg avg {STAT_SHORT[x_stat]} = {lg_x:.2f}",
                          annotation_font=dict(color=MUTED, size=10),
                          annotation_position="top left")

        xr = df_sc[x_stat].max() - df_sc[x_stat].min()
        yr = df_sc[y_stat].max() - df_sc[y_stat].min()
        fig.update_xaxes(range=[df_sc[x_stat].min()-xr*.05, df_sc[x_stat].max()+xr*.18])
        fig.update_yaxes(range=[df_sc[y_stat].min()-yr*.08, df_sc[y_stat].max()+yr*.30])

        hl = (f"Annotated: {', '.join(selected_players)}"
              if selected_players else "Hover any dot · Highlight players in sidebar")
        fig.update_layout(**PLOTLY_THEME,
            title=dict(text=(f"<b>{STATS[x_stat]}</b>  ×  <b>{STATS[y_stat]}</b><br>"
                             f"<span style='font-size:12px;color:{MUTED};'>"
                             f"{scope_label}  ·  {hl}</span>"),
                       font=dict(size=15), x=0, xanchor="left"),
            xaxis_title=STATS[x_stat], yaxis_title=STATS[y_stat],
            height=640, hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)

    # ── BAR ───────────────────────────────────────────────────────────────────
    else:
        df_bar = df_plot_base.dropna(subset=[bar_stat]).copy()
        if df_bar.empty:
            st.warning(f"No data for {STATS[bar_stat]}.")
            st.stop()

        if not is_career and scope == "All Seasons":
            df_bar["DISPLAY"] = df_bar.apply(
                lambda r: f"{r['PLAYER_NAME']}  ({r.get('SEASON','')[2:]}", axis=1)
        elif is_career:
            df_bar["DISPLAY"] = df_bar["PLAYER_NAME"]
        else:
            df_bar["DISPLAY"] = df_bar["PLAYER_NAME"]

        df_bar = df_bar.sort_values(bar_stat, ascending=sort_asc)
        df_bar = df_bar.tail(top_n) if not sort_asc else df_bar.head(top_n)

        lg_val = df_plot_base[bar_stat].mean()

        def bcolor(row):
            if row["PLAYER_NAME"] in selected_players: return BLUE
            return GOLD if row[bar_stat] >= lg_val else RED

        colors = df_bar.apply(bcolor, axis=1).tolist()

        if is_career:
            bar_text = df_bar.apply(
                lambda r: f"  {int(r['SEASONS'])}s  ·  {r[bar_stat]:+.2f}", axis=1).tolist()
            hover_tmpl = (
                "<b>%{customdata[0]}</b><br>"
                "Seasons: %{customdata[1]}<br>"
                f"{STAT_SHORT[bar_stat]}: %{{customdata[2]}}"
                "<extra></extra>"
            )
            cdata = np.stack([
                df_bar["PLAYER_NAME"],
                df_bar["SEASONS"].astype(str),
                df_bar[bar_stat].round(3).astype(str),
            ], axis=-1)
        else:
            team_col = df_bar["TEAM"].fillna("") if "TEAM" in df_bar.columns else pd.Series([""] * len(df_bar))
            bar_text = df_bar.apply(
                lambda r: f"  {r.get('TEAM','')}  ·  {r[bar_stat]:+.2f}", axis=1).tolist()
            season_col3 = df_bar["SEASON"] if "SEASON" in df_bar.columns else pd.Series([""] * len(df_bar))
            hover_tmpl = (
                "<b>%{customdata[0]}</b>  %{customdata[1]}<br>"
                "Season: %{customdata[2]}<br>"
                f"{STAT_SHORT[bar_stat]}: %{{customdata[3]}}"
                "<extra></extra>"
            )
            cdata = np.stack([
                df_bar["PLAYER_NAME"],
                team_col,
                season_col3,
                df_bar[bar_stat].round(3).astype(str),
            ], axis=-1)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df_bar[bar_stat], y=df_bar["DISPLAY"],
            orientation="h", marker_color=colors, marker_line=dict(width=0),
            text=bar_text, textposition="outside",
            textfont=dict(color="#E6EDF3", size=10),
            customdata=cdata, hovertemplate=hover_tmpl,
        ))
        if show_lg:
            fig2.add_vline(x=lg_val, line_dash="dash", line_color=MUTED, line_width=1.5, opacity=0.8,
                           annotation_text=f"Lg avg = {lg_val:.2f}",
                           annotation_font=dict(color=MUTED, size=10), annotation_position="top")

        xmx = df_bar[bar_stat].max()
        xmn = min(df_bar[bar_stat].min(), 0)
        xrng = xmx - xmn
        fig2.update_xaxes(range=[xmn - xrng*.04, xmx + xrng*.28])
        fig2.update_yaxes(tickfont=dict(size=10, color="#E6EDF3"))

        hl = (f"Blue = {', '.join(selected_players)}"
              if selected_players else "Gold = above avg  ·  Red = below avg")
        fig2.update_layout(**PLOTLY_THEME,
            title=dict(text=(f"<b>{STATS[bar_stat]}</b>  ·  Top {top_n}<br>"
                             f"<span style='font-size:12px;color:{MUTED};'>"
                             f"{scope_label}  ·  {hl}</span>"),
                       font=dict(size=15), x=0, xanchor="left"),
            xaxis_title=STATS[bar_stat],
            height=max(440, top_n*34+130), bargap=0.22)
        st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# MODE: PLAYER STATS
# =============================================================================

elif mode == "Player Stats":

    with st.sidebar:
        st.markdown("---")
        all_players_list = sorted(df_full["PLAYER_NAME"].dropna().unique())
        player_sel = st.multiselect(
            "Player(s)",
            all_players_list,
            placeholder="Type a name…",
        )
        if player_sel:
            season_scope = st.radio(
                "Season scope",
                ["Career Average"] + sorted(
                    [s for s in df_full["SEASON"].unique() if s in VALID_SEASONS],
                    reverse=True,
                ),
                index=0,
            )

    st.markdown("## Player Stats")

    if not player_sel:
        st.info("Select one or more players in the sidebar.")
        st.stop()

    for pname in player_sel:
        st.markdown(f"### {pname}")

        if season_scope == "Career Average":
            row = career_df[career_df["PLAYER_NAME"] == pname]
            if row.empty:
                st.warning(f"No career data found for {pname}.")
                continue

            r = row.iloc[0]
            teams = r.get("TEAMS","—")
            seasons = int(r.get("SEASONS", 0))
            total_min = r.get("TOTAL_MIN", 0)

            st.caption(
                f"Career average across **{seasons} season(s)** in data  ·  "
                f"Teams: {teams}  ·  Total minutes: {total_min:,.0f}"
            )

            stat_data = []
            for col in [c for c in ALL_STAT_COLS if c not in ("SEASON","TEAM") and c in r.index]:
                val = r[col]
                if pd.isna(val):
                    continue
                desc = STAT_DESC.get(col, col)
                fmt  = f"{val:+.3f}" if isinstance(val, float) else str(val)
                stat_data.append({"Stat": col, "Description": desc, "Value": fmt})

            if stat_data:
                st.dataframe(pd.DataFrame(stat_data), use_container_width=True, hide_index=True)

        else:
            season_data = df_full[
                (df_full["PLAYER_NAME"] == pname) &
                (df_full["SEASON"] == season_scope)
            ]
            if season_data.empty:
                st.warning(f"No data for {pname} in {season_scope}.")
                continue

            r = season_data.iloc[0]
            avail_cols = [c for c in ALL_STAT_COLS if c in r.index and not pd.isna(r[c])]

            team = r.get("TEAM","—")
            mins = r.get("MIN", np.nan)
            gp   = r.get("GP",  np.nan)
            st.caption(
                f"**{season_scope}**  ·  Team: {team}  ·  "
                f"Minutes: {mins:,.0f}  ·  Games: {int(gp) if not np.isnan(gp) else '—'}"
            )

            stat_data = []
            for col in avail_cols:
                if col in ("SEASON","TEAM"):
                    continue
                val  = r[col]
                desc = STAT_DESC.get(col, col)
                fmt  = f"{val:+.3f}" if isinstance(val, float) else str(val)
                stat_data.append({"Stat": col, "Description": desc, "Value": fmt})

            if stat_data:
                st.dataframe(pd.DataFrame(stat_data), use_container_width=True, hide_index=True)

        st.markdown("---")


# =============================================================================
# MODE: COMPARE
# =============================================================================

elif mode == "Compare":

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Player 1**")
        all_players_list = sorted(df_full["PLAYER_NAME"].dropna().unique())

        p1 = st.selectbox("Player 1", [""] + all_players_list,
                          format_func=lambda x: "Select…" if x == "" else x)
        if p1:
            scope_options_1 = ["Career Average"] + sorted(
                df_full[df_full["PLAYER_NAME"]==p1]["SEASON"].unique().tolist(), reverse=True)
            scope1 = st.selectbox("Period (P1)", scope_options_1, key="s1")

        st.markdown("**Player 2**")
        p2 = st.selectbox("Player 2", [""] + all_players_list,
                          format_func=lambda x: "Select…" if x == "" else x)
        if p2:
            scope_options_2 = ["Career Average"] + sorted(
                df_full[df_full["PLAYER_NAME"]==p2]["SEASON"].unique().tolist(), reverse=True)
            scope2 = st.selectbox("Period (P2)", scope_options_2, key="s2")

    st.markdown("## Compare")

    if not p1 or not p2:
        st.info("Select two players in the sidebar to compare.")
        st.stop()

    def get_player_row(pname, scope):
        if scope == "Career Average":
            row = career_df[career_df["PLAYER_NAME"] == pname]
            if row.empty:
                return None, "career"
            r = row.iloc[0]
            label = f"{pname}  (Career avg · {int(r.get('SEASONS',0))}s)"
            return r, label
        else:
            row = df_full[(df_full["PLAYER_NAME"]==pname) & (df_full["SEASON"]==scope)]
            if row.empty:
                return None, scope
            r = row.iloc[0]
            label = f"{pname}  ·  {scope}"
            return r, label

    r1, label1 = get_player_row(p1, scope1 if p1 else "Career Average")
    r2, label2 = get_player_row(p2, scope2 if p2 else "Career Average")

    if r1 is None:
        st.error(f"No data for {p1} ({scope1}).")
        st.stop()
    if r2 is None:
        st.error(f"No data for {p2} ({scope2}).")
        st.stop()

    # ── Side-by-side display ───────────────────────────────────────────────────
    compare_cols = [c for c in ALL_STAT_COLS if c not in ("SEASON","TEAM")]
    avail_compare = [c for c in compare_cols
                     if c in r1.index and c in r2.index
                     and (not pd.isna(r1[c]) or not pd.isna(r2[c]))]

    col_l, col_mid, col_r = st.columns([5, 1, 5])

    def fmt_val(v):
        if pd.isna(v): return "—"
        if isinstance(v, float): return f"{v:+.3f}"
        return str(v)

    def winner_color(v1, v2, col):
        """Return (c1, c2) colors — gold for better, muted for worse."""
        if pd.isna(v1) or pd.isna(v2):
            return MUTED, MUTED
        if col == "P_VALUE":
            return (GOLD, MUTED) if v1 < v2 else (MUTED, GOLD)
        return (GOLD, MUTED) if v1 > v2 else (MUTED, GOLD)

    with col_l:
        st.markdown(
            f"<div class='compare-header'>{label1}</div>",
            unsafe_allow_html=True,
        )
        for c in avail_compare:
            v1 = r1.get(c, np.nan)
            v2 = r2.get(c, np.nan)
            c1_col, _ = winner_color(
                v1 if isinstance(v1, float) else 0,
                v2 if isinstance(v2, float) else 0,
                c,
            )
            st.markdown(
                f"<div class='stat-row'>"
                f"<span class='stat-name'>{STAT_DESC.get(c,c)}</span>"
                f"<span style='color:{c1_col};font-weight:600'>{fmt_val(v1)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_mid:
        st.markdown("<div style='padding-top:40px'></div>", unsafe_allow_html=True)
        for c in avail_compare:
            st.markdown(
                f"<div style='text-align:center;padding:6px 0;"
                f"border-bottom:1px solid #21262D;font-size:11px;"
                f"color:#8B949E;font-weight:600'>{c}</div>",
                unsafe_allow_html=True,
            )

    with col_r:
        st.markdown(
            f"<div class='compare-header'>{label2}</div>",
            unsafe_allow_html=True,
        )
        for c in avail_compare:
            v1 = r1.get(c, np.nan)
            v2 = r2.get(c, np.nan)
            _, c2_col = winner_color(
                v1 if isinstance(v1, float) else 0,
                v2 if isinstance(v2, float) else 0,
                c,
            )
            st.markdown(
                f"<div class='stat-row'>"
                f"<span style='color:{c2_col};font-weight:600'>{fmt_val(v2)}</span>"
                f"<span class='stat-name' style='text-align:right'>{STAT_DESC.get(c,c)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption(
        "**Gold** = better value for that stat  ·  "
        "Career average uses minute-weighted mean across all qualifying seasons in the dataset"
    )