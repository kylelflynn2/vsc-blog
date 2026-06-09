"""
Chicago Cubs Batting Order Analyzer - Flask Blueprint
Mounted at /projects/cubs-2026 inside the Kyle Flynn portfolio.

Lineup philosophy: Sabermetrics Not Salary.
OBP is king. Two lineups generated — one optimized vs RHP, one vs LHP —
using each player's actual FanGraphs split OBP (vs RHP / vs LHP).
Source: FanGraphs batting leaderboard, month=14 (vs RHP) and month=13 (vs LHP),
2025+2026 combined (2026 only for new Cubs additions).
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, abort
from scipy.optimize import linear_sum_assignment

# -----------------------------------------------------------------------------
# Blueprint setup
# -----------------------------------------------------------------------------
cubs_bp = Blueprint(
    'cubs',
    __name__,
    template_folder='templates',
    url_prefix='/projects/cubs-2026'
)

# Global data store
DATA          = {}
LINEUP_VS_RHP = []
LINEUP_VS_LHP = []

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

REQUIRED_CSVS = [
    'standard_batting.csv',
    'advanced_batting.csv',
    'value_batting.csv',
    'team_win_probability.csv',
]

# Batting handedness for display on cards (L = left, R = right, S = switch)
HANDEDNESS = {
    'crowape01':  'L',   # Pete Crow-Armstrong
    'happia01':   'S',   # Ian Happ (switch)
    'hoernni01':  'R',   # Nico Hoerner
    'buschmi02':  'L',   # Michael Busch
    'suzukse01':  'R',   # Seiya Suzuki
    'swansda01':  'R',   # Dansby Swanson
    'bregmal01':  'R',   # Alex Bregman
    'amayami01':  'R',   # Miguel Amaya
    'kellyca02':  'R',   # Carson Kelly
    'confomi01':  'L',   # Michael Conforto
    'ballemo01':  'L',   # Moisés Ballesteros
    'shawma01':   'R',   # Matt Shaw
    'lopezni01':  'S',   # Nicky Lopez (switch)
    'carlsdy01':  'S',   # Dylan Carlson (switch)
    'alcanke01':  'R',   # Kevin Alcantara
    'kingesc01':  'R',   # Scott Kingery
    'ramirpe01':  'R',   # Pedro Ramirez
}

# Actual FanGraphs OBP splits per player: (vs_rhp_obp, vs_lhp_obp)
# Source: FanGraphs batting leaderboard month=14 (vs RHP) and month=13 (vs LHP)
# 2025+2026 combined average where both available; 2026-only for new Cubs additions.
SPLITS_OBP = {
    'crowape01': (0.318, 0.298),  # PCA: slight edge vs RHP; reverse splits in 2026
    'happia01':  (0.373, 0.274),  # Happ (S): .099 gap — bats left vs RHP, right vs LHP
    'hoernni01': (0.327, 0.373),  # Hoerner: .046 edge vs LHP (typical righty)
    'buschmi02': (0.364, 0.319),  # Busch: .045 edge vs RHP (typical lefty)
    'suzukse01': (0.317, 0.371),  # Suzuki: .054 edge vs LHP (typical righty)
    'swansda01': (0.299, 0.285),  # Swanson: minimal splits either way
    'bregmal01': (0.317, 0.354),  # Bregman: .037 edge vs LHP (2026 only, new Cub)
    'amayami01': (0.325, 0.320),  # Amaya: nearly even splits
    'kellyca02': (0.322, 0.382),  # Kelly: .060 edge vs LHP (typical righty)
    'confomi01': (0.350, 0.500),  # Conforto: vs-LHP is small 2026 sample
    'ballemo01': (0.353, 0.375),  # Ballesteros: slight reverse splits
    'shawma01':  (0.270, 0.310),  # Shaw: estimated from limited data
}


# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

def _clean_pct_cols(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('%', '', regex=False).str.strip(),
                errors='coerce'
            ) / 100.0
    return df


def _filter_totals(df):
    df = df[df['player_id'].notna()]
    df = df[~df['player_id'].astype(str).str.startswith('-')]
    return df.reset_index(drop=True)


def _numeric_all(df, skip=('player_id', 'display_name', 'pos_primary',
                            'Player', 'Name', 'Pos', 'Pos.1',
                            'Awards', 'Pos Summary')):
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _safe_read(path, **kwargs):
    if not os.path.exists(path):
        print(f"  [Cubs] optional file not found, skipping: {os.path.basename(path)}")
        return None
    return pd.read_csv(path, **kwargs)


def load_all_data():
    df_std = pd.read_csv(os.path.join(DATA_DIR, 'standard_batting.csv'),
                         on_bad_lines='skip')
    df_std = df_std.rename(columns={'Player-additional': 'player_id'})
    df_std = _filter_totals(df_std)
    df_std['display_name'] = (
        df_std['Player']
        .str.replace(r'[*#]', '', regex=True)
        .str.replace(r'\s*\(.*?\)', '', regex=True)
        .str.strip()
    )
    df_std['pos_primary'] = (
        df_std['Pos'].astype(str)
        .str.replace(r'[*#]', '', regex=True)
        .str.split('/').str[0]
        .str.strip()
    )

    df_adv = _safe_read(os.path.join(DATA_DIR, 'advanced_batting.csv'), header=1)
    if df_adv is not None:
        df_adv = df_adv.rename(columns={df_adv.columns[-1]: 'player_id'})
        df_adv = _filter_totals(df_adv)
        if 'cWPA' in df_adv.columns:
            df_adv = _clean_pct_cols(df_adv, ['cWPA'])

    df_val = _safe_read(os.path.join(DATA_DIR, 'value_batting.csv'))
    if df_val is not None:
        df_val = df_val.rename(columns={'Player-additional': 'player_id'})
        df_val = _filter_totals(df_val)

    df_neut = _safe_read(os.path.join(DATA_DIR, 'neutralized_batting.csv'))
    if df_neut is not None:
        df_neut = df_neut.rename(columns={'Name-additional': 'player_id'})
        df_neut = _filter_totals(df_neut)

    df_run = _safe_read(os.path.join(DATA_DIR, 'team_baserunning.csv'))
    if df_run is not None:
        df_run = df_run.rename(columns={'Name-additional': 'player_id'})
        df_run = _filter_totals(df_run)
        df_run = _clean_pct_cols(df_run, ['SB%', 'RS%', 'XBT%', 'BRS%'])

    df_rat = _safe_read(os.path.join(DATA_DIR, 'team_batting_ratios.csv'))
    if df_rat is not None:
        df_rat = df_rat.rename(columns={'Name-additional': 'player_id'})
        df_rat = _filter_totals(df_rat)
        df_rat = _clean_pct_cols(df_rat, ['HR%', 'SO%', 'BB%', 'XBH%', 'X/H%', 'IP%', 'LD%', 'HR/FB', 'IF/FB'])

    df_sab = _safe_read(os.path.join(DATA_DIR, 'team_sabermetric_batting.csv'))
    if df_sab is not None:
        df_sab = df_sab.rename(columns={'Name-additional': 'player_id'})
        df_sab = _filter_totals(df_sab)

    df_wp = _safe_read(os.path.join(DATA_DIR, 'team_win_probability.csv'))
    if df_wp is not None:
        df_wp = df_wp.rename(columns={'Name-additional': 'player_id'})
        df_wp = _filter_totals(df_wp)
        df_wp = _clean_pct_cols(df_wp, ['cWPA', 'cWPA+', 'cWPA-'])

    df_situ = _safe_read(os.path.join(DATA_DIR, 'team_ph_hr_situ_hitting.csv'), header=1)
    if df_situ is not None:
        df_situ = df_situ.rename(columns={df_situ.columns[-1]: 'player_id'})
        df_situ = _filter_totals(df_situ)

    df = df_std.copy()

    def _safe_merge(df, df_sup, cols):
        if df_sup is None:
            return df
        keep = [c for c in cols if c in df_sup.columns]
        if len(keep) <= 1:
            return df
        return df.merge(df_sup[keep], on='player_id', how='left')

    adv_cols  = ['player_id', 'BAbip', 'ISO', 'SO%', 'EV', 'HardH%',
                 'LD%', 'GB%', 'FB%', 'GB/FB', 'Pull%', 'Cent%', 'Oppo%',
                 'RS%', 'SB%', 'XBT%']
    df = _safe_merge(df, df_adv, adv_cols)

    df = _safe_merge(df, df_val,
                     ['player_id', 'Rbat', 'Rbaser', 'Rfield', 'oWAR', 'dWAR'])

    if df_neut is not None and 'RC' in df_neut.columns:
        df_neut_sub = df_neut[['player_id', 'RC']].rename(columns={'RC': 'RC_neut'})
        df = df.merge(df_neut_sub, on='player_id', how='left')

    if df_run is not None:
        run_cols = ['player_id', 'SBO', 'SB%', 'RS%', 'XBT%',
                    'SB2', 'CS2', 'SB3', 'CS3', 'OOB']
        run_keep = [c for c in run_cols if c in df_run.columns]
        df = df.merge(df_run[run_keep], on='player_id', how='left',
                      suffixes=('', '_run'))
        df = df[[c for c in df.columns if not c.endswith('_run')]]

    df = _safe_merge(df, df_rat,
                     ['player_id', 'SO/W', 'AB/SO', 'AB/HR', 'AB/RBI',
                      'GO/AO', 'HR/FB', 'IF/FB'])

    df = _safe_merge(df, df_sab,
                     ['player_id', 'RC', 'RC/G', 'AIR', 'lgBA', 'lgOBP',
                      'lgSLG', 'lgOPS', 'OWn%', 'BtRuns', 'BtWins',
                      'TotA', 'SecA', 'PwrSpd'])

    df = _safe_merge(df, df_wp,
                     ['player_id', 'WPA', 'WPA+', 'WPA-', 'cWPA',
                      'Clutch', 'cClutch', 'aLI', 'RE24', 'REW'])

    df = _safe_merge(df, df_situ,
                     ['player_id', 'vRH', 'vLH', 'Hm', 'Rd'])

    df = _numeric_all(df)

    DATA['master'] = df
    DATA['df_wp']  = df_wp if df_wp is not None else pd.DataFrame()

    print(f"Cubs data loaded: {len(df)} players, {df.shape[1]} columns")


# =============================================================================
# SECTION 2: BATTING ORDER ALGORITHM
# OBP-first sabermetric framework:
#   1. Leadoff  — highest OBP + speed
#   2. 2-hole   — best overall hitter (OBP + wOBA/rOBA + OPS+)
#   3. 3-hole   — balanced OBP & power (OPS+, SLG, WAR)
#   4. Cleanup  — maximum power (SLG, HR)
#   5. 5-hole   — secondary power (RBI, SLG)
#   6-7         — table setters (OBP, OPS)
#   8-9         — weakest by OPS
# Platoon: each player's actual FanGraphs split OBP (vs RHP or vs LHP) is used
#          directly — no blanket +.022 estimate needed.
# =============================================================================

def _safe_float(val, default=0.0):
    try:
        f = float(val)
        return f if not np.isnan(f) else default
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0):
    try:
        f = float(val)
        return int(f) if not np.isnan(f) else default
    except (TypeError, ValueError):
        return default


def _minmax_normalize(series):
    s = pd.to_numeric(series, errors='coerce').fillna(0.0)
    mn, mx = s.min(), s.max()
    if mx > mn:
        return (s - mn) / (mx - mn)
    return pd.Series([0.5] * len(s), index=s.index)


def _spot_score(row, spot):
    """
    Score how well a player fits a batting spot.
    OBP is the primary driver at the top of the lineup.
    """
    obp      = row.get('OBP', 0)
    slg      = row.get('SLG', 0)
    ops      = row.get('OPS', 0)
    ops_plus = row.get('OPS+', 0)
    ba       = row.get('BA', 0)
    hr       = row.get('HR', 0)
    rbi      = row.get('RBI', 0)
    iso      = row.get('ISO', 0)
    war      = row.get('WAR', 0)
    roba     = row.get('rOBA', 0)
    rcg      = row.get('RC/G', 0)
    seca     = row.get('SecA', 0)
    sb_pct   = row.get('SB%', 0)
    so_inv   = row.get('SO%_inv', 0)

    if spot == 1:
        # Leadoff: OBP is king, then speed and contact
        return obp * 0.40 + sb_pct * 0.25 + so_inv * 0.20 + seca * 0.15
    elif spot == 2:
        # Best overall: OBP + rOBA + OPS+ + WAR
        return obp * 0.35 + roba * 0.25 + ops_plus * 0.20 + war * 0.20
    elif spot == 3:
        # RBI/contact: balanced OBP, SLG, and overall value
        return obp * 0.25 + ops_plus * 0.25 + slg * 0.25 + war * 0.25
    elif spot == 4:
        # Cleanup: maximum power
        return slg * 0.30 + hr * 0.30 + iso * 0.20 + rbi * 0.20
    elif spot == 5:
        # Secondary power / RBI protection
        return slg * 0.25 + hr * 0.20 + rbi * 0.30 + rcg * 0.25
    elif spot == 6:
        # Top of the bottom third: OBP table setter
        return obp * 0.40 + ops * 0.30 + ba * 0.30
    return 0.0


def _has_platoon_advantage(player_id, vs_rhp):
    """True if the player's actual split OBP is meaningfully higher in this matchup."""
    pid = str(player_id)
    splits = SPLITS_OBP.get(pid)
    if splits is None:
        hand = HANDEDNESS.get(pid, 'R')
        return (hand in ('L', 'S')) if vs_rhp else (hand in ('R', 'S'))
    rhp_obp, lhp_obp = splits
    if vs_rhp:
        return (rhp_obp - lhp_obp) >= 0.015
    else:
        return (lhp_obp - rhp_obp) >= 0.015


def _build_rationale(player, spot, vs_rhp, split_obp=None):
    name      = player['display_name']
    first     = name.split()[0]
    obp       = split_obp if split_obp is not None else _safe_float(player.get('OBP'))
    slg       = _safe_float(player.get('SLG'))
    ops       = _safe_float(player.get('OPS'))
    hr        = _safe_int(player.get('HR'))
    rbi       = _safe_int(player.get('RBI'))
    sb        = _safe_int(player.get('SB'))
    war       = _safe_float(player.get('WAR'))
    ops_p     = _safe_int(player.get('OPS+'))

    pid       = str(player.get('player_id', ''))
    adv       = _has_platoon_advantage(pid, vs_rhp)
    splits    = SPLITS_OBP.get(pid)
    arm       = 'right-handed' if vs_rhp else 'left-handed'
    fmt       = lambda v: f".{int(round(v * 1000)):03d}"

    # Natural platoon note — only when the split gap is meaningful
    platoon_note = ''
    if adv and splits:
        gap = abs(splits[0] - splits[1])
        platoon_note = f' That {fmt(gap)} OBP edge against {arm} pitching is exactly why {first} is here.'

    rationales = {
        1: (f"{name} leads off with a {fmt(obp)} OBP against {arm} pitchers, "
            f"combining elite on-base ability with {sb} stolen bases to set the table.{platoon_note}"),
        2: (f"{name} bats second — the most important spot in any lineup. "
            f"A {fmt(obp)} OBP against {arm} pitchers, {ops_p} OPS+, and {war:.1f} WAR "
            f"make {first} the best all-around hitter on this roster.{platoon_note}"),
        3: (f"{name} hits third with {ops_p} OPS+, {war:.1f} WAR, and {fmt(slg)} SLG — "
            f"balanced OBP and power with the table already set.{platoon_note}"),
        4: (f"{name} bats cleanup with {hr} HR, {rbi} RBI, and {fmt(slg)} SLG — "
            f"the primary run-producer and the most dangerous bat in the order.{platoon_note}"),
        5: (f"{name} hits fifth with {rbi} RBI and {fmt(slg)} SLG, "
            f"extending the heart of the order and protecting the cleanup hitter.{platoon_note}"),
        6: (f"{name} bats sixth with a {fmt(obp)} OBP against {arm} pitchers — "
            f"the best table-setter at the top of the bottom third.{platoon_note}"),
        7: (f"{name} bats seventh ({fmt(ops)} OPS), "
            f"keeping pressure on opposing pitchers deep into games.{platoon_note}"),
        8: (f"{name} hits eighth ({fmt(ops)} OPS) — "
            f"a reliable bat that helps turn the lineup over late.{platoon_note}"),
        9: (f"{name} bats ninth ({fmt(ops)} OPS), "
            f"setting the table for the top of the order to come back around.{platoon_note}"),
    }
    return rationales[spot]


def compute_lineup(df_master, vs_rhp=True):
    """
    Build an OBP-optimized batting order using actual FanGraphs split OBPs.

    Each player's vs-RHP or vs-LHP OBP from SPLITS_OBP replaces their overall
    OBP before normalization. No blanket platoon bonus — the real numbers do the work.
    """
    df = df_master.copy()
    df['PA'] = pd.to_numeric(df['PA'], errors='coerce').fillna(0)
    starters = df.nlargest(9, 'PA').reset_index(drop=True)

    # Replace each player's OBP with their actual split OBP before normalization
    for idx in starters.index:
        pid = str(starters.at[idx, 'player_id'])
        splits = SPLITS_OBP.get(pid)
        if splits is not None:
            starters.at[idx, 'OBP'] = splits[0] if vs_rhp else splits[1]

    score_cols = ['OBP', 'SLG', 'OPS', 'OPS+', 'BA', 'HR', 'RBI',
                  'ISO', 'WAR', 'rOBA', 'RC/G', 'SecA', 'SB%', 'SO%']

    norm = pd.DataFrame(index=starters.index)
    for col in score_cols:
        if col in starters.columns:
            norm[col] = _minmax_normalize(starters[col])
        else:
            norm[col] = 0.0

    norm['SO%_inv'] = 1.0 - norm.get('SO%', pd.Series([0.5]*len(starters), index=starters.index))

    n_players = len(starters)
    n_spots   = 6
    score_matrix = np.zeros((n_players, n_spots))
    for i in range(n_players):
        row = norm.iloc[i].to_dict()
        for j in range(n_spots):
            score_matrix[i][j] = _spot_score(row, j + 1)

    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    spot_to_idx = {}
    for player_row, spot_col in zip(row_ind, col_ind):
        spot_to_idx[spot_col + 1] = player_row

    assigned = set(row_ind)
    remaining_idx = [i for i in range(n_players) if i not in assigned]
    remaining = starters.iloc[remaining_idx].copy()
    remaining['OPS'] = pd.to_numeric(remaining['OPS'], errors='coerce').fillna(0)
    remaining = remaining.sort_values(['OPS', 'player_id'], ascending=[False, True])
    for rank, (_, _row) in enumerate(remaining.iterrows()):
        spot_to_idx[7 + rank] = starters.index.get_loc(_row.name)

    starters_raw = df_master.nlargest(9, 'PA').reset_index(drop=True)

    lineup = []
    for spot in range(1, 10):
        idx     = spot_to_idx[spot]
        p       = starters_raw.iloc[idx]
        pid     = str(p['player_id'])
        splits  = SPLITS_OBP.get(pid)
        obp_rhp = splits[0] if splits else _safe_float(p.get('OBP'))
        obp_lhp = splits[1] if splits else _safe_float(p.get('OBP'))
        split_obp = obp_rhp if vs_rhp else obp_lhp
        lineup.append({
            'spot':        spot,
            'player_id':   pid,
            'name':        str(p['display_name']),
            'pos':         str(p.get('pos_primary', '')),
            'age':         _safe_int(p.get('Age')),
            'PA':          _safe_int(p.get('PA')),
            'BA':          _safe_float(p.get('BA')),
            'OBP':         split_obp,
            'obp_vs_rhp':  obp_rhp,
            'obp_vs_lhp':  obp_lhp,
            'SLG':         _safe_float(p.get('SLG')),
            'OPS':         _safe_float(p.get('OPS')),
            'OPS+':        _safe_int(p.get('OPS+')),
            'HR':          _safe_int(p.get('HR')),
            'RBI':         _safe_int(p.get('RBI')),
            'SB':          _safe_int(p.get('SB')),
            'WAR':         _safe_float(p.get('WAR')),
            'RC/G':        _safe_float(p.get('RC/G')),
            'SecA':        _safe_float(p.get('SecA')),
            'SB%':         _safe_float(p.get('SB%')),
            'WPA':         _safe_float(p.get('WPA')),
            'Clutch':      _safe_float(p.get('Clutch')),
            'hand':        HANDEDNESS.get(pid, 'R'),
            'platoon_adv': _has_platoon_advantage(pid, vs_rhp),
            'rationale':   _build_rationale(p, spot, vs_rhp, split_obp),
        })

    return lineup


# =============================================================================
# SECTION 3: FLASK ROUTES
# =============================================================================

def _fmt(val, decimals=3):
    try:
        f = float(val)
        return '—' if np.isnan(f) else f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return '—'


def _lineup_ids(lineup):
    return {e['player_id'] for e in lineup}


def _quick_stats(starters_df):
    total_hr  = int(pd.to_numeric(starters_df['HR'],  errors='coerce').fillna(0).sum())
    total_rbi = int(pd.to_numeric(starters_df['RBI'], errors='coerce').fillna(0).sum())
    avg_obp   = pd.to_numeric(starters_df['OBP'], errors='coerce').mean()
    avg_ops   = pd.to_numeric(starters_df['OPS'], errors='coerce').mean()
    total_war = pd.to_numeric(starters_df['WAR'], errors='coerce').sum()
    return {
        'total_hr':  total_hr,
        'total_rbi': total_rbi,
        'avg_obp':   _fmt(avg_obp),
        'avg_ops':   _fmt(avg_ops),
        'total_war': _fmt(total_war, 1),
    }


def _build_chart_data(lineup, df):
    lineup_names = [e['name'] for e in lineup]
    qualified    = df[pd.to_numeric(df['PA'], errors='coerce') >= 30].copy()

    scatter_obp_slg = [
        {
            'x': round(_safe_float(r.get('OBP')), 3),
            'y': round(_safe_float(r.get('SLG')), 3),
            'label': str(r['display_name']),
            'player_id': str(r['player_id']),
            'in_lineup': str(r['display_name']) in lineup_names,
        }
        for _, r in qualified.iterrows()
    ]

    ops_df = qualified.copy()
    ops_df['OPS+'] = pd.to_numeric(ops_df['OPS+'], errors='coerce').fillna(0)
    ops_df = ops_df[ops_df['OPS+'] > 0].sort_values('OPS+', ascending=False)
    ops_plus_data = {
        'labels': ops_df['display_name'].tolist(),
        'values': ops_df['OPS+'].tolist(),
    }

    clutch_df = qualified.copy()
    clutch_df['Clutch'] = pd.to_numeric(clutch_df['Clutch'], errors='coerce').fillna(0)
    clutch_df = clutch_df.sort_values('Clutch', ascending=False)
    clutch_data = {
        'labels': clutch_df['display_name'].tolist(),
        'values': [round(float(v), 2) for v in clutch_df['Clutch'].tolist()],
    }

    rcg_wpa_pts = [
        {
            'x': round(_safe_float(r.get('RC/G')), 2),
            'y': round(_safe_float(r.get('WPA')), 2),
            'label': str(r['display_name']),
            'in_lineup': str(r['display_name']) in lineup_names,
        }
        for _, r in qualified.iterrows()
    ]

    return {
        'lineup_names':    lineup_names,
        'scatter_obp_slg': scatter_obp_slg,
        'ops_plus':        ops_plus_data,
        'clutch':          clutch_data,
        'scatter_rcg_wpa': rcg_wpa_pts,
    }


@cubs_bp.route('/')
def index():
    if not DATA:
        return render_template('cubs/no_data.html', required=REQUIRED_CSVS, data_dir=DATA_DIR)

    df = DATA['master']

    rhp_ids = _lineup_ids(LINEUP_VS_RHP)
    lhp_ids = _lineup_ids(LINEUP_VS_LHP)
    rhp_df  = df[df['player_id'].isin(rhp_ids)]

    qs_rhp = _quick_stats(rhp_df)
    chart_data = _build_chart_data(LINEUP_VS_RHP, df)

    return render_template(
        'cubs/index.html',
        lineup_rhp=LINEUP_VS_RHP,
        lineup_lhp=LINEUP_VS_LHP,
        quick_stats=qs_rhp,
        chart_data=json.dumps(chart_data),
    )


@cubs_bp.route('/players')
def players():
    if not DATA:
        return render_template('cubs/no_data.html', required=REQUIRED_CSVS, data_dir=DATA_DIR)

    df = DATA['master']
    lineup_ids = list(_lineup_ids(LINEUP_VS_RHP) | _lineup_ids(LINEUP_VS_LHP))

    display_cols = [
        'display_name', 'pos_primary', 'Age', 'PA', 'AB', 'R', 'H',
        '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'SO',
        'BA', 'OBP', 'SLG', 'OPS', 'OPS+', 'WAR', 'player_id'
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    rows = []
    for _, r in df[display_cols].iterrows():
        row = {}
        for col in display_cols:
            val = r[col]
            if col in ('display_name', 'pos_primary', 'player_id'):
                row[col] = str(val) if pd.notna(val) else '—'
            elif col in ('BA', 'OBP', 'SLG', 'OPS', 'WAR'):
                row[col] = _fmt(val)
            else:
                try:
                    row[col] = int(float(val)) if pd.notna(val) else '—'
                except (ValueError, TypeError):
                    row[col] = '—'
        rows.append(row)

    return render_template('cubs/players.html', players=rows, lineup_ids=lineup_ids)


@cubs_bp.route('/player/<player_id>')
def player_detail(player_id):
    if not DATA:
        return render_template('cubs/no_data.html', required=REQUIRED_CSVS, data_dir=DATA_DIR)

    df   = DATA['master']
    mask = df['player_id'] == player_id
    if not mask.any():
        abort(404)
    p    = df[mask].iloc[0]
    qual = df[pd.to_numeric(df['PA'], errors='coerce') >= 30].copy()

    radar_stats  = ['OBP', 'SLG', 'ISO', 'WAR', 'OPS+', 'RC/G']
    radar_labels = ['OBP', 'SLG', 'ISO', 'WAR', 'OPS+', 'RC/G']
    radar_values = []
    for stat in radar_stats:
        col_data = pd.to_numeric(qual[stat], errors='coerce').fillna(0)
        val = _safe_float(p.get(stat))
        mn, mx = col_data.min(), col_data.max()
        norm_val = ((val - mn) / (mx - mn) * 100) if mx > mn else 50.0
        radar_values.append(round(max(0.0, min(100.0, norm_val)), 1))

    bar_stats  = ['BA', 'OBP', 'SLG', 'OPS']
    bar_player = [_safe_float(p.get(s)) for s in bar_stats]
    bar_avg    = [
        round(float(pd.to_numeric(qual[s], errors='coerce').mean()), 3)
        if s in qual.columns else 0.0
        for s in bar_stats
    ]

    adv_fields = ['BAbip', 'ISO', 'EV', 'HardH%', 'LD%', 'GB%', 'FB%', 'Pull%',
                  'WPA', 'RE24', 'Clutch', 'aLI']
    adv_table = []
    for f in adv_fields:
        if f in ('WPA', 'RE24', 'Clutch', 'aLI'):
            adv_table.append({'stat': f, 'value': _fmt(p.get(f), 2)})
        elif f == 'EV':
            adv_table.append({'stat': f, 'value': _fmt(p.get(f), 1)})
        else:
            adv_table.append({'stat': f, 'value': _fmt(p.get(f), 3)})

    run_fields = [('SB%', 'SB%'), ('RS%', 'RS%'), ('XBT%', 'XBT%'),
                  ('SecA', 'SecA'), ('SBO', 'SB Opps')]
    run_table = [{'stat': lbl, 'value': _fmt(p.get(col), 3)} for col, lbl in run_fields]

    situ_table = [
        {'stat': 'vs RHP',  'value': _fmt(p.get('vRH'), 3)},
        {'stat': 'vs LHP',  'value': _fmt(p.get('vLH'), 3)},
        {'stat': 'Home HR', 'value': str(_safe_int(p.get('Hm')))},
        {'stat': 'Road HR', 'value': str(_safe_int(p.get('Rd')))},
    ]

    chart_data = {
        'player_name':  str(p['display_name']),
        'radar_labels': radar_labels,
        'radar_values': radar_values,
        'bar_labels':   bar_stats,
        'bar_player':   bar_player,
        'bar_avg':      bar_avg,
    }

    lineup_spot_rhp = next(
        (e['spot'] for e in LINEUP_VS_RHP if e['player_id'] == player_id), None)
    lineup_spot_lhp = next(
        (e['spot'] for e in LINEUP_VS_LHP if e['player_id'] == player_id), None)

    player_info = {
        'player_id':    player_id,
        'name':         str(p.get('display_name', '—')),
        'pos':          str(p.get('pos_primary', '—')),
        'age':          _safe_int(p.get('Age')),
        'PA':           _safe_int(p.get('PA')),
        'WAR':          _fmt(p.get('WAR'), 1),
        'oWAR':         _fmt(p.get('oWAR'), 1),
        'dWAR':         _fmt(p.get('dWAR'), 1),
        'OPS+':         _safe_int(p.get('OPS+')),
        'BA':           _fmt(p.get('BA')),
        'OBP':          _fmt(p.get('OBP')),
        'SLG':          _fmt(p.get('SLG')),
        'OPS':          _fmt(p.get('OPS')),
        'HR':           _safe_int(p.get('HR')),
        'RBI':          _safe_int(p.get('RBI')),
        'SB':           _safe_int(p.get('SB')),
        'hand':         HANDEDNESS.get(player_id, 'R'),
    }

    return render_template(
        'cubs/player_detail.html',
        player=player_info,
        lineup_spot=lineup_spot_rhp,
        lineup_spot_rhp=lineup_spot_rhp,
        lineup_spot_lhp=lineup_spot_lhp,
        chart_data=json.dumps(chart_data),
        adv_table=adv_table,
        run_table=run_table,
        situ_table=situ_table,
    )


@cubs_bp.route('/analysis')
def analysis():
    if not DATA:
        return render_template('cubs/no_data.html', required=REQUIRED_CSVS, data_dir=DATA_DIR)

    df        = DATA['master']
    qualified = df[pd.to_numeric(df['PA'], errors='coerce') >= 30].copy()
    chart_data = _build_chart_data(LINEUP_VS_RHP, df)

    return render_template('cubs/analysis.html', chart_data=json.dumps(chart_data))


@cubs_bp.route('/methodology')
def methodology():
    if not DATA:
        return render_template('cubs/no_data.html', required=REQUIRED_CSVS, data_dir=DATA_DIR)
    return render_template('cubs/methodology.html', lineup=LINEUP_VS_RHP)


# =============================================================================
# LOAD DATA AT IMPORT TIME
# =============================================================================
try:
    load_all_data()
    _rhp = compute_lineup(DATA['master'], vs_rhp=True)
    _lhp = compute_lineup(DATA['master'], vs_rhp=False)
    LINEUP_VS_RHP.extend(_rhp)
    LINEUP_VS_LHP.extend(_lhp)
    print(f"Cubs vs-RHP lineup: {[e['name'] for e in LINEUP_VS_RHP]}")
    print(f"Cubs vs-LHP lineup: {[e['name'] for e in LINEUP_VS_LHP]}")
except FileNotFoundError as _e:
    print(f"Cubs CSV not found — add files to cubs/data/: {_e}")
except Exception as _e:
    print(f"Cubs data error: {_e}")
