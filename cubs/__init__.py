"""
Chicago Cubs Batting Order Analyzer - Flask Blueprint
Mounted at /projects/cubs-2026 inside the Kyle Flynn portfolio.

All logic is adapted from the standalone app.py in the baseball-stats repo.
The only Flask-specific changes are:
  - Blueprint instead of a Flask app object
  - Template names prefixed with 'cubs/'
  - url_for() uses 'cubs.' endpoint names
  - Data is loaded once at import time with a graceful fallback
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
# 'cubs'            = the name used in url_for(), e.g. url_for('cubs.index')
# template_folder   = look for templates in cubs/templates/
# url_prefix        = every route in this blueprint starts with /projects/cubs-2026
cubs_bp = Blueprint(
    'cubs',
    __name__,
    template_folder='templates',
    url_prefix='/projects/cubs-2026'
)

# Global data store – filled by load_all_data()
DATA   = {}   # dict of DataFrames keyed by name
LINEUP = []   # list of 9 dicts, one per batting spot

# Where the CSV files live (cubs/data/ folder)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Required CSV filenames (for the "data missing" page)
REQUIRED_CSVS = [
    'standard_batting.csv',
    'advanced_batting.csv',
    'value_batting.csv',
    'neutralized_batting.csv',
    'team_baserunning.csv',
    'team_batting_ratios.csv',
    'team_sabermetric_batting.csv',
    'team_win_probability.csv',
    'team_ph_hr_situ_hitting.csv',
]


# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

def _clean_pct_cols(df, cols):
    """
    Strip trailing '%' from percentage string columns and convert to floats
    in decimal form (e.g. "38%" becomes 0.38).
    Only processes columns that actually exist in the DataFrame.
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('%', '', regex=False).str.strip(),
                errors='coerce'
            ) / 100.0
    return df


def _filter_totals(df):
    """Remove Team Totals and League Average rows (player_id starts with '-')."""
    df = df[df['player_id'].notna()]
    df = df[~df['player_id'].astype(str).str.startswith('-')]
    return df.reset_index(drop=True)


def _numeric_all(df, skip=('player_id', 'display_name', 'pos_primary',
                            'Player', 'Name', 'Pos', 'Pos.1',
                            'Awards', 'Pos Summary')):
    """Convert every non-string column to numeric where possible."""
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _safe_read(path, **kwargs):
    """
    Try to read a CSV file.
    Returns None if the file doesn't exist — callers skip that merge.
    """
    if not os.path.exists(path):
        print(f"  [Cubs] optional file not found, skipping: {os.path.basename(path)}")
        return None
    return pd.read_csv(path, **kwargs)


def load_all_data():
    """
    Load CSV files from cubs/data/, clean them up, and join everything
    into one master DataFrame stored in DATA['master'].
    Only standard_batting.csv is required; all other files are optional.
    """

    # 1. standard_batting.csv – primary source for slash stats and WAR
    # on_bad_lines='skip' silently drops pitcher rows that have an extra
    # field in Baseball Reference's CSV export (they have PA=0 and are
    # irrelevant to the lineup anyway).
    df_std = pd.read_csv(os.path.join(DATA_DIR, 'standard_batting.csv'),
                         on_bad_lines='skip')
    df_std = df_std.rename(columns={'Player-additional': 'player_id'})
    df_std = _filter_totals(df_std)
    # Clean display name: strip handedness markers (*/#) and IL/DFA notes
    df_std['display_name'] = (
        df_std['Player']
        .str.replace(r'[*#]', '', regex=True)
        .str.replace(r'\s*\(.*?\)', '', regex=True)
        .str.strip()
    )
    # Use only the first position code (e.g. '1B' from '1B/DH')
    df_std['pos_primary'] = (
        df_std['Pos'].astype(str)
        .str.replace(r'[*#]', '', regex=True)
        .str.split('/').str[0]
        .str.strip()
    )

    # 2-9. Supplemental files (all optional — missing files are skipped gracefully)

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

    # ------------------------------------------------------------------
    # Join everything onto the standard_batting base.
    # Only merge a supplemental source if it loaded successfully.
    # ------------------------------------------------------------------
    df = df_std.copy()

    def _safe_merge(df, df_sup, cols):
        """Merge df_sup into df only if df_sup is not None."""
        if df_sup is None:
            return df
        keep = [c for c in cols if c in df_sup.columns]
        if len(keep) <= 1:   # only player_id, nothing useful
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
    DATA['df_std']  = df_std
    DATA['df_wp']   = df_wp if df_wp is not None else pd.DataFrame()

    print(f"Cubs data loaded: {len(df)} players, {df.shape[1]} columns")


# =============================================================================
# SECTION 2: BATTING ORDER ALGORITHM
# Uses a weighted score per batting spot + the Hungarian algorithm (scipy)
# to find the globally optimal assignment for spots 1-6, then fills 7-9
# with the remaining players sorted by OPS.
# =============================================================================

def _safe_float(val, default=0.0):
    """Convert a value to float, returning default if conversion fails."""
    try:
        f = float(val)
        return f if not np.isnan(f) else default
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0):
    """Convert a value to int, returning default if conversion fails."""
    try:
        f = float(val)
        return int(f) if not np.isnan(f) else default
    except (TypeError, ValueError):
        return default


def _minmax_normalize(series):
    """
    Scale a pandas Series to the range [0, 1].
    If all values are identical, returns 0.5 for every element.
    """
    s = pd.to_numeric(series, errors='coerce').fillna(0.0)
    mn, mx = s.min(), s.max()
    if mx > mn:
        return (s - mn) / (mx - mn)
    return pd.Series([0.5] * len(s), index=s.index)


def _spot_score(row, spot):
    """
    Score how well a player (row of normalized stats) fits a batting spot (1-6).
    Returns a float where higher = better fit.
    All input values are already normalized to [0, 1].
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
    so_inv   = row.get('SO%_inv', 0)   # 1 - normalized SO% (lower K rate = better)

    if spot == 1:   # Leadoff: needs OBP, speed, and contact
        return obp * 0.35 + sb_pct * 0.25 + so_inv * 0.25 + seca * 0.15
    elif spot == 2: # 2-hole: OBP + contact + secondary skills
        return obp * 0.30 + ba * 0.20 + seca * 0.25 + sb_pct * 0.25
    elif spot == 3: # Best overall hitter
        return ops_plus * 0.25 + roba * 0.25 + rcg * 0.25 + war * 0.25
    elif spot == 4: # Cleanup: power first, RBI second
        return slg * 0.30 + hr * 0.30 + iso * 0.20 + rbi * 0.20
    elif spot == 5: # Secondary power
        return slg * 0.25 + hr * 0.20 + rbi * 0.30 + rcg * 0.25
    elif spot == 6: # Best of the bottom third
        return ops * 0.40 + rbi * 0.30 + ba * 0.30
    return 0.0


def _build_rationale(player, spot):
    """
    Return a one-sentence explanation of why this player was placed
    in the given batting spot, using their actual (un-normalized) stats.
    """
    name   = player['display_name']
    obp    = _safe_float(player.get('OBP'))
    slg    = _safe_float(player.get('SLG'))
    ops    = _safe_float(player.get('OPS'))
    ba     = _safe_float(player.get('BA'))
    hr     = _safe_int(player.get('HR'))
    rbi    = _safe_int(player.get('RBI'))
    sb     = _safe_int(player.get('SB'))
    war    = _safe_float(player.get('WAR'))
    ops_p  = _safe_int(player.get('OPS+'))
    rcg    = _safe_float(player.get('RC/G'))
    seca   = _safe_float(player.get('SecA'))

    # Helper: format 0.347 as ".347"
    fmt = lambda v: f".{int(round(v * 1000)):03d}"

    rationales = {
        1: (f"{name} leads off with a {fmt(obp)} OBP and {fmt(seca)} secondary average, "
            f"combining elite on-base ability with {sb} stolen bases to set the table."),
        2: (f"{name} bats second with a {fmt(obp)} OBP and {fmt(ba)} average, "
            f"providing contact skills and secondary value ({fmt(seca)} SecA) to move runners."),
        3: (f"{name} hits third with an OPS+ of {ops_p}, {war:.1f} WAR, and {rcg:.1f} RC/G — "
            f"the most complete offensive threat in the lineup."),
        4: (f"{name} bats cleanup with {hr} home runs, {rbi} RBI, and {fmt(slg)} slugging — "
            f"the primary power source who drives in the top of the order."),
        5: (f"{name} hits fifth with {rbi} RBI and {fmt(slg)} SLG, "
            f"extending the heart of the order and protecting the cleanup hitter."),
        6: (f"{name} bats sixth with a {fmt(ops)} OPS and {rbi} RBI, "
            f"providing consistent run production at the top of the bottom third."),
        7: (f"{name} bats seventh ({fmt(ops)} OPS) — the strongest of the three bottom-order spots, "
            f"keeping pressure on opposing pitchers deep into games."),
        8: (f"{name} hits eighth ({fmt(ops)} OPS), a reliable contact bat that "
            f"helps turn the lineup over to the top of the order."),
        9: (f"{name} bats ninth ({fmt(ops)} OPS), setting the table for the leadoff hitter "
            f"and providing {rbi} RBI despite the lower-lineup positioning."),
    }
    return rationales[spot]


def compute_lineup(df_master):
    """
    Find the optimal 9-man batting order.

    Steps:
      1. Pick the top 9 players by Plate Appearances (the everyday starters).
      2. Normalize all scoring stats to [0, 1] within those 9 players.
      3. Build a 9x6 score matrix: score[player][spot] = fitness for spots 1-6.
      4. Run the Hungarian algorithm to find the best assignment for spots 1-6.
      5. Fill spots 7-9 with the remaining 3 players sorted by OPS descending.
      6. Attach a plain-English rationale to each slot.

    Returns a list of 9 dicts ordered by batting spot.
    """
    df = df_master.copy()
    df['PA'] = pd.to_numeric(df['PA'], errors='coerce').fillna(0)
    starters = df.nlargest(9, 'PA').reset_index(drop=True)

    score_cols = ['OBP', 'SLG', 'OPS', 'OPS+', 'BA', 'HR', 'RBI',
                  'ISO', 'WAR', 'rOBA', 'RC/G', 'SecA', 'SB%', 'SO%']

    # Build a normalized copy of the starters' stats
    norm = pd.DataFrame(index=starters.index)
    for col in score_cols:
        if col in starters.columns:
            norm[col] = _minmax_normalize(starters[col])
        else:
            norm[col] = 0.0  # stat missing – contributes nothing to scores

    # Lower strikeout rate is better, especially for the leadoff spot
    norm['SO%_inv'] = 1.0 - norm['SO%']

    # Build the 9x6 score matrix for the Hungarian algorithm
    n_players = len(starters)   # always 9
    n_spots   = 6               # spots 1-6 are assigned optimally; 7-9 are filled by OPS
    score_matrix = np.zeros((n_players, n_spots))
    for i in range(n_players):
        row = norm.iloc[i].to_dict()
        for j in range(n_spots):
            score_matrix[i][j] = _spot_score(row, j + 1)

    # linear_sum_assignment minimizes cost, so we negate scores to maximize
    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    # Map batting spot (1-indexed) to the starters DataFrame row index
    spot_to_idx = {}
    for player_row, spot_col in zip(row_ind, col_ind):
        spot_to_idx[spot_col + 1] = player_row   # spots 1-6

    # Fill spots 7-9 with the unassigned players, best OPS first
    assigned = set(row_ind)
    remaining_idx = [i for i in range(n_players) if i not in assigned]
    remaining = starters.iloc[remaining_idx].copy()
    remaining['OPS'] = pd.to_numeric(remaining['OPS'], errors='coerce').fillna(0)
    remaining = remaining.sort_values(['OPS', 'player_id'], ascending=[False, True])
    for rank, (_, _row) in enumerate(remaining.iterrows()):
        spot_to_idx[7 + rank] = starters.index.get_loc(_row.name)

    # Assemble the final lineup list
    lineup = []
    for spot in range(1, 10):
        idx = spot_to_idx[spot]
        p   = starters.iloc[idx]
        lineup.append({
            'spot':       spot,
            'player_id':  str(p['player_id']),
            'name':       str(p['display_name']),
            'pos':        str(p.get('pos_primary', '—')),
            'age':        _safe_int(p.get('Age')),
            'PA':         _safe_int(p.get('PA')),
            'BA':         _safe_float(p.get('BA')),
            'OBP':        _safe_float(p.get('OBP')),
            'SLG':        _safe_float(p.get('SLG')),
            'OPS':        _safe_float(p.get('OPS')),
            'OPS+':       _safe_int(p.get('OPS+')),
            'HR':         _safe_int(p.get('HR')),
            'RBI':        _safe_int(p.get('RBI')),
            'SB':         _safe_int(p.get('SB')),
            'WAR':        _safe_float(p.get('WAR')),
            'RC/G':       _safe_float(p.get('RC/G')),
            'SecA':       _safe_float(p.get('SecA')),
            'SB%':        _safe_float(p.get('SB%')),
            'WPA':        _safe_float(p.get('WPA')),
            'Clutch':     _safe_float(p.get('Clutch')),
            'rationale':  _build_rationale(p, spot),
        })

    return lineup


# =============================================================================
# SECTION 3: FLASK ROUTES
# =============================================================================

def _fmt(val, decimals=3):
    """Format a float for display; return '—' if the value is missing."""
    try:
        f = float(val)
        return '—' if np.isnan(f) else f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return '—'


def _lineup_ids():
    """Return the set of player_ids currently in the starting lineup."""
    return {e['player_id'] for e in LINEUP}


@cubs_bp.route('/')
def index():
    """
    Main project page — lineup, analysis charts, and methodology all in one view.
    """
    if not DATA:
        return render_template('cubs/no_data.html', required=REQUIRED_CSVS, data_dir=DATA_DIR)

    df = DATA['master']
    lineup_pids = _lineup_ids()
    starters_df = df[df['player_id'].isin(lineup_pids)]
    qualified   = df[pd.to_numeric(df['PA'], errors='coerce') >= 30].copy()

    # Quick stats summary
    total_hr  = int(pd.to_numeric(starters_df['HR'],  errors='coerce').fillna(0).sum())
    total_rbi = int(pd.to_numeric(starters_df['RBI'], errors='coerce').fillna(0).sum())
    avg_obp   = pd.to_numeric(starters_df['OBP'], errors='coerce').mean()
    avg_ops   = pd.to_numeric(starters_df['OPS'], errors='coerce').mean()
    total_war = pd.to_numeric(starters_df['WAR'], errors='coerce').sum()

    quick_stats = {
        'total_hr':  total_hr,
        'total_rbi': total_rbi,
        'avg_obp':   _fmt(avg_obp),
        'avg_ops':   _fmt(avg_ops),
        'total_war': _fmt(total_war, 1),
    }

    # Chart data (same logic as the /analysis route, embedded on this page)
    lineup_names = [e['name'] for e in LINEUP]

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

    chart_data = {
        'lineup_names':    lineup_names,
        'scatter_obp_slg': scatter_obp_slg,
        'ops_plus':        ops_plus_data,
        'clutch':          clutch_data,
        'scatter_rcg_wpa': rcg_wpa_pts,
    }

    return render_template(
        'cubs/index.html',
        lineup=LINEUP,
        quick_stats=quick_stats,
        chart_data=json.dumps(chart_data),
    )


@cubs_bp.route('/players')
def players():
    """Roster page — full sortable stats table."""
    if not DATA:
        return render_template('cubs/no_data.html', required=REQUIRED_CSVS, data_dir=DATA_DIR)

    df = DATA['master']
    lineup_ids = list(_lineup_ids())

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
    """Individual player profile with charts and advanced stats."""
    if not DATA:
        return render_template('cubs/no_data.html', required=REQUIRED_CSVS, data_dir=DATA_DIR)

    df   = DATA['master']
    mask = df['player_id'] == player_id
    if not mask.any():
        abort(404)
    p    = df[mask].iloc[0]
    qual = df[pd.to_numeric(df['PA'], errors='coerce') >= 30].copy()

    # Radar chart: 6 dimensions normalized 0-100 vs qualified roster
    radar_stats  = ['OBP', 'SLG', 'ISO', 'WAR', 'OPS+', 'RC/G']
    radar_labels = ['OBP', 'SLG', 'ISO', 'WAR', 'OPS+', 'RC/G']
    radar_values = []
    for stat in radar_stats:
        col_data = pd.to_numeric(qual[stat], errors='coerce').fillna(0)
        val = _safe_float(p.get(stat))
        mn, mx = col_data.min(), col_data.max()
        norm_val = ((val - mn) / (mx - mn) * 100) if mx > mn else 50.0
        radar_values.append(round(max(0.0, min(100.0, norm_val)), 1))

    # Bar chart: player vs team average
    bar_stats  = ['BA', 'OBP', 'SLG', 'OPS']
    bar_player = [_safe_float(p.get(s)) for s in bar_stats]
    bar_avg    = [
        round(float(pd.to_numeric(qual[s], errors='coerce').mean()), 3)
        if s in qual.columns else 0.0
        for s in bar_stats
    ]

    # Advanced stats table
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

    # Baserunning table
    run_fields = [('SB%', 'SB%'), ('RS%', 'RS%'), ('XBT%', 'XBT%'),
                  ('SecA', 'SecA'), ('SBO', 'SB Opps')]
    run_table = [{'stat': lbl, 'value': _fmt(p.get(col), 3)} for col, lbl in run_fields]

    # Situational table
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

    lineup_spot = next(
        (e['spot'] for e in LINEUP if e['player_id'] == player_id),
        None
    )

    player_info = {
        'player_id': player_id,
        'name':      str(p.get('display_name', '—')),
        'pos':       str(p.get('pos_primary', '—')),
        'age':       _safe_int(p.get('Age')),
        'PA':        _safe_int(p.get('PA')),
        'WAR':       _fmt(p.get('WAR'), 1),
        'oWAR':      _fmt(p.get('oWAR'), 1),
        'dWAR':      _fmt(p.get('dWAR'), 1),
        'OPS+':      _safe_int(p.get('OPS+')),
        'BA':        _fmt(p.get('BA')),
        'OBP':       _fmt(p.get('OBP')),
        'SLG':       _fmt(p.get('SLG')),
        'OPS':       _fmt(p.get('OPS')),
        'HR':        _safe_int(p.get('HR')),
        'RBI':       _safe_int(p.get('RBI')),
        'SB':        _safe_int(p.get('SB')),
    }

    return render_template(
        'cubs/player_detail.html',
        player=player_info,
        lineup_spot=lineup_spot,
        chart_data=json.dumps(chart_data),
        adv_table=adv_table,
        run_table=run_table,
        situ_table=situ_table,
    )


@cubs_bp.route('/analysis')
def analysis():
    """Comparative analysis page with four Chart.js charts."""
    if not DATA:
        return render_template('cubs/no_data.html', required=REQUIRED_CSVS, data_dir=DATA_DIR)

    df        = DATA['master']
    qualified = df[pd.to_numeric(df['PA'], errors='coerce') >= 30].copy()
    lineup_names = [e['name'] for e in LINEUP]

    # Chart 1: OBP vs SLG scatter
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

    # Chart 2: OPS+ bar
    ops_df = qualified.copy()
    ops_df['OPS+'] = pd.to_numeric(ops_df['OPS+'], errors='coerce').fillna(0)
    ops_df = ops_df[ops_df['OPS+'] > 0].sort_values('OPS+', ascending=False)
    ops_plus_data = {
        'labels': ops_df['display_name'].tolist(),
        'values': ops_df['OPS+'].tolist(),
    }

    # Chart 3: RC/G vs WPA scatter
    scatter_rcg_wpa = [
        {
            'x': round(_safe_float(r.get('RC/G')), 2),
            'y': round(_safe_float(r.get('WPA')), 2),
            'label': str(r['display_name']),
            'player_id': str(r['player_id']),
            'in_lineup': str(r['display_name']) in lineup_names,
        }
        for _, r in qualified.iterrows()
    ]

    # Chart 4: Clutch bar
    clutch_df = qualified.copy()
    clutch_df['Clutch'] = pd.to_numeric(clutch_df['Clutch'], errors='coerce').fillna(0)
    clutch_df = clutch_df.sort_values('Clutch', ascending=False)
    clutch_data = {
        'labels': clutch_df['display_name'].tolist(),
        'values': [round(float(v), 2) for v in clutch_df['Clutch'].tolist()],
    }

    chart_data = {
        'lineup_names':    lineup_names,
        'scatter_obp_slg': scatter_obp_slg,
        'ops_plus':        ops_plus_data,
        'scatter_rcg_wpa': scatter_rcg_wpa,
        'clutch':          clutch_data,
    }

    return render_template('cubs/analysis.html', chart_data=json.dumps(chart_data))


@cubs_bp.route('/methodology')
def methodology():
    """Methodology page — algorithm explanation, stat definitions, limitations."""
    if not DATA:
        return render_template('cubs/no_data.html', required=REQUIRED_CSVS, data_dir=DATA_DIR)
    return render_template('cubs/methodology.html', lineup=LINEUP)


# =============================================================================
# LOAD DATA AT IMPORT TIME
# Wrapped in try/except so missing CSV files don't crash the whole portfolio.
# The routes check DATA and show a friendly message if it's still empty.
# =============================================================================
try:
    load_all_data()
    _result = compute_lineup(DATA['master'])
    LINEUP.extend(_result)
    print(f"Cubs lineup ready: {[e['name'] for e in LINEUP]}")
except FileNotFoundError as _e:
    print(f"Cubs CSV not found — add files to cubs/data/: {_e}")
except Exception as _e:
    print(f"Cubs data error: {_e}")
