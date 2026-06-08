"""
Rebuilding the 2026 White Sox: A WAR-Based Valuation Model
Author: Kyle Flynn

Four analyses are exposed:
    1. $/WAR surplus valuation          -> build_valuations()
    2. Revenue maximization (W -> $)    -> revenue_model()
    3. Playing time reallocation        -> playing_time_analysis()
    4. wOBA-weighted optimal lineup     -> optimal_lineup()

All analyses operate on the batter CSV in data/whitesox_2026_batting.csv.
Pitcher WAR is excluded from this build because the source CSV is batter-only.

Model assumptions
-----------------
- 2026 market rate: $9.0M / WAR
- Replacement-level team: 48 wins (industry-standard baseline)
- Marginal revenue per win: Gaussian curve peaking at 88 wins ($6.0M/win)
- League-average wOBA used in lineup weighting: .315
- Linear extrapolation from YTD to full season (63 games observed as of 6/8/26)
"""

import csv
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOLLARS_PER_WAR        = 9_000_000
TEAM_GAMES_TO_DATE     = 63
FULL_SEASON_GAMES      = 162
MLB_MIN_SALARY         = 760_000
REPLACEMENT_TEAM_WINS  = 48
PEAK_MARGINAL_REV      = 6_000_000   # $/win at the peak of the curve
PEAK_WINS              = 88          # wins where marginal revenue is highest
REV_CURVE_WIDTH        = 12          # std-dev of the Gaussian, in wins
LEAGUE_AVG_WOBA        = 0.315

# wOBA linear weights (FanGraphs 2024 set, slightly rounded)
WOBA_WEIGHTS = {
    "BB":  0.69,
    "HBP": 0.72,
    "1B":  0.89,
    "2B":  1.27,
    "3B":  1.62,
    "HR":  2.10,
}

SALARY_OVERRIDES: Dict[str, int] = {
    # Guaranteed contracts (FanGraphs Roster Resource, June 2026)
    "Andrew Benintendi":  17_100_000,   # 5 yr/$75M (2023-27), FA after 2027
    "Munetaka Murakami":  16_500_000,   # 2 yr/$34M (2026-27), FA after 2027
    "Austin Hays":         5_000_000,   # 1 yr/$6M + 2027 mutual option
    "Randal Grichuk":      1_250_000,   # 1 yr/$1.25M, FA after 2026
    # Arbitration-eligible
    "Derek Hill":            900_000,   # Arb-1; split contract
    # Pre-arb (MLB minimum approximations)
    "Jarred Kelenic":        740_000,
    "Miguel Vargas":         740_000,
    "Reese McGuire":         740_000,
    "Lenyn Sosa":            740_000,
}


# Injury notes for batters whose CSV row is tagged with an IL_status. The
# status itself comes from the CSV; this dict only adds a human-readable
# injury description where one was reported.
BATTER_INJURY_NOTES: Dict[str, str] = {
    "Munetaka Murakami": "Left quadriceps strain",
    "Everson Pereira":   "Right pectoral strain",
    "Tanner Murray":     "Left shoulder dislocation",
    "Austin Hays":       "Currently on the IL",
    "Jarred Kelenic":    "Designated for assignment",
}


# Pitchers and other non-batters on the IL — these don't appear in the
# batting CSV, so they're listed here directly from the team's injury report.
IL_OTHER: List[Dict] = [
    {"name": "Cannon",   "status": "Rehab",            "injury": "Hip surgery; rehab assignment pending"},
    {"name": "Teel",     "status": "Day-to-day",       "injury": "Right hamstring affecting running"},
    {"name": "Murphy",   "status": "Rehab",            "injury": "Rehab assignment with Triple-A Charlotte"},
    {"name": "Berroa",   "status": "60-day IL",        "injury": "Transferred to 60-day IL"},
    {"name": "Baldwin",  "status": "Season-ending IL", "injury": "Internal brace surgery for right UCL tear"},
    {"name": "Vasil",    "status": "60-day IL",        "injury": "Transferred to 60-day IL"},
    {"name": "Bush",     "status": "60-day IL",        "injury": "On 60-day IL"},
    {"name": "Thorpe",   "status": "15-day IL",        "injury": "On 15-day IL"},
]


def _is_unavailable(row: Dict) -> bool:
    """True if the player is on the IL or has been DFA'd."""
    return bool((row.get("IL_status") or "").strip())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _data_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", "whitesox_2026_batting.csv")


def _advanced_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", "whitesox_2026_advanced.csv")


def load_batting(csv_path: str | None = None) -> List[Dict]:
    csv_path = csv_path or _data_path()
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_advanced(csv_path: str | None = None) -> Dict[str, Dict]:
    """Load advanced/sabermetric stats, keyed by player name."""
    csv_path = csv_path or _advanced_path()
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        return {r["Player"]: r for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# 1. $/WAR surplus valuation
# ---------------------------------------------------------------------------
@dataclass
class PlayerValuation:
    rank: int
    name: str
    age: int
    position: str
    war_ytd: float
    projected_war: float
    salary: int
    market_value: int
    surplus_value: int
    tier: str
    ops_plus: int

    def to_dict(self):
        return asdict(self)


def estimated_salary(name: str) -> int:
    return SALARY_OVERRIDES.get(name, MLB_MIN_SALARY)


def classify_tier(age: int, projected_war: float, surplus: int) -> str:
    if projected_war >= 2.0 and surplus >= 5_000_000:
        return "Core"
    if projected_war >= 2.0 and surplus < 5_000_000:
        return "Trade-Up"
    if age <= 26 and 0 <= projected_war < 2.0:
        return "Develop"
    if projected_war < 0 or (age >= 30 and surplus < 0):
        return "Cut/Replace"
    return "Hold"


def build_valuations(csv_path: str | None = None) -> List[PlayerValuation]:
    rows = load_batting(csv_path)
    team_scale = FULL_SEASON_GAMES / TEAM_GAMES_TO_DATE
    out: List[PlayerValuation] = []

    for row in rows:
        name = row["Player"]
        war = float(row["WAR"])
        # Scale every player by the team's pace. Scaling by the player's own
        # games (e.g., 162/3 for Drew Romo) blows up small-sample noise; using
        # team pace projects each player at their current usage rate.
        projected = round(war * team_scale, 2)

        salary = estimated_salary(name)
        market_value = int(projected * DOLLARS_PER_WAR)
        surplus = market_value - salary
        tier = classify_tier(int(row["Age"]), projected, surplus)

        out.append(PlayerValuation(
            rank=int(row["Rk"]),
            name=name,
            age=int(row["Age"]),
            position=row["Pos"],
            war_ytd=war,
            projected_war=projected,
            salary=salary,
            market_value=market_value,
            surplus_value=surplus,
            tier=tier,
            ops_plus=int(row["OPS_plus"]),
        ))

    out.sort(key=lambda p: p.surplus_value, reverse=True)
    return out


def valuation_summary(vals: List[PlayerValuation]) -> Dict:
    return {
        "total_projected_war": round(sum(p.projected_war for p in vals), 1),
        "total_salary":        sum(p.salary for p in vals),
        "total_surplus":       sum(p.surplus_value for p in vals),
        "n_players":           len(vals),
        "tier_counts":         {t: sum(1 for p in vals if p.tier == t)
                                for t in ("Core", "Trade-Up", "Develop", "Hold", "Cut/Replace")},
    }


# ---------------------------------------------------------------------------
# 2. Revenue maximization model (Wins -> Attendance -> Revenue)
# ---------------------------------------------------------------------------
def marginal_revenue_per_win(wins: float) -> float:
    """
    Gaussian marginal revenue curve. Each additional win is worth the most
    when the team sits near the playoff threshold (~88 wins) and less when
    the team is either well below contention or already locked in.
    """
    return PEAK_MARGINAL_REV * math.exp(-((wins - PEAK_WINS) ** 2) / (2 * REV_CURVE_WIDTH ** 2))


def cumulative_revenue(wins: float) -> float:
    """
    Integrate marginal revenue from replacement-level (48 wins) up to `wins`.
    Approximated with a discrete sum over integer-win steps.
    """
    total = 0.0
    for w in range(REPLACEMENT_TEAM_WINS + 1, int(wins) + 1):
        total += marginal_revenue_per_win(w)
    return total


def projected_team_wins(total_pwar: float) -> int:
    return int(round(REPLACEMENT_TEAM_WINS + total_pwar))


def revenue_model(vals: List[PlayerValuation]) -> Dict:
    total_pwar = sum(p.projected_war for p in vals)
    projected_wins = projected_team_wins(total_pwar)

    # Revenue uplift scenarios: what does each additional WAR investment buy?
    scenarios = []
    for added_war in (0, 5, 10, 15, 20):
        w = projected_team_wins(total_pwar + added_war)
        scenarios.append({
            "added_war": added_war,
            "wins": w,
            "revenue_above_replacement": round(cumulative_revenue(w)),
            "marginal_per_win": round(marginal_revenue_per_win(w)),
        })

    curve = [{"wins": w, "marginal": round(marginal_revenue_per_win(w))}
             for w in range(REPLACEMENT_TEAM_WINS, 111)]

    return {
        "projected_wins":     projected_wins,
        "projected_pwar":     round(total_pwar, 1),
        "current_revenue":    round(cumulative_revenue(projected_wins)),
        "scenarios":          scenarios,
        "curve":              curve,
        "peak_wins":          PEAK_WINS,
        "peak_marginal":      PEAK_MARGINAL_REV,
    }


# ---------------------------------------------------------------------------
# 3. Playing time analysis
# ---------------------------------------------------------------------------
def playing_time_analysis(vals: List[PlayerValuation], csv_path: str | None = None) -> Dict:
    """
    For each player, compute WAR per 100 PA, current PA pace, and a recommended
    PA shift. Bench / IL players with positive WAR/PA are flagged as candidates
    for added playing time; veterans with negative WAR/PA are flagged for cuts.
    """
    rows = {r["Player"]: r for r in load_batting(csv_path)}
    team_scale = FULL_SEASON_GAMES / TEAM_GAMES_TO_DATE
    results = []

    for p in vals:
        row = rows[p.name]
        pa = int(row["PA"])
        if pa < 1:
            continue
        war_per_100 = round(p.war_ytd / pa * 100, 3)
        projected_pa = int(pa * team_scale)

        if war_per_100 >= 0.4:
            rec = "Increase PA"
        elif war_per_100 <= -0.2:
            rec = "Reduce PA"
        else:
            rec = "Hold PA"

        results.append({
            "name": p.name,
            "position": p.position,
            "age": p.age,
            "pa_ytd": pa,
            "projected_pa": projected_pa,
            "war_per_100_pa": war_per_100,
            "ops_plus": p.ops_plus,
            "recommendation": rec,
        })

    results.sort(key=lambda r: r["war_per_100_pa"], reverse=True)
    return {
        "rows": results,
        "n_increase": sum(1 for r in results if r["recommendation"] == "Increase PA"),
        "n_reduce":   sum(1 for r in results if r["recommendation"] == "Reduce PA"),
    }


# ---------------------------------------------------------------------------
# 4. wOBA-weighted optimal batting order
# ---------------------------------------------------------------------------
def compute_woba(row: Dict) -> float:
    """
    wOBA = (0.69*BB + 0.72*HBP + 0.89*1B + 1.27*2B + 1.62*3B + 2.10*HR)
           / (AB + BB - IBB + SF + HBP)
    The provided CSV does not include HBP/SF/IBB columns broken out at this
    detail; we approximate the denominator as PA and ignore HBP/IBB in the
    numerator. This is a deliberate simplification documented in the writeup.
    """
    pa = int(row["PA"])
    if pa < 1:
        return 0.0
    h  = int(row["H"])
    d  = int(row["2B"])
    t  = int(row["3B"])
    hr = int(row["HR"])
    bb = int(row["BB"])
    singles = h - d - t - hr

    num = (WOBA_WEIGHTS["BB"]  * bb
         + WOBA_WEIGHTS["1B"]  * singles
         + WOBA_WEIGHTS["2B"]  * d
         + WOBA_WEIGHTS["3B"]  * t
         + WOBA_WEIGHTS["HR"]  * hr)
    return round(num / pa, 3)


def optimal_lineup(csv_path: str | None = None, min_pa: int = 20) -> Dict:
    """
    Build a 9-man lineup from qualified hitters and order them using The Book's
    heuristic:
        Slot 1: best OBP (table-setter)
        Slot 2: best overall hitter (highest wOBA)  -- gets the most PA after 1
        Slot 3: third-best hitter (fewest PA among the top 4 slots)
        Slot 4: second-best hitter with power preference
        Slot 5: next-best hitter with power
        Slots 6-9: remaining hitters by wOBA descending
    """
    rows = load_batting(csv_path)
    eligible = []
    for r in rows:
        if int(r["PA"]) < min_pa:
            continue
        if _is_unavailable(r):
            continue
        eligible.append({
            "name":    r["Player"],
            "pos":     r["Pos"],
            "pa":      int(r["PA"]),
            "obp":     float(r["OBP"]),
            "slg":     float(r["SLG"]),
            "ops":     float(r["OPS"]),
            "woba":    compute_woba(r),
            "hr":      int(r["HR"]),
            "ops_plus": int(r["OPS_plus"]),
        })

    if len(eligible) < 9:
        # Fall back: relax min_pa
        return optimal_lineup(csv_path, min_pa=max(min_pa - 10, 1))

    by_woba   = sorted(eligible, key=lambda x: x["woba"], reverse=True)
    by_obp    = sorted(eligible, key=lambda x: x["obp"], reverse=True)
    by_power  = sorted(eligible, key=lambda x: x["slg"], reverse=True)

    used = set()
    placements: Dict[int, Dict] = {}

    def take(candidates, label):
        for c in candidates:
            if c["name"] not in used:
                used.add(c["name"])
                return {**c, "selected_for": label}
        return None

    # The Book (Tango/Lichtman/Dolphin): the #2 hitter gets the most leveraged
    # PA, then #4, then #1. So put your best two hitters there first.
    placements[2] = take(by_woba,  "Best hitter (highest wOBA)")
    placements[4] = take(by_power, "Cleanup (highest SLG among rest)")
    placements[1] = take(by_obp,   "Leadoff (highest OBP among rest)")
    placements[3] = take(by_woba,  "Third-best hitter (sees fewest PA up top)")
    placements[5] = take(by_power, "Secondary power")
    for slot in range(6, 10):
        placements[slot] = take(by_woba, f"wOBA rank #{slot}")

    lineup = [placements[i] for i in range(1, 10)]

    # Expected runs/game proxy: mean wOBA scaled vs league average (very rough)
    mean_woba = sum(p["woba"] for p in lineup) / 9
    runs_per_game_est = round(4.50 * (mean_woba / LEAGUE_AVG_WOBA), 2)

    return {
        "lineup":              lineup,
        "mean_woba":           round(mean_woba, 3),
        "league_woba":         LEAGUE_AVG_WOBA,
        "runs_per_game_est":   runs_per_game_est,
        "candidates_considered": len(eligible),
    }


# ---------------------------------------------------------------------------
# 5. Injured / DFA list
# ---------------------------------------------------------------------------
def injured_list(csv_path: str | None = None) -> Dict:
    """
    Build the full club injury report:
      - position players from the batting CSV whose IL_status is set
      - pitchers / non-batters from IL_OTHER (provided by the team report)
    """
    rows = load_batting(csv_path)
    batters = []
    for r in rows:
        if not _is_unavailable(r):
            continue
        batters.append({
            "name":     r["Player"],
            "position": r["Pos"],
            "status":   r["IL_status"],
            "injury":   BATTER_INJURY_NOTES.get(r["Player"], "—"),
            "pa":       int(r["PA"]),
            "war_ytd":  float(r["WAR"]),
            "ops":      float(r["OPS"]),
        })

    # Sort by status (DFA / season-ending first, then 60-day, then 10/15-day)
    order = {"DFA": 0, "Season-ending IL": 1, "60-day IL": 2,
             "15-day IL": 3, "10-day IL": 4, "Rehab": 5, "Day-to-day": 6}
    batters.sort(key=lambda b: order.get(b["status"], 99))
    others = sorted(IL_OTHER, key=lambda b: order.get(b["status"], 99))

    return {
        "batters":   batters,
        "others":    others,
        "total":     len(batters) + len(others),
    }


# ---------------------------------------------------------------------------
# 6. Top-PA lineup (everyday regulars)
# ---------------------------------------------------------------------------
def top_pa_lineup(csv_path: str | None = None) -> Dict:
    """
    Build a lineup from the nine available batters with the most PA.
    IL/DFA players are excluded. Slotting then follows The Book using the
    same heuristic as optimal_lineup() but on the PA-restricted pool.
    """
    rows = load_batting(csv_path)
    available = [r for r in rows if not _is_unavailable(r) and int(r["PA"]) > 0]
    if len(available) < 9:
        return {"lineup": [], "candidates_considered": len(available)}

    available.sort(key=lambda r: int(r["PA"]), reverse=True)
    pool = available[:9]

    enriched = []
    for r in pool:
        enriched.append({
            "name":     r["Player"],
            "pos":      r["Pos"],
            "pa":       int(r["PA"]),
            "obp":      float(r["OBP"]),
            "slg":      float(r["SLG"]),
            "ops":      float(r["OPS"]),
            "ops_plus": int(r["OPS_plus"]),
            "woba":     compute_woba(r),
            "hr":       int(r["HR"]),
        })

    by_woba  = sorted(enriched, key=lambda x: x["woba"], reverse=True)
    by_obp   = sorted(enriched, key=lambda x: x["obp"],  reverse=True)
    by_power = sorted(enriched, key=lambda x: x["slg"],  reverse=True)

    used = set()
    placements: Dict[int, Dict] = {}

    def take(candidates, label):
        for c in candidates:
            if c["name"] not in used:
                used.add(c["name"])
                return {**c, "selected_for": label}
        return None

    placements[2] = take(by_woba,  "Best hitter among regulars (wOBA)")
    placements[4] = take(by_power, "Cleanup — most power among regulars (SLG)")
    placements[1] = take(by_obp,   "Leadoff — best OBP among regulars")
    placements[3] = take(by_woba,  "Third-best hitter")
    placements[5] = take(by_power, "Secondary power")
    for slot in range(6, 10):
        placements[slot] = take(by_woba, f"wOBA rank #{slot} of the regulars")

    lineup = [placements[i] for i in range(1, 10)]
    total_pa = sum(p["pa"] for p in lineup)
    mean_woba = sum(p["woba"] for p in lineup) / 9

    return {
        "lineup":               lineup,
        "total_pa":             total_pa,
        "mean_pa":              round(total_pa / 9, 1),
        "mean_woba":            round(mean_woba, 3),
        "candidates_considered": len(available),
    }


# ---------------------------------------------------------------------------
# 7. Sabermetric lineup (Rbat+ / rOBA / ISO / WPA composite)
# ---------------------------------------------------------------------------
def _sabermetric_score(adv: Dict) -> float:
    """
    Composite offensive score that weights:
      - Rbat+ (park-adjusted batting runs vs avg, 100 = league avg) — primary
      - rOBA centered on league average — secondary
      - WPA (clutch-leverage credit) — light tilt
    Returns a unitless score where higher = better offensive contributor.
    """
    rbat_plus = float(adv.get("Rbat_plus", 100))
    roba      = float(adv.get("rOBA", 0.315))
    wpa       = float(adv.get("WPA", 0.0))
    return rbat_plus + 50.0 * (roba - 0.315) + 2.0 * wpa


def sabermetric_lineup(csv_path: str | None = None,
                       advanced_path: str | None = None,
                       min_pa: int = 30) -> Dict:
    """
    Build an optimal lineup ranked by a composite sabermetric score
    (Rbat+ / rOBA / WPA). The Book leverages slotting:
      #2 sees most leveraged PA, then #4, then #1.
    Excludes IL/DFA players and requires a PA floor to filter noise.
    """
    rows = load_batting(csv_path)
    adv  = load_advanced(advanced_path)

    enriched = []
    for r in rows:
        if _is_unavailable(r):
            continue
        if int(r["PA"]) < min_pa:
            continue
        a = adv.get(r["Player"])
        if a is None:
            continue
        enriched.append({
            "name":      r["Player"],
            "pos":       r["Pos"],
            "pa":        int(r["PA"]),
            "obp":       float(r["OBP"]),
            "slg":       float(r["SLG"]),
            "ops":       float(r["OPS"]),
            "ops_plus":  int(r["OPS_plus"]),
            "roba":      float(a["rOBA"]),
            "rbat_plus": int(float(a["Rbat_plus"])),
            "iso":       float(a["ISO"]),
            "bb_pct":    float(a["BB_pct"]),
            "so_pct":    float(a["SO_pct"]),
            "wpa":       float(a["WPA"]),
            "re24":      float(a["RE24"]),
            "score":     round(_sabermetric_score(a), 1),
        })

    if len(enriched) < 9:
        return {"lineup": [], "candidates_considered": len(enriched)}

    by_score = sorted(enriched, key=lambda x: x["score"],     reverse=True)
    by_power = sorted(enriched, key=lambda x: x["iso"] + x["slg"], reverse=True)
    by_obp   = sorted(enriched, key=lambda x: x["obp"],       reverse=True)

    used = set()
    placements: Dict[int, Dict] = {}

    def take(candidates, label):
        for c in candidates:
            if c["name"] not in used:
                used.add(c["name"])
                return {**c, "selected_for": label}
        return None

    placements[2] = take(by_score, "Best overall offensive value (Rbat+ / rOBA)")
    placements[4] = take(by_power, "Cleanup — most isolated power (ISO + SLG)")
    placements[1] = take(by_obp,   "Leadoff — highest OBP for table-setting")
    placements[3] = take(by_score, "Third-best sabermetric hitter")
    placements[5] = take(by_power, "Secondary power for the middle")
    for slot in range(6, 10):
        placements[slot] = take(by_score, f"Sabermetric rank #{slot}")

    lineup = [placements[i] for i in range(1, 10)]
    mean_rbat_plus = sum(p["rbat_plus"] for p in lineup) / 9
    mean_roba      = sum(p["roba"]      for p in lineup) / 9
    total_wpa      = sum(p["wpa"]       for p in lineup)

    return {
        "lineup":                lineup,
        "mean_rbat_plus":        round(mean_rbat_plus, 1),
        "mean_roba":             round(mean_roba, 3),
        "total_wpa":             round(total_wpa, 2),
        "candidates_considered": len(enriched),
    }


# ---------------------------------------------------------------------------
# CLI for manual sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vals = build_valuations()
    summ = valuation_summary(vals)
    rev  = revenue_model(vals)
    pt   = playing_time_analysis(vals)
    lo   = optimal_lineup()

    print("=== Surplus Value ===")
    for p in vals[:8]:
        print(f"  {p.name:<22} pWAR {p.projected_war:>5.2f}  surplus ${p.surplus_value:>14,}  {p.tier}")
    print(f"  Roster total: {summ['total_projected_war']} pWAR  surplus ${summ['total_surplus']:,}")

    print("\n=== Revenue Model ===")
    print(f"  Projected wins: {rev['projected_wins']} | est. revenue above replacement: ${rev['current_revenue']:,}")
    for s in rev["scenarios"]:
        print(f"  +{s['added_war']:>2} WAR -> {s['wins']} W  | revenue ${s['revenue_above_replacement']:,}")

    print("\n=== Playing Time ===")
    for r in pt["rows"][:6]:
        print(f"  {r['name']:<22} WAR/100PA {r['war_per_100_pa']:>+0.3f} | {r['recommendation']}")

    print("\n=== Optimal Lineup ===")
    for i, p in enumerate(lo["lineup"], 1):
        print(f"  {i}. {p['name']:<22} ({p['pos']:<3}) wOBA {p['woba']:.3f} | {p['selected_for']}")
    print(f"  Mean wOBA: {lo['mean_woba']}  |  R/G estimate: {lo['runs_per_game_est']}")
