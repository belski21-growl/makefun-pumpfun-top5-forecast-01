#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry — Python 3.13, standard library only.

Forecasts the 5 largest pump.fun coins by market cap at the 7 answer reads in
/input/task.json, from the 10-minute marks in /input/history.ndjson.

Method (all calibrated from the history itself, nothing hardcoded to the dev
rehearsal):
  * Per coin, a log10(market cap in SOL) random walk fitted on its own marks:
    shrunk drift (short-term momentum decays toward a rank-dependent prior
    with a ~1.5 h time constant) plus idiosyncratic volatility that is
    rank-dependent (top-20 coins swing ~1 dex/hr, small coins ~0.3) and
    includes each coin's own return volatility, so jumps are not smoothed
    away by the fitted trend; plus a common market factor (moves sizes,
    not ranks).
  * A seeded Monte Carlo (4000 sims x 7 reads) over the candidate pool gives,
    per place, P(coin at place) and the distribution of the place's cap.
  * Probabilities are the honest Monte Carlo estimates, scaled by (1 - newborn
    mass): coins born after the history ends cannot be named, so a per-place
    newborn reserve stays in the unnamed share (the practice reads had 2 of
    the top 5 at the last read born after the cut; in the dev marks the top 5
    turned over completely within 40 minutes, all via newborns).
  * Each place's size distribution is a mixture of the Monte Carlo caps and
    a newborn template built from the last mark's top-5 log-caps: a newborn
    taking a place has a cap like a current top-5 coin, far above what the
    simulated old coins reach. The mixture is scaled to leave a 5% unnamed
    share as fat-tail insurance.
  * Naming is greedy-optimal for the log score: a coin/bin is named only while
    its probability beats what it would get as an unnamed share; far-tail
    coins with negligible probability are named at p=0 (the rules allow
    this: it removes them from the unnamed share, so the share left for the
    movers and the newborns is larger), guarded by a gain-vs-loss test and a
    last-rank floor so no plausible top-5 coin is ever zeroed. p=0 is NOT
    used for size bins: the concentrated share would sit on a far-tail bin
    the actual cap almost never lands in, while each zeroed bin becomes a
    -9.21 floor risk.
  * top5 is the argmax coin per place (distinct).

Deterministic: fixed seed, no wall-clock, no network, no randomness outside
random.Random(SEED). Output is /output/forecast.json (names from task.json),
exactly {probabilities, top5, sizes}.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

SEED = 20261112
N_SIMS = 4000
TAU_H = 1.5            # momentum decay time constant (hours)
MU_LONG_FALLBACK = -0.02  # pooled long-run drift, log10 SOL per hour
SIGMA_FLOOR_H = 0.30   # per-hour idiosyncratic sd floor
SIGMA_CAP_H = 1.20     # per-hour idiosyncratic sd cap
TOP_SIGMA_FLOOR = 0.60  # top-20 coins are proven jumpy (MAD ~0.9/hr in dev)
SIGMA_MIN_TOTAL = 0.30  # total per-read sd floor (model uncertainty)
SIGMA_MAX_TOTAL = 1.20
MARKET_SIGMA_H = 0.08  # common market factor, per-hour sd
# P(a coin born after the history ends takes the place), per place 1..5.
# Practice: 2 of the top 5 at the last read were born after the cut.
NEWBORN = [0.22, 0.27, 0.32, 0.37, 0.42]
POOL_TOP = 80          # candidates: last mark's top N by cap
POOL_EXTRA_TOP = 30    # plus any coin top-N at any mark
SIZE_KEEP = 0.95       # share of the empirical size mass named (rest unnamed)
MAX_P0_PER_PLACE = 60  # cap on p=0 namings per place
MIN_P0_LAST_RANK = 15  # never p=0 a coin ranked better than this at last mark
MICRO = 1_000_000
LOG_FLOOR = -9.210340  # ln(0.0001), the scorer's per-component floor


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def log10_cap_sol(lamports):
    return math.log10(int(lamports)) - 9.0


def floor6(p):
    return math.floor(p * MICRO) / MICRO


def linfit_slope(xs, ys):
    """OLS slope of ys on xs; None when degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(in_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    out_name = task["output"]
    chart_size = int(task.get("chart_size", 100))
    reads = [parse_ts(r) for r in task["reads"]]
    n_bins = len(edges) + 1  # 82

    marks = []
    with open(os.path.join(in_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") and m.get("rows"):
                marks.append(m)
    marks.sort(key=lambda m: m["at"])
    if not marks:
        write_fallback(out_dir, out_name, places, n_bins)
        return

    t0 = parse_ts(marks[0]["at"])
    last_mark = marks[-1]
    t_last = parse_ts(last_mark["at"])
    horizons = [max((r - t_last).total_seconds() / 3600.0, 0.0) for r in reads]

    # ---- per-coin series: (hours since first mark, log10 cap SOL) ----
    series = {}
    last_row = {}
    for m in marks:
        t = (parse_ts(m["at"]) - t0).total_seconds() / 3600.0
        rows = sorted(m["rows"], key=lambda r: r["rank"])
        for r in rows:
            mint = r["mint"]
            try:
                y = log10_cap_sol(r["market_cap_lamports"])
            except (ValueError, TypeError):
                continue
            series.setdefault(mint, []).append(
                (t, y, r.get("venue", "curve"), r.get("created_at"))
            )
            last_row[mint] = (r["rank"], y, r.get("venue", "curve"))

    # ---- pooled drift and volatility ----
    slopes = []
    resid_sq = []
    for mint, pts in series.items():
        if len(pts) >= 3:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            s = linfit_slope(xs, ys)
            if s is None:
                continue
            slopes.append(s)
            b = sum(ys) / len(ys) - s * (sum(xs) / len(xs))
            for x, y in zip(xs, ys):
                resid_sq.append((y - (b + s * x)) ** 2)
    slopes.sort()
    mu_long = slopes[len(slopes) // 2] if slopes else MU_LONG_FALLBACK
    mu_long = max(min(mu_long, 0.05), -0.15)
    pooled_var = (sum(resid_sq) / max(len(resid_sq) - 1, 1)) if resid_sq else 0.04
    # residual variance is over ~10-min steps; convert to per-hour sd.
    pooled_sigma_h = math.sqrt(pooled_var) * math.sqrt(6.0)
    pooled_sigma_h = max(min(pooled_sigma_h, SIGMA_CAP_H), SIGMA_FLOOR_H)

    # ---- candidate pool ----
    last_rows = sorted(last_mark["rows"], key=lambda r: r["rank"])
    pool = {}
    for r in last_rows[:POOL_TOP]:
        pool[r["mint"]] = True
    for m in marks:
        for r in sorted(m["rows"], key=lambda x: x["rank"])[:POOL_EXTRA_TOP]:
            pool[r["mint"]] = True
    # momentum risers present at the last mark
    for mint, pts in series.items():
        if len(pts) >= 3 and mint in last_row:
            s = linfit_slope([p[0] for p in pts], [p[1] for p in pts])
            if s is not None and s > 0.10 and last_row[mint][0] <= 100:
                pool[mint] = True
    pool = sorted(pool.keys())
    if len(pool) < places:
        pool = sorted(last_row.keys())[: max(places, len(last_row))]

    # ---- per-coin drift / volatility ----
    coins = []
    for mint in pool:
        pts = sorted(series.get(mint, []))
        if not pts:
            continue
        y0 = pts[-1][1]
        venue = pts[-1][2]
        rank_last, _, _ = last_row.get(mint, (999, y0, venue))
        # Drift prior is rank-dependent: top-20 coins that are still top-20
        # tend to be the growers (median 10-min return +0.38/hr in dev,
        # selection-inflated); use a conservative +0.15, others pooled.
        mu_prior = 0.15 if rank_last <= 20 else mu_long
        if len(pts) >= 3:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            s = linfit_slope(xs, ys)
            if s is None:
                s = mu_prior
            n = len(pts)
            s = s * n / (n + 3.0) + mu_prior * 3.0 / (n + 3.0)
            b = sum(ys) / n - s * (sum(xs) / n)
            var = sum((y - (b + s * x)) ** 2 for x, y in zip(xs, ys)) / max(n - 2, 1)
            sig = math.sqrt(var) * math.sqrt(6.0)
            # Own return volatility: captures the jumps a smooth fit absorbs.
            rets = []
            for i in range(len(pts) - 1):
                dt = pts[i + 1][0] - pts[i][0]
                if dt > 0:
                    rets.append((pts[i + 1][1] - pts[i][1]) / dt)
            if len(rets) >= 2:
                mret = sum(rets) / len(rets)
                sig = max(
                    sig,
                    min(1.2, (sum((r - mret) ** 2 for r in rets) / len(rets)) ** 0.5),
                )
        elif len(pts) == 2:
            s = (pts[1][1] - pts[0][1]) / max(pts[1][0] - pts[0][0], 1e-9)
            s = s * 0.3 + mu_prior * 0.7
            sig = 0.45
        else:
            s = mu_prior
            sig = pooled_sigma_h
        s = max(min(s, 0.6), -0.6)
        sig = max(min(sig, SIGMA_CAP_H), SIGMA_FLOOR_H)
        if rank_last <= 20:
            sig = max(sig, TOP_SIGMA_FLOOR)
        if venue == "curve":
            sig *= 1.15
        age_h = None
        created = pts[-1][3]
        if created:
            try:
                age_h = (t_last - parse_ts(created)).total_seconds() / 3600.0
            except ValueError:
                age_h = None
        if age_h is not None and age_h < 1.0:
            sig *= 1.25
        coins.append(
            {"mint": mint, "y0": y0, "mu": s, "sig": min(sig, SIGMA_CAP_H),
             "rank_last": rank_last}
        )
    if len(coins) < places:
        write_fallback(out_dir, out_name, places, n_bins)
        return

    # ---- Monte Carlo over reads ----
    rng = random.Random(SEED)
    n_coins = len(coins)
    place_counts = [[0.0] * n_coins for _ in range(places)]
    place_caps = [[] for _ in range(places)]  # log10 SOL of the coin at place k
    n_draws = 0
    for h in horizons:
        decay = (1.0 - math.exp(-h / TAU_H)) / (h / TAU_H) if h > 1e-9 else 1.0
        means = []
        sds = []
        for c in coins:
            mu_eff = mu_long + (c["mu"] - mu_long) * decay
            means.append(c["y0"] + mu_eff * h)
            sds.append(min(max(c["sig"] * math.sqrt(max(h, 0.05)), SIGMA_MIN_TOTAL),
                           SIGMA_MAX_TOTAL))
        mkt_sd = MARKET_SIGMA_H * math.sqrt(max(h, 0.05))
        for _ in range(N_SIMS):
            mkt = rng.gauss(0.0, mkt_sd) if mkt_sd > 0 else 0.0
            draws = [means[i] + mkt + rng.gauss(0.0, sds[i]) for i in range(n_coins)]
            order = sorted(range(n_coins), key=lambda i: (-draws[i], coins[i]["mint"]))
            for k in range(places):
                place_counts[k][order[k]] += 1.0
                place_caps[k].append(draws[order[k]])
            n_draws += 1
    total_draws = float(n_draws)
    # q[c][k]: P(coin c at place k), averaged over reads
    q = [[place_counts[k][i] / total_draws for i in range(n_coins)]
         for k in range(places)]

    # ---- assemble probabilities ----
    probabilities = []
    p0_rows = []
    for k in range(places):
        nb = NEWBORN[k] if k < len(NEWBORN) else NEWBORN[-1]
        # true P(coin at place) = q * (1 - nb)
        cands = sorted(
            ((q[k][i] * (1.0 - nb), coins[i]) for i in range(n_coins)),
            key=lambda t: -t[0],
        )
        named, named_sum = greedy_name(
            [(c["mint"], p) for p, c in cands], chart_size
        )
        for mint, p in named:
            fp = floor6(p)
            if fp > 0:
                probabilities.append({"mint": mint, "place": k + 1, "p": fp})
        # p=0 tail: coins likely on the chart but ~never top-5 here.
        named_mints = {m for m, _ in named}
        tail = sorted(
            ((p, c) for p, c in cands if c["mint"] not in named_mints),
            key=lambda t: t[0],
        )
        n_unnamed = chart_size - len(named)
        u = 1.0 - named_sum
        p0 = greedy_p0(tail, u, n_unnamed, MAX_P0_PER_PLACE)
        for mint in p0:
            p0_rows.append({"mint": mint, "place": k + 1, "p": 0})
    probabilities.extend(p0_rows)

    # ---- assemble sizes (empirical place-cap distribution) ----
    # The Monte Carlo only simulates coins that exist in the history, but at
    # the reads the top 5 is often taken by NEWBORNS (4 of 5 at the 20:30
    # dev mark), whose caps are as large as a current top-5 coin's - far
    # above what the simulated old coins reach. So each place's size
    # distribution is a mixture: weight (1 - nb) the Monte Carlo caps, weight
    # nb a newborn template built from the last mark's top-5 log-caps
    # (kernel sd 0.3 dex). Without this the size score floors whenever a
    # newborn takes a place, which is the common case, not the exception.
    # The mixture is then scaled by SIZE_KEEP, leaving a 5% unnamed share as
    # insurance against caps outside the simulated support (fat-tail jumps
    # the diffusion model understates). The p=0 pass below then concentrates
    # that share on the plausible bins.
    last_top5_logs = []
    for r in sorted(last_mark["rows"], key=lambda x: x["rank"])[:places]:
        try:
            last_top5_logs.append(log10_cap_sol(r["market_cap_lamports"]))
        except (KeyError, TypeError, ValueError):
            pass

    def newborn_bin_p():
        out = [0.0] * n_bins
        if not last_top5_logs:
            return out
        sig_nb = 0.3
        for c in last_top5_logs:
            for b in range(n_bins):
                lo = edges[b - 1] if b > 0 else None
                hi = edges[b] if b < n_bins - 1 else None
                phi_hi = 1.0 if hi is None else norm_cdf((hi - c) / sig_nb)
                phi_lo = 0.0 if lo is None else norm_cdf((lo - c) / sig_nb)
                out[b] += (phi_hi - phi_lo) / len(last_top5_logs)
        return out

    nb_template = newborn_bin_p()
    sizes = []
    for k in range(places):
        nb = NEWBORN[k] if k < len(NEWBORN) else NEWBORN[-1]
        samples = place_caps[k]
        samples.sort()
        n = len(samples)
        bin_p = [0.0] * n_bins
        for v in samples:
            b = bin_of(v, edges)
            bin_p[b] += 1.0 / n
        # mixture: Monte Carlo caps + newborn template, then keep SIZE_KEEP
        bin_p = [
            SIZE_KEEP * ((1.0 - nb) * p + nb * t)
            for p, t in zip(bin_p, nb_template)
        ]
        cands = sorted(
            ((bin_p[b], b) for b in range(n_bins)), key=lambda t: -t[0]
        )
        named, named_sum = greedy_name(
            [(b, p) for p, b in cands], n_bins
        )
        for b, p in named:
            fp = floor6(p)
            if fp > 0:
                sizes.append({"place": k + 1, "bin": b, "p": fp})
        # p=0 far-tail bins: DISABLED for sizes. The p=0 trick pays for
        # coins because the unnamed share concentrates on coins that really
        # do take places (newborns). For size bins it does not: the
        # concentrated share would sit on a far-tail bin the actual cap
        # almost never lands in, while each zeroed bin becomes a -9.21 floor
        # risk. The tail bins stay unnamed instead (share ~0.06 spread over
        # them, ln ~ -6 rather than the floor).

    # ---- top5: argmax per place, distinct ----
    top5 = []
    for k in range(places):
        best = None
        best_q = -1.0
        for i in range(n_coins):
            if coins[i]["mint"] in top5:
                continue
            if q[k][i] > best_q:
                best_q = q[k][i]
                best = coins[i]["mint"]
        if best is None:
            best = coins[k]["mint"]
        top5.append(best)
    if len(set(top5)) < places:
        for c in coins:
            if c["mint"] not in top5:
                top5.append(c["mint"])
            if len(top5) == places:
                break

    forecast = {"probabilities": probabilities, "top5": top5[:places],
                "sizes": sizes}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
        json.dump(forecast, f, sort_keys=True)
        f.write("\n")


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bin_of(v, edges):
    """Bin index 0..len(edges) for log10 cap v."""
    if v < edges[0]:
        return 0
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if edges[mid] <= v:
            lo = mid + 1
        else:
            hi = mid
    return lo  # edges[lo-1] <= v < edges[lo]; lo == len(edges) -> last bin


def greedy_name(cands, total_outcomes):
    """Name (key, p) pairs while p beats the unnamed share it would get.

    cands: [(key, p)] sorted by p desc. Returns (named, named_sum).
    Naming stops when the next p no longer beats (1 - S) / (total - named).
    """
    named = []
    s = 0.0
    for key, p in cands:
        denom = total_outcomes - len(named)
        if denom <= 0:
            break
        # p <= 0 never belongs in the named pass (it would be a p=0 row, and
        # a floating-point sum a hair above 1.0 must not make 0 "pass").
        if p > 0 and p > (1.0 - s) / denom:
            named.append((key, p))
            s += p
        else:
            break
    return named, s


def greedy_p0(tail, u, n_unnamed, cap):
    """p=0 namings for coins: smallest true-p first, while gain beats loss.

    tail: [(true_p, coin)] sorted by true_p asc. Gain of removing one more
    coin from the unnamed pool: u * ln(N/(N-1)); loss if it does take the
    place: true_p * (ln(u/N) + 9.210340). Only coins ranked >= MIN_P0_LAST_RANK.
    """
    out = []
    n = n_unnamed
    for p, c in tail:
        if len(out) >= cap or n <= 1:
            break
        if c["rank_last"] < MIN_P0_LAST_RANK:
            continue
        if p <= 0:
            out.append(c["mint"])
            n -= 1
            continue
        gain = u * math.log(n / (n - 1.0))
        # loss if it does take the place: p * (9.210340 + ln(u/n))
        loss = p * (9.210340 + math.log(max(u / n, 1e-12)))
        if gain > loss:
            out.append(c["mint"])
            n -= 1
    return out


def greedy_p0_bins(tail, u, n_unnamed, cap):
    """p=0 namings for size bins: same gain/loss test as coins."""
    out = []
    n = n_unnamed
    for p, b in tail:
        if len(out) >= cap or n <= 1:
            break
        if p <= 0:
            out.append(b)
            n -= 1
            continue
        gain = u * math.log(n / (n - 1.0))
        loss = p * (9.210340 + math.log(max(u / n, 1e-12)))
        if gain > loss:
            out.append(b)
            n -= 1
    return out


def write_fallback(out_dir, out_name, places, n_bins):
    """Last-resort valid forecast when the history holds no usable marks."""
    placeholders = [
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAApump",
        "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBpump",
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCpump",
        "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDpump",
        "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEpump",
    ]
    sizes = []
    per_bin = floor6(1.0 / n_bins)
    for k in range(places):
        for b in range(n_bins):
            if per_bin > 0:
                sizes.append({"place": k + 1, "bin": b, "p": per_bin})
    forecast = {"probabilities": [], "top5": placeholders[:places],
                "sizes": sizes}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
        json.dump(forecast, f, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
