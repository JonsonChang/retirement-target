# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""退休金目標計算：確定性公式 + 蒙地卡羅。全部以「今天的購買力」（實質）計算。

用法：
    uv run calc.py input.json [--report retirement_report_YYYYMMDD.md]

輸入 JSON 格式見 ../reference/input_schema.md。結果以 JSON 輸出到 stdout。
"""
import argparse
import json
import math
import sys

import numpy as np

# 勞保老年年金：兩式擇優
LI_RATE_A, LI_BASE_A, LI_RATE_B = 0.00775, 3000, 0.0155
LI_ADJ_PER_YEAR, LI_ADJ_MAX_YEARS = 0.04, 5
LI_MIN_ANNUITY_YEARS, LI_LUMP_MAX_MONTHS = 15, 45
LP_MIN_CLAIM_AGE = 60


# ---------- 基本參數 ----------

def real_return(a, infl):
    """實質幾何預期報酬。
    股票（有 gordon）用高登方程式：殖利率 + 實質股息成長率 − drag；
    其他用費雪方程式把名目報酬（債券、現金為目前殖利率）換成實質。"""
    g = a.get("gordon")
    if g:
        return g["div_yield"] + g["real_growth"] - a.get("drag", 0.0)
    return (1 + a["expected_return"] - a.get("drag", 0.0)) / (1 + infl) - 1


def nominal_to_real(nom, drag, infl):
    return None if nom is None else (1 + nom - drag) / (1 + infl) - 1


def asset_params(cfg):
    """回傳 symbols、實質對數平均 mu、對數共變異數 cov、目前市值 values。"""
    infl = cfg["inflation"]
    assets = cfg["assets"]
    syms = [a["symbol"] for a in assets]
    mu = np.array([math.log(1 + real_return(a, infl)) for a in assets])
    vol = np.array([a["vol"] for a in assets])
    corr = np.eye(len(syms))
    c = cfg.get("correlation")
    if c:
        idx = {s: i for i, s in enumerate(c["symbols"])}
        for i, si in enumerate(syms):
            for j, sj in enumerate(syms):
                if i != j and si in idx and sj in idx:
                    corr[i, j] = c["matrix"][idx[si]][idx[sj]]
    cov = np.outer(vol, vol) * corr
    # 修正成半正定（缺漏或成對計算的相關係數可能不一致）
    w, v = np.linalg.eigh((cov + cov.T) / 2)
    cov = (v * np.clip(w, 0, None)) @ v.T
    values = np.array([float(a.get("value", 0.0)) for a in assets])
    return syms, mu, cov, values


def weights_from(alloc, syms):
    w = np.array([float(alloc.get(s, 0.0)) for s in syms])
    missing = set(alloc) - set(syms)
    if missing:
        raise ValueError(f"retirement_allocation 含有 assets 裡沒有的標的: {missing}")
    return w / w.sum()


def portfolio_return_detail(w, mu, cov):
    """再平衡投資組合的報酬拆解：加權算術報酬 E、組合波動率、波動拖累、實質幾何報酬（已扣拖累）。"""
    arith = np.exp(mu + np.diag(cov) / 2)
    e = float(w @ arith)
    var = float(w @ cov @ w)
    return {"arith": e - 1, "vol": math.sqrt(var), "drag": var / 2, "real": math.exp(math.log(e) - var / 2) - 1}


# ---------- 勞保、勞退、其他收入（實質年額） ----------

def labor_insurance_detail(li):
    s, y = li["avg_insured_salary"], li["years"]
    mode = li.get("mode", "monthly")
    if mode == "lump_sum":
        if y >= LI_MIN_ANNUITY_YEARS and not li.get("pre2009_years", False):
            raise ValueError("勞保：98 年後才初次加保且年資滿 15 年者只能月領，不能選 lump_sum")
        # 前 15 年每滿 1 年 1 個月，超過部分每滿 1 年 2 個月，最高 45 個月（年資未滿 15 年即老年一次金，同為每年 1 個月）
        months = min(LI_LUMP_MAX_MONTHS, min(y, 15) + 2 * max(0.0, y - 15))
        return {"mode": mode, "salary": s, "years": y, "claim_age": li["claim_age"],
                "months_raw": min(y, 15) + 2 * max(0.0, y - 15), "months": months, "lump": s * months}
    if y < LI_MIN_ANNUITY_YEARS:
        raise ValueError("勞保：年資未滿 15 年只能一次請領（老年一次金），mode 請設 lump_sum")
    a, b = s * y * LI_RATE_A + LI_BASE_A, s * y * LI_RATE_B
    d = max(-LI_ADJ_MAX_YEARS, min(LI_ADJ_MAX_YEARS, li["claim_age"] - li["legal_age"]))
    f = 1 + LI_ADJ_PER_YEAR * d
    return {"mode": mode, "salary": s, "years": y, "claim_age": li["claim_age"], "legal_age": li["legal_age"],
            "formula_a": a, "formula_b": b, "adj_years": d, "adj_factor": f, "monthly": max(a, b) * f}


def labor_pension(lp, age, retire_age, plan_end_age, infl):
    """回傳請領年齡、逐年累積明細、請領時專戶餘額，以及攤成期初年金後的每年實質月退總額。"""
    rate_nom = lp.get("return_nominal", 0.03)
    rr = (1 + rate_nom) / (1 + infl) - 1
    claim = max(LP_MIN_CLAIM_AGE, retire_age, lp.get("claim_age", LP_MIN_CLAIM_AGE))
    bal = lp.get("balance", 0.0)
    yearly = 12 * lp.get("monthly_wage", 0.0) * lp.get("rate", 0.06)
    rows = []
    for a in range(age, claim):
        g, c = bal * rr, (yearly if a < retire_age else 0.0)
        rows.append({"age": a, "start": bal, "growth": g, "contrib": c, "end": bal + g + c})
        bal = bal + g + c
    out = {"mode": lp.get("mode", "monthly"), "claim_age": claim, "return_nominal": rate_nom, "real_return": rr,
           "monthly_wage": lp.get("monthly_wage", 0.0), "rate": lp.get("rate", 0.06), "yearly_contrib": yearly,
           "rows": rows, "balance_at_claim": bal}
    if out["mode"] == "lump_sum":
        return {**out, "lump": bal}
    n = max(1, plan_end_age - claim)
    pv_factor = n if abs(rr) < 1e-12 else (1 - (1 + rr) ** -n) / rr * (1 + rr)  # 期初年金
    return {**out, "years": n, "annuity_factor": pv_factor, "annual": bal / pv_factor}


def income_by_age(cfg, plan_end_age):
    """各年齡的實質年收入（勞保 + 勞退 + 其他）、一次領金額、摘要、各來源逐年金額與計算明細。
    一次領的金額在請領當年存入投資組合，所以不計入 inc，另放 lumps。"""
    age, ra, infl = cfg["age"], cfg["retire_age"], cfg["inflation"]
    detail, inc, parts, lumps, lump_parts, calc = {}, {}, {}, {}, {}, {}

    def add(name, a_from, amount, a_to=None):
        p = parts.setdefault(name, {})
        for a in range(a_from, a_to or 200):
            inc[a] = inc.get(a, 0.0) + amount
            p[a] = p.get(a, 0.0) + amount

    def add_lump(name, a, amount):
        if a < ra:
            raise ValueError(f"{name}一次領的請領年齡 {a} 早於退休年齡 {ra}")
        lumps[a] = lumps.get(a, 0.0) + amount
        lump_parts[name] = {a: amount}

    li = cfg.get("labor_insurance")
    if li:
        x = labor_insurance_detail(li)
        if x["mode"] == "lump_sum":
            add_lump("勞保一次領", li["claim_age"], x["lump"])
            detail["labor_insurance"] = {"mode": "lump_sum", "claim_age": li["claim_age"], "lump": x["lump"]}
        else:
            add("勞保", li["claim_age"], 12 * x["monthly"])
            detail["labor_insurance"] = {"mode": "monthly", "claim_age": li["claim_age"], "monthly": x["monthly"]}
        calc["labor_insurance"] = x
    lp = cfg.get("labor_pension")
    if lp:
        x = labor_pension(lp, age, ra, plan_end_age, infl)
        if x["mode"] == "lump_sum":
            add_lump("勞退一次領", x["claim_age"], x["lump"])
            detail["labor_pension"] = {"mode": "lump_sum", "claim_age": x["claim_age"], "lump": x["lump"]}
        else:
            add("勞退", x["claim_age"], x["annual"], plan_end_age)
            detail["labor_pension"] = {"mode": "monthly", "claim_age": x["claim_age"],
                                       "balance_at_claim": x["balance_at_claim"], "monthly": x["annual"] / 12}
        calc["labor_pension"] = x
    for o in cfg.get("other_income", []):
        add(o.get("name", "其他收入"), o["start_age"], o["annual"], o.get("end_age"))
    return inc, lumps, detail, parts, lump_parts, calc


# ---------- 確定性計算 ----------

def needs_vector(cfg, start_age, end_age, inc, lumps=None):
    """每年須由投資組合提領的金額；一次領的金額當年存入，所以 C_t 可能為負。"""
    exp, lumps = 12 * cfg["monthly_expense"], lumps or {}
    return np.array([max(0.0, exp - inc.get(a, 0.0)) - lumps.get(a, 0.0) for a in range(start_age, end_age)])


def pv_due(needs, r):
    """期初提領的現值（退休當天需要的金額）。"""
    return float(sum(n / (1 + r) ** k for k, n in enumerate(needs)))


def project_rows(a0, monthly, r, start_age, years):
    """退休前逐年累積：年末餘額 = 年初 × (1 + r) + 12 × 每月投入。"""
    rows, a = [], a0
    for k in range(years):
        g = a * r
        rows.append({"age": start_age + k, "start": a, "growth": g, "contrib": 12 * monthly, "end": a + g + 12 * monthly})
        a = rows[-1]["end"]
    return rows


def drawdown_rows(a, needs, r, start_age):
    """退休後逐年：年初提領 C_t，剩餘部分成長；耗盡後停止。"""
    rows = []
    for k, n in enumerate(needs):
        after = a - n
        end = max(0.0, after) * (1 + r)
        rows.append({"age": start_age + k, "start": a, "withdraw": n, "growth": end - max(0.0, after), "end": end})
        if after < 0:
            break
        a = end
    return rows


def crossover_rows(a0, monthly, r_acc, r_ret, age, ra, end_age, inc, lumps, expense):
    """跨越點：假設持續工作、每月投入、不提領，逐年比較年初的被動收入
    （投資資產 × 實質報酬 + 勞保等固定收入）與年生活費。一次領的金額在請領當年年初存入。"""
    rows, a = [], a0
    for x in range(age, end_age + 1):
        a += lumps.get(x, 0.0)
        r = r_acc if x < ra else r_ret
        rows.append({"age": x, "assets": a, "rate": r, "invest": a * r, "fixed": inc.get(x, 0.0), "expense": expense})
        a = a * (1 + r) + 12 * monthly
    return rows


def depletion_age(a, needs, r, start_age):
    for k, n in enumerate(needs):
        a -= n
        if a < 0:
            return start_age + k
        a *= 1 + r
    return None


# ---------- 蒙地卡羅 ----------

def simulate(mu, cov, w_acc, w_ret, a0, monthly, needs, n_acc, z, track=False):
    """z: (paths, n_acc + len(needs), K) 標準常態亂數。
    回傳 (退休時資產, 耗盡的期數（未耗盡為 -1）, 各年初資產的 P10/P50/P90 清單（track=True 時）,
    退休後各年投資組合毛報酬 (paths, 年數))。"""
    k = len(mu)
    L = np.linalg.cholesky(cov + np.eye(k) * 1e-12)
    gross = np.exp(mu + z @ L.T)  # 各資產實質毛報酬
    g_acc = gross[:, :n_acc, :] @ w_acc
    g_ret = gross[:, n_acc:, :] @ w_ret
    a = np.full(z.shape[0], a0, dtype=float)
    pcts = [np.percentile(a, [10, 50, 90])] if track else None
    for t in range(n_acc):
        a = a * g_acc[:, t] + 12 * monthly
        if track:
            pcts.append(np.percentile(a, [10, 50, 90]))
    a_ret = a.copy()
    dep = np.full(z.shape[0], -1)
    for t, n in enumerate(needs):
        a = a - n
        newly = (a < 0) & (dep < 0)
        dep[newly] = t
        a = np.where(a < 0, 0.0, a) * g_ret[:, t]
        if track:
            pcts.append(np.percentile(a, [10, 50, 90]))
    return a_ret, dep, pcts, g_ret


def success_rate(dep, horizon):
    return float(np.mean((dep < 0) | (dep >= horizon)))


# ---------- 主程式 ----------

def run(cfg):
    age, ra = cfg["age"], cfg["retire_age"]
    plan_end = math.ceil(age + cfg["remaining_life_years"] + cfg.get("longevity_buffer_years", 5))
    stress_end = max(plan_end, cfg.get("stress_end_age", 95))
    n_acc = max(0, ra - age)
    monthly = cfg.get("monthly_contribution", 0.0)

    syms, mu, cov, values = asset_params(cfg)
    a0 = float(values.sum())
    w_ret = weights_from(cfg["retirement_allocation"], syms) if cfg.get("retirement_allocation") else (
        values / a0 if a0 > 0 else None)
    w_acc = values / a0 if a0 > 0 else w_ret
    if w_acc is None:
        raise ValueError("沒有現有資產時必須提供 retirement_allocation")
    port_acc = portfolio_return_detail(w_acc, mu, cov)
    port_ret = portfolio_return_detail(w_ret, mu, cov)
    r_acc, r_ret = port_acc["real"], port_ret["real"]

    inc, lumps, inc_detail, inc_parts, lump_parts, inc_calc = income_by_age(cfg, plan_end)
    needs_plan = needs_vector(cfg, ra, plan_end, inc, lumps)
    needs_stress = needs_vector(cfg, ra, stress_end, inc, lumps)
    no_income = needs_vector(cfg, ra, plan_end, {})

    target_raw = pv_due(needs_plan, r_ret)
    target = max(0.0, target_raw)  # 固定收入與一次領已足夠時目標為 0
    target_no_pension = pv_due(no_income, r_ret)
    acc_rows = project_rows(a0, monthly, r_acc, age, n_acc)
    projected = acc_rows[-1]["end"] if acc_rows else a0
    gap = target - projected
    fv_factor = n_acc if abs(r_acc) < 1e-12 else ((1 + r_acc) ** n_acc - 1) / r_acc
    extra_det = (max(0.0, gap) / (12 * fv_factor)) if fv_factor > 0 else (None if gap > 0 else 0.0)

    mc = cfg.get("mc", {})
    paths, target_sr = mc.get("paths", 10000), mc.get("target_success", 0.85)
    rng = np.random.default_rng(mc.get("seed", 42))
    z = rng.standard_normal((paths, n_acc + len(needs_stress), len(syms)))
    T_plan = len(needs_plan)

    def mc_at(m, track=False):
        return simulate(mu, cov, w_acc, w_ret, a0, m, needs_stress, n_acc, z, track)

    a_ret, dep, mc_pcts, g_ret = mc_at(monthly, track=True)
    sr_plan, sr_95 = success_rate(dep, T_plan), success_rate(dep, len(needs_stress))

    # 二分搜尋：成功率達標所需的每月投入（共用同一組亂數，結果單調）
    extra_mc = 0.0
    if sr_plan < target_sr:
        if n_acc == 0:
            extra_mc = None
        else:
            lo, hi = monthly, max(monthly, 1000.0) * 2
            while success_rate(mc_at(hi)[1], T_plan) < target_sr and hi < 1e8:
                lo, hi = hi, hi * 2
            for _ in range(30):
                mid = (lo + hi) / 2
                if success_rate(mc_at(mid)[1], T_plan) >= target_sr:
                    hi = mid
                else:
                    lo = mid
            extra_mc = hi - monthly

    cross = crossover_rows(a0, monthly, r_acc, r_ret, age, ra, stress_end, inc, lumps, 12 * cfg["monthly_expense"])
    cross_age = next((x["age"] for x in cross if x["invest"] + x["fixed"] >= x["expense"]), None)

    dep_ages = ra + dep[(dep >= 0) & (dep < T_plan)]
    pct = lambda x: {f"P{p}": float(np.percentile(x, p)) for p in (10, 50, 90)}
    return {
        "inputs_echo": {"age": age, "retire_age": ra, "plan_end_age": plan_end, "stress_end_age": stress_end,
                        "years_to_retire": n_acc, "retirement_years": T_plan, "inflation": cfg["inflation"]},
        "returns": {"accumulation_real": r_acc, "retirement_real": r_ret,
                    "assets": [{"symbol": s, "real_geometric": math.exp(m) - 1, "vol": math.sqrt(cov[i, i]),
                                "weight_now": float(w_acc[i]), "weight_retire": float(w_ret[i])}
                               for i, (s, m) in enumerate(zip(syms, mu))]},
        "income": inc_detail,
        "deterministic": {
            "target": target, "target_without_pension": target_no_pension,
            "current_assets": a0, "projected_at_retirement": projected, "gap": gap,
            "extra_monthly_needed": extra_det,
            "depletion_age": depletion_age(projected, needs_stress, r_ret, ra),
            "first_year_need": float(needs_plan[0]) if T_plan else 0.0,
            "crossover_age": cross_age,
        },
        "monte_carlo": {
            "paths": paths, "success_plan": sr_plan, "success_to_95": sr_95,
            "prob_reach_target": float(np.mean(a_ret >= target)),
            "assets_at_retirement": pct(a_ret),
            "failed_depletion_age_median": float(np.median(dep_ages)) if len(dep_ages) else None,
            "target_success": target_sr, "extra_monthly_needed": extra_mc,
        },
        # 報告用的計算明細；main() 不會印到 stdout
        "_calc": {
            "life_years": cfg["remaining_life_years"], "buffer_years": cfg.get("longevity_buffer_years", 5),
            "assets": [{"symbol": a["symbol"], "nominal": a.get("expected_return"), "drag": a.get("drag", 0.0),
                        "gordon": a.get("gordon"),
                        "ltcma_real": nominal_to_real(a.get("ltcma_return"), a.get("drag", 0.0), cfg["inflation"]),
                        "hist_real": nominal_to_real(a.get("hist_return"), 0.0, cfg["inflation"])}
                       for a in cfg["assets"]],
            "target_raw": target_raw, "port_acc": port_acc, "port_ret": port_ret, "income": inc_calc,
            "target_rows": [{"age": ra + t, "t": t, "expense": 12 * cfg["monthly_expense"],
                             "parts": {k: v.get(ra + t, 0.0) for k, v in inc_parts.items()},
                             "lumps": {k: v.get(ra + t, 0.0) for k, v in lump_parts.items()},
                             "need": float(n), "df": (1 + r_ret) ** t, "pv": float(n) / (1 + r_ret) ** t}
                            for t, n in enumerate(needs_plan)],
            "acc_rows": acc_rows, "fv_factor": fv_factor, "cross_rows": cross,
            "draw_rows": drawdown_rows(projected, needs_stress, r_ret, ra),
            "mc_pcts": [{"age": age + k, "P10": float(p[0]), "P50": float(p[1]), "P90": float(p[2])}
                        for k, p in enumerate(mc_pcts)],
            "mc_analysis": mc_analysis(a_ret, dep, g_ret, target, ra, T_plan, len(needs_stress)),
        },
    }


def mc_analysis(a_ret, dep, g_ret, target, ra, T_plan, T_stress):
    """把蒙地卡羅結果整理成報告「數據解讀」需要的統計。"""
    ok_plan = (dep < 0) | (dep >= T_plan)
    reach = a_ret >= target
    # 撐到某年齡 = 該年齡之前每年的提領都付得出
    survival = [{"age": ra + h, "rate": success_rate(dep, h)} for h in range(T_stress + 1)]
    fail = dep[(dep >= 0) & (dep < T_stress)] + ra
    # 報酬順序風險：依退休後前 n 年的年化報酬把路徑分四組
    n = min(5, g_ret.shape[1])
    seq = []
    if n:
        ann = np.exp(np.log(g_ret[:, :n]).mean(axis=1)) - 1
        edges = np.percentile(ann, [0, 25, 50, 75, 100])
        names = ["最差 25%", "偏差 25%", "偏好 25%", "最好 25%"]
        for q in range(4):
            sel = (ann >= edges[q]) & ((ann <= edges[q + 1]) if q == 3 else (ann < edges[q + 1]))
            if sel.any():
                seq.append({"name": names[q], "lo": float(edges[q]), "hi": float(edges[q + 1]),
                            "success_plan": success_rate(dep[sel], T_plan),
                            "success_stress": success_rate(dep[sel], T_stress)})
    return {
        "ret_pcts": {p: float(np.percentile(a_ret, p)) for p in (5, 10, 25, 50, 75, 90)},
        "cross": {"reach_ok": float(np.mean(reach & ok_plan)), "reach_fail": float(np.mean(reach & ~ok_plan)),
                  "miss_ok": float(np.mean(~reach & ok_plan)), "miss_fail": float(np.mean(~reach & ~ok_plan))},
        "survival": survival,
        "fail_stress": {"rate": len(fail) / len(dep),
                        "pcts": {p: float(np.percentile(fail, p)) for p in (10, 50, 90)} if len(fail) else None},
        "seq_years": n, "seq": seq,
    }


def fmt(x):
    return "—" if x is None else f"{x:,.0f}"


def pc(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def pc2(x):
    return f"{x * 100:.2f}%"


CHART_PALETTE = ["#4E79A7", "#F28E2B", "#59A14F"]  # Mermaid 預設的第一色太淡，改用對比明顯的顏色


def chart(title, xs, series, unit="萬元", scale=1e4, palette=None):
    """Mermaid xychart；series 為 [("bar"|"line", 數值清單)]，數值 ÷ scale 後以 unit 顯示。
    palette 為依序對應各 series 的顏色清單，省略時用 CHART_PALETTE。"""
    if len(xs) < 2:
        return []
    L = ["```mermaid",
         '%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "' + ", ".join(palette or CHART_PALETTE) + '"}}}}%%',
         "xychart-beta", f'    title "{title}"', f"    x-axis [{', '.join(str(x) for x in xs)}]",
          f'    y-axis "{unit}"']
    for kind, vals in series:
        L.append(f"    {kind} [{', '.join(f'{v / scale:.1f}' for v in vals)}]")
    return L + ["```", ""]


def calc_sections(cfg, res):
    """報告中「計算過程」：每個關鍵數字的公式、代入值與逐年明細。"""
    e, d, r, c = res["inputs_echo"], res["deterministic"], res["returns"], res["_calc"]
    i, ra, age, end = e["inflation"], e["retire_age"], e["age"], e["plan_end_age"]
    L = ["## 計算過程", "",
         "### 1. 規劃期", "",
         f"- 規劃終點 = ⌈目前年齡 {age} + 平均餘命 {c['life_years']} + 長壽緩衝 {c['buffer_years']}⌉"
         f" = ⌈{age + c['life_years'] + c['buffer_years']:.2f}⌉ = **{end} 歲**",
         f"- 退休年數 $T$ = {end} − {ra} = **{e['retirement_years']} 年**（{ra}～{end - 1} 歲每年年初提領一次）",
         f"- 壓力測試另算到 {e['stress_end_age']} 歲", "",
         "### 2. 各資產實質報酬", ""]
    pairs = list(zip(c["assets"], r["assets"]))
    stocks = [(a, x) for a, x in pairs if a["gordon"]]
    others = [(a, x) for a, x in pairs if not a["gordon"]]
    if stocks:
        L += ["**股票：高登方程式**（伯恩斯坦《投資金律》的基本報酬率，假設本益比／股息比維持現狀，不計估值膨脹的投機報酬）", "",
              "$r_{real} = \\text{起始股息殖利率} + \\text{實質股息成長率} - \\text{drag}$", "",
              "| 標的 | 股息殖利率 | 實質股息成長率 | 拖累 drag | 代入 | 實質報酬 |", "|---|--:|--:|--:|---|--:|"]
        for a, x in stocks:
            g = a["gordon"]
            L.append(f"| {a['symbol']} | {pc2(g['div_yield'])} | {pc2(g['real_growth'])} | {pc2(a['drag'])} | "
                     f"{g['div_yield']:.4f} + {g['real_growth']:.4f} − {a['drag']:.4f} | {pc2(x['real_geometric'])} |")
        L.append("")
    if others:
        L += ["**債券與現金：目前殖利率**（以費雪方程式換成實質）", "",
              f"$r_{{real}} = \\dfrac{{1 + r_{{nom}} - \\text{{drag}}}}{{1 + i}} - 1$，通膨 $i$ = {pc(i)}", "",
              "| 標的 | 名目殖利率 | 拖累 drag | 代入 | 實質報酬 |", "|---|--:|--:|---|--:|"]
        for a, x in others:
            L.append(f"| {a['symbol']} | {pc2(a['nominal'])} | {pc2(a['drag'])} | "
                     f"(1 + {a['nominal']:.4f} − {a['drag']:.4f}) ÷ {1 + i:.2f} − 1 | {pc2(x['real_geometric'])} |")
        L.append("")
    if any(a["ltcma_real"] is not None or a["hist_real"] is not None for a, _ in pairs):
        opt = lambda v: "—" if v is None else pc2(v)
        L += ["**對照（不用於計算）**：J.P. Morgan LTCMA 已扣 drag；歷史報酬為近 10 年含息總報酬，"
              "其中包含估值變動的投機報酬", "",
              "| 標的 | 本試算 | J.P. Morgan LTCMA | 歷史 10 年 |", "|---|--:|--:|--:|"]
        for a, x in pairs:
            L.append(f"| {a['symbol']} | {pc2(x['real_geometric'])} | {opt(a['ltcma_real'])} | {opt(a['hist_real'])} |")
        L.append("")
    L.pop()
    L += ["", "### 3. 投資組合實質報酬率", "",
          "各資產先換成算術平均 $e^{\\mu_k + \\sigma_k^2/2} - 1$ 依權重加權得 $E$，再扣掉波動拖累：",
          "$r = \\exp\\left(\\ln(1+E) - \\sigma_p^2/2\\right) - 1$，$\\sigma_p^2 = w^\\top \\Sigma w$（含相關係數）", "",
          "| 階段 | 加權算術報酬 $E$ | 組合波動率 $\\sigma_p$ | 波動拖累 $\\sigma_p^2/2$ | 實質幾何報酬 $r$ |",
          "|---|--:|--:|--:|--:|"]
    for name, p in (("退休前（目前配置）", c["port_acc"]), ("退休後（退休配置）", c["port_ret"])):
        L.append(f"| {name} | {pc2(p['arith'])} | {pc2(p['vol'])} | {pc2(p['drag'])} | **{pc2(p['real'])}** |")
    L.append("")

    n = 4
    li = c["income"].get("labor_insurance")
    if li and li["mode"] == "lump_sum":
        y = li["years"]
        L += [f"### {n}. 勞保老年給付（一次請領，存入投資組合）", "",
              f"- 給付月數 = 前 15 年每年 1 個月 + 超過 15 年部分每年 2 個月 = {min(y, 15):g} + 2 × {max(0.0, y - 15):.2f}"
              f" = {li['months_raw']:.2f}，上限 45 個月 → **{li['months']:.2f} 個月**",
              f"- 一次領 = 平均月投保薪資 {fmt(li['salary'])} × {li['months']:.2f} = **{fmt(li['lump'])} 元**，"
              f"{li['claim_age']} 歲請領當年存入投資組合",
              "- 一次請領老年給付限 98 年前有勞保年資者；平均月投保薪資以退保前 3 年計；60 歲後年資未計", ""]
        n += 1
    elif li:
        pick = "A" if li["formula_a"] >= li["formula_b"] else "B"
        L += [f"### {n}. 勞保老年年金", "",
              f"- 公式 A：{fmt(li['salary'])} × {li['years']} 年 × 0.775% + 3,000 = {fmt(li['formula_a'])}",
              f"- 公式 B：{fmt(li['salary'])} × {li['years']} 年 × 1.55% = {fmt(li['formula_b'])}",
              f"- 取較高者（公式 {pick}）= {fmt(max(li['formula_a'], li['formula_b']))}",
              f"- {li['claim_age']} 歲請領、法定 {li['legal_age']} 歲，相差 {li['adj_years']:+d} 年"
              f" → × (1 + 4% × {li['adj_years']}) = × {li['adj_factor']:.2f}",
              f"- **月領 {fmt(li['monthly'])} 元**（年領 {fmt(12 * li['monthly'])}），"
              f"勞保年金隨 CPI 調整，視為實質金額，{li['claim_age']} 歲起終身", ""]
        n += 1
    lp = c["income"].get("labor_pension")
    if lp:
        L += [f"### {n}. 勞退個人專戶", "",
              f"- 實質報酬 = (1 + {pc(lp['return_nominal'])}) ÷ (1 + {pc(i)}) − 1 = {pc2(lp['real_return'])}",
              f"- 每年提繳 = 月提繳工資 {fmt(lp['monthly_wage'])} × {pc(lp['rate'])} × 12 = {fmt(lp['yearly_contrib'])}（只到退休前）",
              "- 年末餘額 = 年初 + 年初 × 實質報酬 + 提繳", "",
              "| 年齡 | 年初餘額 | 收益 | 提繳 | 年末餘額 |", "|--:|--:|--:|--:|--:|"]
        for x in lp["rows"]:
            L.append(f"| {x['age']} | {fmt(x['start'])} | {fmt(x['growth'])} | {fmt(x['contrib'])} | {fmt(x['end'])} |")
        L += ["", *chart("勞退專戶累積（萬元）", [x["age"] for x in lp["rows"]] + [lp["claim_age"]],
                         [("bar", [x["start"] for x in lp["rows"]] + [lp["balance_at_claim"]])]),
              f"- {lp['claim_age']} 歲請領時餘額 $B$ = **{fmt(lp['balance_at_claim'])}**"]
        if lp["mode"] == "lump_sum":
            L += [f"- 一次領：{lp['claim_age']} 歲把 {fmt(lp['lump'])} 元全數存入投資組合，依退休配置投資", ""]
        else:
            rr, nn = lp["real_return"], lp["years"]
            L += [
              f"- 把 $B$ 在 {lp['claim_age']}～{lp['claim_age'] + nn - 1} 歲共 {nn} 年攤成期初年金：",
              f"  年金因子 = $\\dfrac{{1-(1+r)^{{-n}}}}{{r}}(1+r)$ = (1 − {1 + rr:.4f}^−{nn}) ÷ {rr:.4f} × {1 + rr:.4f}"
              f" = {lp['annuity_factor']:.3f}",
              f"- 年領 = {fmt(lp['balance_at_claim'])} ÷ {lp['annuity_factor']:.3f} = {fmt(lp['annual'])}，"
              f"**月領 {fmt(lp['annual'] / 12)} 元**",
              f"- 簡化假設：只攤到規劃終點 {end} 歲，之後不計勞退收入", ""]
        n += 1

    rows = c["target_rows"]
    names = list(rows[0]["parts"]) if rows else []
    lnames = list(rows[0]["lumps"]) if rows else []
    cols = names + lnames
    L += [f"### {n}. 退休金目標", "",
          "每年須由投資組合提領 $C_t$ = max(0, 年生活費 − 月領收入) − 一次領金額"
          + "（負數代表當年存入投資組合）" * bool(lnames)
          + f"，以退休後實質報酬 $r$ = {pc2(r['retirement_real'])} 折現（期初提領，第 0 年不折現）：", "",
          "$$\\text{目標} = \\sum_{t=0}^{T-1} \\frac{C_t}{(1+r)^t}$$", "",
          f"年生活費 = {fmt(cfg['monthly_expense'])} × 12 = {fmt(12 * cfg['monthly_expense'])}", "",
          "| 年齡 | $t$ | 年生活費 | " + "".join(f"{k} | " for k in cols) + "$C_t$ | $(1+r)^t$ | 現值 |",
          "|--:|--:|--:|" + "--:|" * len(cols) + "--:|--:|--:|"]
    for x in rows:
        L.append(f"| {x['age']} | {x['t']} | {fmt(x['expense'])} | "
                 + "".join(f"{fmt(x['parts'][k])} | " for k in names)
                 + "".join(f"{fmt(x['lumps'][k])} | " for k in lnames)
                 + f"{fmt(x['need'])} | {x['df']:.3f} | {fmt(x['pv'])} |")
    L += [f"| **合計** | | | " + "| " * len(cols) + f"{fmt(sum(x['need'] for x in rows))} | | **{fmt(c['target_raw'])}** |", ""]
    if c["target_raw"] < 0:
        L += [f"- 現值合計為負，代表固定收入與一次領已足夠支應，目標取 max(0, 合計) = **0**", ""]
    # 年生活費拆成「固定收入」與「投資組合提領」：兩組長條重疊繪製，整根 = 年生活費，下段 = 固定收入，上段 = 缺口
    fixed = [min(x["expense"], sum(x["parts"].values())) for x in rows]
    segs = []  # 固定收入相同的連續年齡合併成一段
    for x, f in zip(rows, fixed):
        if segs and abs(segs[-1][2] - f) < 0.5:
            segs[-1][1] = x["age"]
        else:
            segs.append([x["age"], x["age"], f, x["expense"]])
    L += ["年生活費由誰支付：整根長條 = 年生活費，底部藍色 = **勞保、勞退等固定收入**，"
          "上方橘色 = **缺口，須由投資組合提領**：", "",
          *chart("年生活費 = 固定收入（下段）+ 投資組合提領（上段）（萬元）", [x["age"] for x in rows],
                 [("bar", [x["expense"] for x in rows]), ("bar", fixed)], palette=["#F28E2B", "#4E79A7"]),
          "| 年齡 | 年生活費 | 固定收入 | 投資組合提領（缺口） | 缺口佔生活費 |", "|---|--:|--:|--:|--:|"]
    L += [f"| {a}{'' if a == b else f'～{b}'} | {fmt(ex)} | {fmt(f)} | {fmt(ex - f)} | {pc((ex - f) / ex if ex else 0)} |"
          for a, b, f, ex in segs]
    lump_total = sum(sum(x["lumps"].values()) for x in rows)
    if lump_total:
        L += ["", f"- 一次領合計 {fmt(lump_total)} 元在請領當年存入投資組合，不算固定收入；"
              f"它可支付約 {lump_total / max(1.0, segs[0][3] - segs[0][2]):.1f} 年的退休初期缺口"]
    L += ["", "每年須由投資組合提領 $C_t$（長條，已扣除一次領存入）與其折現值（折線）：", "",
          *chart("每年提領 C_t 與現值（萬元）", [x["age"] for x in rows],
                 [("bar", [x["need"] for x in rows]), ("line", [x["pv"] for x in rows])]),
          f"- 不計勞保、勞退等固定收入時，每年都要提領 {fmt(12 * cfg['monthly_expense'])}，"
          f"目標為 {fmt(d['target_without_pension'])} 元", ""]
    n += 1

    acc = c["acc_rows"]
    L += [f"### {n}. 退休時預估資產", "",
          f"以退休前實質報酬 $r$ = {pc2(r['accumulation_real'])} 逐年複利，每月投入 "
          f"{fmt(cfg.get('monthly_contribution', 0))}（實質，隨通膨調整）在年底加入：", "",
          "$A_{k+1} = A_k (1 + r) + 12 \\times \\text{每月投入}$", ""]
    if acc:
        L += ["| 年齡 | 年初資產 | 報酬 | 投入 | 年末資產 |", "|--:|--:|--:|--:|--:|"]
        L += [f"| {x['age']} | {fmt(x['start'])} | {fmt(x['growth'])} | {fmt(x['contrib'])} | {fmt(x['end'])} |" for x in acc]
        L += ["", *chart("退休前資產累積（萬元）", [x["age"] for x in acc] + [ra],
                         [("bar", [x["start"] for x in acc] + [d["projected_at_retirement"]])])]
    L += [f"- {ra} 歲退休時資產 = **{fmt(d['projected_at_retirement'])}**", ""]
    n += 1

    gap = d["gap"]
    L += [f"### {n}. 缺口與每月需再多投入", "",
          f"- 缺口 = 目標 {fmt(d['target'])} − 退休時資產 {fmt(d['projected_at_retirement'])} = {fmt(gap)}"
          + ("（負數代表已達標）" if gap <= 0 else ""),
          f"- 每月需再多投入 = max(0, 缺口) ÷ (12 × 年金終值因子)，年金終值因子 = $\\dfrac{{(1+r)^n - 1}}{{r}}$"
          f" = {c['fv_factor']:.3f}（$n$ = {e['years_to_retire']} 年）",
          f"- = {fmt(max(0.0, gap))} ÷ (12 × {c['fv_factor']:.3f}) = **{fmt(d['extra_monthly_needed'])} 元／月**", ""]
    n += 1

    dr = c["draw_rows"]
    L += [f"### {n}. 退休後資產走勢（確定性）", "",
          f"每年年初先提領 $C_t$，剩餘部分以 $r$ = {pc2(r['retirement_real'])} 成長：$A_{{t+1}} = (A_t - C_t)(1+r)$"
          f"，算到 {e['stress_end_age']} 歲", "",
          "| 年齡 | 年初資產 | 提領 $C_t$（負數為存入） | 報酬 | 年末資產 |", "|--:|--:|--:|--:|--:|"]
    L += [f"| {x['age']} | {fmt(x['start'])} | {fmt(x['withdraw'])} | {fmt(x['growth'])} | {fmt(x['end'])} |" for x in dr]
    if lp and lp["mode"] == "monthly" and e["stress_end_age"] > end:
        L += ["", f"- {end} 歲起勞退已在規劃期內攤完、不再計入，提領金額因此改變"]
    L += ["", *chart("退休後年初資產（萬元）", [x["age"] for x in dr], [("line", [x["start"] for x in dr])]),
          "- " + (f"{d['depletion_age']} 歲資產耗盡" if d["depletion_age"] else f"{e['stress_end_age']} 歲前不會耗盡"), ""]
    return L


def asset_group(cls):
    if cls in ("tw_cash",):
        return "現金"
    if cls.startswith("us_treasury") or cls in ("us_aggregate_bond", "us_ig_corp", "us_high_yield"):
        return "債券"
    if cls in ("gold",):
        return "黃金"
    return "股票"


def portfolio_section(cfg, res):
    """報告中「退休後的投資組合」：配置、與目前配置的差異、風險報酬、怎麼運作。"""
    e, d, r = res["inputs_echo"], res["deterministic"], res["returns"]
    pa, pr = res["_calc"]["port_acc"], res["_calc"]["port_ret"]
    ra, A = e["retire_age"], d["projected_at_retirement"]
    cls = {a["symbol"]: a.get("asset_class", "") for a in cfg["assets"]}
    custom = bool(cfg.get("retirement_allocation"))
    L = ["## 退休後的投資組合", "",
         (f"{ra} 歲退休時，投資組合從目前配置改成下表的退休配置。" if custom
          else "未指定退休配置，退休後沿用目前配置。")
         + f"金額以確定性預估的退休時資產 {fmt(A)} 元計算（假設退休前每年再平衡維持目前比例）：", "",
         "| 標的 | 類別 | 目前權重 | 退休後權重 | 退休時金額 | 退休時調整（+買入／−賣出） |",
         "|---|---|--:|--:|--:|--:|"]
    groups = {}
    for a in r["assets"]:
        s, wn, wr = a["symbol"], a["weight_now"], a["weight_retire"]
        g = asset_group(cls[s])
        groups.setdefault(g, [0.0, 0.0])
        groups[g][0] += wn
        groups[g][1] += wr
        if wn == 0 and wr == 0:
            continue
        adj = A * (wr - wn)
        L.append(f"| {s} | {g} | {pc(wn)} | {pc(wr)} | {fmt(A * wr)} | "
                 f"{'—' if abs(adj) < 0.5 else ('+' if adj > 0 else '−') + fmt(abs(adj))} |")
    L += ["", "依大類彙總：", "", "| 大類 | 目前 | 退休後 |", "|---|--:|--:|"]
    L += [f"| {g} | {pc(v[0])} | {pc(v[1])} |" for g, v in groups.items()]

    def bad_year(p):  # 一年報酬的第 5 百分位（約 20 年一遇的壞年）
        return (1 + p["real"]) * math.exp(-1.645 * p["vol"]) - 1
    ba, br = bad_year(pa), bad_year(pr)
    L += ["", "風險與報酬（實質，已扣通膨與波動拖累）：", "",
          "| | 目前配置 | 退休配置 |", "|---|--:|--:|",
          f"| 長期年化報酬 | {pc2(pa['real'])} | {pc2(pr['real'])} |",
          f"| 年波動率 | {pc2(pa['vol'])} | {pc2(pr['vol'])} |",
          f"| 約 20 年一遇的壞年（單年報酬第 5 百分位） | {pc(ba)} | {pc(br)} |",
          f"| 以退休時資產 {fmt(A)} 元計，壞年約虧損 | {fmt(-A * ba)} | {fmt(-A * br)} |", ""]
    if custom:
        dr, dv = pr["real"] - pa["real"], pr["vol"] - pa["vol"]
        L += [f"- **為什麼退休要換配置**：退休後每年都要從投資組合提領生活費，大跌時被迫低價賣出，"
              "虧損就再也賺不回來（報酬順序風險，見「蒙地卡羅模擬」第 4 節）。"
              f"退休配置的年波動率{'降低' if dv < 0 else '提高'} {pc2(abs(dv))}，"
              f"長期報酬{'少' if dr < 0 else '多'} {pc2(abs(dr))}，用一點報酬換取比較平穩的提領"]
    L += [f"- **怎麼運作**：{ra} 歲起每年年初先提領當年需要的生活費，剩下的依退休配置投資；"
          "每年再平衡一次，把漲多的賣一些、跌多的補一些，維持固定比例",
          "- **提領從哪裡賣**：本試算假設提領後整體再平衡，等於依比例賣出各資產；實務上可在再平衡時順便提領，"
          "優先賣出超過目標比例的資產，股市大跌的年份多從債券提領，讓股票有時間回升"]
    lumps = [(k, v) for k, v in res["income"].items() if v.get("mode") == "lump_sum"]
    names = {"labor_insurance": "勞保", "labor_pension": "勞退"}
    for k, v in lumps:
        L.append(f"- **{names[k]}一次領**：{v['claim_age']} 歲領到 {fmt(v['lump'])} 元，當年存入投資組合，同樣依退休配置投資")
    L.append("")
    return L


def crossover_section(cfg, res):
    """報告中「跨越點」：被動收入什麼時候超過生活費（財務自由門檻）。"""
    e, d = res["inputs_echo"], res["deterministic"]
    rows, ca, ra = res["_calc"]["cross_rows"], d["crossover_age"], e["retire_age"]
    exp_m = cfg["monthly_expense"]
    shown = [x for x in rows if x["age"] <= (min(ca + 3, e["stress_end_age"]) if ca else e["stress_end_age"])]
    by_age = {x["age"]: x for x in rows}
    L = ["## 跨越點（財務自由門檻）", "",
         "跨越點法則（Crossover Point）：**每月的被動收入 ≥ 每月生活支出**，"
         "就跨過財務自由的門檻，不必再靠工作收入。", "",
         "- **被動收入** = 投資資產 × 實質預期報酬 ÷ 12，加上已開始請領的勞保等固定月收入",
         "- 用**實質**報酬（已扣通膨）計算：只花掉這部分，本金的購買力不會縮水，可以一直花下去",
         f"- 假設一直工作到該年齡、每月持續投入 {fmt(cfg.get('monthly_contribution', 0))} 元、不提領；"
         f"{ra} 歲前用目前配置的報酬 {pc2(res['returns']['accumulation_real'])}，"
         f"{ra} 歲起用退休配置的報酬 {pc2(res['returns']['retirement_real'])}；勞保、勞退依原規劃的請領年齡計入",
         f"- 生活支出 = 退休後每月生活費 {fmt(exp_m)} 元（今天的物價）", "",
         *chart("跨越點：每月被動收入（長條）vs 每月生活費（線）（萬元／月）", [x["age"] for x in shown],
                [("bar", [(x["invest"] + x["fixed"]) / 12 for x in shown]), ("line", [x["expense"] / 12 for x in shown])],
                unit="萬元／月"),
         "| 年齡 | 年初投資資產 | 每月投資收益 | 每月固定收入 | 每月被動收入合計 | 佔生活費 |", "|--:|--:|--:|--:|--:|--:|"]
    for x in shown:
        if (x["age"] - e["age"]) % 5 == 0 or x["age"] in (ra, ca, shown[-1]["age"]):
            tot = (x["invest"] + x["fixed"]) / 12
            mark = "（預計退休）" if x["age"] == ra else "（跨越點）" if x["age"] == ca else ""
            L.append(f"| {x['age']}{mark} | {fmt(x['assets'])} | {fmt(x['invest'] / 12)} | {fmt(x['fixed'] / 12)} | "
                     f"{fmt(tot)} | {pc(tot / exp_m)} |")
    r = by_age.get(ra)
    L += [""]
    if ca:
        L.append(f"- **跨越點在 {ca} 歲**：從這一年起，只靠投資收益與固定收入就付得起每月 {fmt(exp_m)} 元的生活費")
    else:
        L.append(f"- 到 {e['stress_end_age']} 歲前都**不會**跨越：被動收入始終低於每月 {fmt(exp_m)} 元")
    if r:
        need = max(0.0, r["expense"] - r["fixed"]) / r["rate"] if r["rate"] > 0 else None
        L.append(f"- 預計 {ra} 歲退休時，被動收入約每月 {fmt((r['invest'] + r['fixed']) / 12)} 元，"
                 f"只佔生活費 {pc((r['invest'] + r['fixed']) / r['expense'])}；要在 {ra} 歲就跨越，"
                 f"投資資產需達 {fmt(need)} 元（年生活費扣掉固定收入 ÷ 實質報酬）")
    L += ["- **跟「退休金目標」的差別**：退休金目標允許退休後逐年花掉本金，算到規劃終點 "
          f"{e['plan_end_age']} 歲剛好用完；跨越點只花收益、不動本金，資產可以永續，所以門檻高得多、時間也晚得多",
          "- 介於兩者之間的做法：先到退休金目標就退休，但保留彈性（熊市少花、兼職收入），越接近跨越點越安全", ""]
    return L


def mc_section(res):
    """報告中「蒙地卡羅模擬」：結果數據與白話解讀。"""
    e, d, m = res["inputs_echo"], res["deterministic"], res["monte_carlo"]
    a, mp = res["_calc"]["mc_analysis"], res["_calc"]["mc_pcts"]
    ra, end, stress = e["retire_age"], e["plan_end_age"], e["stress_end_age"]
    sp, s95, tgt, cr = m["success_plan"], m["success_to_95"], d["target"], a["cross"]
    per100 = lambda x: f"{x * 100:.0f}"

    # 一句話結論
    concl = [f"- 每 100 個平行人生中，約 **{per100(sp)} 個**到規劃終點 {end} 歲前生活費都付得出，"
             f"約 **{per100(s95)} 個**能撐到 {stress} 歲"]
    if sp >= m["target_success"]:
        concl.append(f"- 規劃期成功率 {pc(sp)}，**已達**目標成功率 {pc(m['target_success'])}")
    else:
        concl.append(f"- 規劃期成功率 {pc(sp)}，**低於**目標成功率 {pc(m['target_success'])}；"
                     + (f"每月再多投入 {fmt(m['extra_monthly_needed'])} 元可達標" if m["extra_monthly_needed"] is not None
                        else "已退休無法再投入，需調降生活費或調整配置"))
    fail = cr["reach_fail"] + cr["miss_fail"]
    if fail > 0:
        post = cr["reach_fail"] >= cr["miss_fail"]
        concl.append(f"- 失敗的路徑中，{pc(cr['reach_fail'] / fail)} 在退休時其實已達標 → 主要風險在"
                     + ("**退休後**（報酬順序與長壽），不在退休前存不夠" if post
                        else "**退休前累積不足**，提高投入或延後退休最有效"))
    if len(a["seq"]) >= 2:
        lo, hi = a["seq"][0], a["seq"][-1]
        concl.append(f"- 退休後前 {a['seq_years']} 年報酬落在{lo['name']}時，活到 {stress} 歲的成功率只有 "
                     f"{pc(lo['success_stress'])}；落在{hi['name']}時為 {pc(hi['success_stress'])}")
    fs = a["fail_stress"]
    if fs["pcts"]:
        concl.append(f"- 一旦錢不夠用，資產耗盡年齡中位數是 **{fs['pcts'][50]:.0f} 歲**")

    L = ["## 蒙地卡羅模擬", "",
         "### 這個模擬在做什麼", "",
         f"- 用電腦隨機產生 **{m['paths']:,} 種**未來的市場走勢，可以想成 {m['paths']:,} 個平行人生，"
         "每個人生從現在一路走到 " + f"{stress} 歲",
         "- 每一年各資產的報酬隨機抽樣（多變量常態分布），抽樣時保留資產之間的連動（相關係數），"
         "例如美股大跌的年份，國際股通常也會一起跌",
         f"- 每年再平衡：退休前維持目前配置，{ra} 歲起改用退休配置；退休後每年初先提領生活費，剩下的繼續投資",
         "- **成功**＝規劃期內每年的生活費都付得出；資產在那之前歸零就算失敗",
         "- **P10／P50／P90**：把所有路徑由小排到大，P10 是第 10%（運氣差）、P50 是中間（一般）、P90 是第 90%（運氣好）", "",
         "### 重點解讀", "", *concl, "",
         "### 1. 退休時會有多少錢", "",
         f"退休金目標是 **{fmt(tgt)}** 元（確定性預估退休時資產 {fmt(d['projected_at_retirement'])} 元）。"
         f"{ra} 歲時各情境的資產：", "",
         "| 情境 | 百分位 | 退休時資產 | 與目標相比 |", "|---|---|--:|--:|"]
    for p, name in ((5, "極差"), (10, "運氣差"), (25, "偏差"), (50, "一般"), (75, "偏好"), (90, "運氣好")):
        v = a["ret_pcts"][p]
        diff = f"{'+' if v >= tgt else '−'}{fmt(abs(v - tgt))}" + (f"（{v / tgt:.2f} 倍）" if tgt > 0 else "")
        L.append(f"| {name} | P{p} | {fmt(v)} | {diff} |")
    reach = m["prob_reach_target"]
    L += ["", f"- 退休時資產達到目標的機率：**{pc(reach)}**（每 100 個人生中約 {per100(reach)} 個）",
          "- " + ("運氣差的 P10 仍高於目標，退休前的累積相當穩" if a["ret_pcts"][10] >= tgt else
                  "一般情況的 P50 高於目標，但運氣差時可能存不夠" if a["ret_pcts"][50] >= tgt else
                  "連一般情況的 P50 都低於目標，退休前的累積明顯不足"), "",
          "### 2. 錢能撐到幾歲", "",
          "某年齡的「還有錢的機率」＝到那個年齡之前，每年的生活費都付得出的機率：", "",
          "| 年齡 | 還有錢的機率 | 每 100 人中已用完的人數 |", "|--:|--:|--:|"]
    for x in a["survival"]:
        if (x["age"] - ra) % 5 == 0 or x["age"] in (end, stress):
            mark = "（規劃終點）" if x["age"] == end else "（壓力測試）" if x["age"] == stress else ""
            L.append(f"| {x['age']}{mark} | {pc(x['rate'])} | {per100(1 - x['rate'])} |")
    L += ["", *chart("各年齡還有錢的機率（%）", [x["age"] for x in a["survival"]],
                     [("line", [x["rate"] for x in a["survival"]])], unit="%", scale=0.01),
          "- 曲線越晚往下掉越好；規劃終點之後的下滑，代表活得比平均餘命久時的風險", "",
          "### 3. 退休時達標，不代表之後一定成功", "",
          f"把每條路徑依「{ra} 歲時有沒有達標」和「規劃期內有沒有成功」分成四類：", "",
          "| | 規劃期成功 | 規劃期失敗 |", "|---|--:|--:|",
          f"| 退休時達標 | {pc(cr['reach_ok'])} | {pc(cr['reach_fail'])} |",
          f"| 退休時未達標 | {pc(cr['miss_ok'])} | {pc(cr['miss_fail'])} |", "",
          f"- **達標卻失敗（{pc(cr['reach_fail'])}）**：目標是假設退休後每年都拿到平均報酬算出來的。"
          "實際上若退休初期遇到大跌，一邊賣資產付生活費一邊虧損，本金會掉得特別快",
          f"- **未達標卻成功（{pc(cr['miss_ok'])}）**：退休時錢不夠，但退休後行情好，仍然撐了過去",
          f"- 所以「退休時達標機率」{pc(reach)} 和「規劃期成功率」{pc(sp)} 不一樣，後者才代表整個退休期夠不夠用", ""]
    if a["seq"]:
        L += ["### 4. 報酬順序風險：退休初期的運氣最關鍵", "",
              f"把路徑依退休後前 {a['seq_years']} 年的年化實質報酬分成四組，看各組的成功率：", "",
              f"| 前 {a['seq_years']} 年報酬 | 年化實質報酬區間 | 規劃期成功率 | 活到 {stress} 歲成功率 |",
              "|---|--:|--:|--:|"]
        for s in a["seq"]:
            L.append(f"| {s['name']} | {pc(s['lo'])} ～ {pc(s['hi'])} | {pc(s['success_plan'])} | {pc(s['success_stress'])} |")
        L += ["", "- 長期平均報酬相同時，**下跌發生在退休初期**比發生在晚年傷害更大，因為那時本金最多、又同時在提領",
              "- 因應方式：退休前後保留 1～3 年生活費的現金或短債、熊市時暫時降低提領、或晚一點退休", ""]
    L += ["### 5. 失敗時通常在幾歲沒錢", ""]
    if fs["pcts"]:
        L += [f"- 算到 {stress} 歲，有 {pc(fs['rate'])} 的路徑會把錢用完",
              f"- 這些路徑的耗盡年齡：P10 {fs['pcts'][10]:.0f} 歲／中位數 {fs['pcts'][50]:.0f} 歲／P90 {fs['pcts'][90]:.0f} 歲",
              "- 規劃期內失敗路徑的耗盡年齡中位數：" + (f"{m['failed_depletion_age_median']:.0f} 歲，"
              f"比規劃終點 {end} 歲早 {end - m['failed_depletion_age_median']:.0f} 年" if m["failed_depletion_age_median"] else "—（規劃期內沒有失敗路徑）"),
              "- 失敗通常不是只差最後一兩年，而是提早好幾年沒錢，因此值得保留安全邊際", ""]
    else:
        L += [f"- 算到 {stress} 歲，所有路徑都沒有把錢用完", ""]
    L += ["### 6. 各年初資產分布", "",
          f"三條線由下而上為 P10、P50、P90；{ra} 歲起每年初先提領再投資。P10 線碰到 0，代表運氣差的 10% 在那個年齡已經沒錢：", "",
          *chart("蒙地卡羅：各年初資產百分位（萬元）", [x["age"] for x in mp],
                 [("line", [x[p] for x in mp]) for p in ("P10", "P50", "P90")]),
          "- 限制：報酬假設為常態分布，真實市場的極端事件（肥尾）可能更常發生，實際風險可能比模擬略高", ""]
    return L


def report(cfg, res):
    e, d, m, r = res["inputs_echo"], res["deterministic"], res["monte_carlo"], res["returns"]
    L = [f"# 退休金目標試算報告（{cfg.get('as_of', '')}）", "",
         "> 所有金額皆以**今天的購買力（實質新台幣）**表示。本報告僅供規劃參考，不構成投資建議。", "",
         "## 使用者基本資料", "", "| 項目 | 內容 |", "|---|---|"]
    profile = cfg.get("profile") or [
        {"item": "目前年齡", "value": f"{e['age']} 歲"},
        {"item": "預計退休年齡", "value": f"{e['retire_age']} 歲"},
        {"item": "退休後每月生活費", "value": f"{fmt(cfg['monthly_expense'])} 元"},
        {"item": "退休前每月投入", "value": f"{fmt(cfg.get('monthly_contribution', 0))} 元"},
        {"item": "目前資產", "value": "、".join(f"{a['symbol']} {fmt(a['value'])} 元" for a in cfg["assets"] if a.get("value"))}]
    L += [f"| {p['item']} | {p['value']} |" for p in profile]
    has_inc = any(cfg.get(k) for k in ("labor_insurance", "labor_pension", "other_income"))
    save = lambda x: ("已經退休，沒辦法再存，只能減少生活費或調整配置" if x is None
                      else "不用多存" if x <= 0 else f"每月多存 **{fmt(x)} 元**")
    L += ["",
         "## 結論", "",
         f"- **退休時要準備多少錢**：**{fmt(d['target'])} 元**。"
         + ("這筆錢加上勞保、勞退等每月固定收入，" if has_inc else "")
         + f"夠你 {e['retire_age']} 歲退休後，每月花 {fmt(cfg['monthly_expense'])} 元，一直花到 {e['plan_end_age']} 歲"
         f"（平均壽命再多抓 {cfg.get('longevity_buffer_years', 5)} 年，共 {e['retirement_years']} 年）",
         f"- **照現在的存法，退休時大約會有**：{fmt(d['projected_at_retirement'])} 元（假設每年都拿到平均報酬）",
         (f"- **還差多少**：**{fmt(d['gap'])} 元**" if d["gap"] > 0 else f"- **還差多少**：不差，還多出 {fmt(-d['gap'])} 元"),
         "- **每月要多存多少**：",
         f"  - 理想情況（每年報酬都剛好是平均值）：{save(d['extra_monthly_needed'])}",
         f"  - 考慮市場有漲有跌，想有 {m['target_success'] * 100:.0f}% 的把握錢夠用：{save(m['extra_monthly_needed'])}",
         f"- **錢夠用的機率**：電腦模擬 {m['paths']:,} 種未來的市場走勢，其中 **{pc(m['success_plan'])}** 的情況錢能撐到 "
         f"{e['plan_end_age']} 歲；如果活到 {e['stress_end_age']} 歲，機率是 {pc(m['success_to_95'])}",
         f"- **錢大約幾歲用完**（理想情況）：" + (f"{d['depletion_age']} 歲" if d["depletion_age"] else f"活到 {e['stress_end_age']} 歲都還有剩"),
         "- **跨越點（被動收入 ≥ 生活費）**：" + (f"{d['crossover_age']} 歲（假設一直工作、持續投入到那時）" if d["crossover_age"]
                                            else f"{e['stress_end_age']} 歲前不會跨越") + "，詳見「跨越點」一節",
         "", "> 「理想情況」假設每年報酬都一樣，真實市場不會這麼平順，所以建議以「有把握」那個數字為準。"
         "機率怎麼算出來的，請看下方「蒙地卡羅模擬」一節。", "",
         "## 輸入摘要", "",
         f"- 目前 {e['age']} 歲，{e['years_to_retire']} 年後退休",
         f"- 退休後每月生活費 {fmt(cfg['monthly_expense'])} 元，目前每月投入 {fmt(cfg.get('monthly_contribution', 0))} 元",
         f"- 目前資產合計 {fmt(d['current_assets'])} 元", "",
         "| 標的 | 資產類別 | 市值 | 目前權重 | 退休後權重 | 預期實質報酬（幾何） | 波動率 |",
         "|---|---|--:|--:|--:|--:|--:|"]
    cls = {a["symbol"]: a for a in cfg["assets"]}
    for a in r["assets"]:
        c = cls[a["symbol"]]
        L.append(f"| {a['symbol']} | {c.get('asset_class', '')} | {fmt(c.get('value', 0))} | {pc(a['weight_now'])} | "
                 f"{pc(a['weight_retire'])} | {pc(a['real_geometric'])} | {pc(a['vol'])} |")
    L += [""] + portfolio_section(cfg, res) + crossover_section(cfg, res) + calc_sections(cfg, res) + mc_section(res)
    L += ["## 資料來源", "", "| 項目 | 數值 | 來源 |", "|---|---|---|"]
    for s in cfg.get("sources", []):
        L.append(f"| {s['item']} | {s['value']} | {s['source']} |")
    L += ["", "## 假設與限制", ""] + [f"- {n}" for n in cfg.get("notes", [])] + [
        "- 未計所得稅與基金內扣費用（美國掛牌股票 ETF 已依殖利率扣除 30% 股利預扣稅拖累，見各資產實質報酬表的 drag 欄）",
        "- 股票預期報酬用高登方程式，假設估值維持現狀；若未來估值下修（負的投機報酬），實際報酬會更低",
        "- 未計匯率風險：海外資產以新台幣計價，視同本地資產",
        "- 報酬分布假設為對數常態，真實市場的極端事件（肥尾）可能更常發生", ""]
    return "\n".join(L)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--report")
    a = ap.parse_args()
    with open(a.input, encoding="utf-8") as f:
        cfg = json.load(f)
    res = run(cfg)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write(report(cfg, res))
    res.pop("_calc")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
