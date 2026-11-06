# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""退休金目標計算：確定性公式 + 蒙地卡羅。全部以「今天的購買力」（實質）計算。

用法：
    uv run calc.py input.json [--report retirement_report_YYYYMMDD_raw.md] [--json retirement_result_YYYYMMDD.json]

輸入 JSON 格式見 ../reference/input_schema.md。結果以 JSON 輸出到 stdout；--report 寫計算底稿，--json 寫結構化數據檔。
"""
import argparse
import copy
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
        p = lump_parts.setdefault(name, {})
        p[a] = p.get(a, 0.0) + amount

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
    for i, o in enumerate(cfg.get("other_income", [])):
        name = o.get("name", "其他收入")
        if o.get("lump"):
            # 一次性收入：當年存入投資組合；退休前領到的延到退休當年才存入（不計中間的報酬）
            a = max(o["age"], ra)
            add_lump(f"{name}一次領", a, o["amount"])
            detail[f"other_lump_{i}"] = {"mode": "lump_sum", "name": name, "claim_age": a, "lump": o["amount"]}
        else:
            add(name, o["start_age"], o["annual"], o.get("end_age"))
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


def four_pct_rule(cfg, inc, ra, end_age, cross, projected, lumps):
    """4% 法則（William Bengen）：目標 = 25 × (年生活費 − 年固定收入)。
    固定收入用所有月領收入都開始請領後的金額；請領前的過渡期缺口另外加上（實質、不折現）。
    reach_age：假設持續工作與投入，年初投資資產第一次 ≥ 「該年齡退休」門檻的年齡。"""
    exp = 12 * cfg["monthly_expense"]
    full_age = max([ra] + [a for a in inc if inc[a] > 0 and a < end_age and inc.get(a - 1, 0.0) < inc[a]])
    fixed = inc.get(full_age, 0.0)
    gap = max(0.0, exp - fixed)

    def bridge(x):
        return sum(max(0.0, exp - inc.get(a, 0.0)) - gap for a in range(x, full_age))

    reach = next((r["age"] for r in cross if r["age"] < end_age and r["assets"] >= 25 * gap + bridge(r["age"])), None)
    return {"annual_expense": exp, "fixed_income": fixed, "full_income_age": full_age, "annual_gap": gap,
            "base": 25 * gap, "bridge": bridge(ra), "target": 25 * gap + bridge(ra),
            "projected": projected, "lumps": sum(lumps.values()), "reach_age": reach,
            "reach_target": 25 * gap + bridge(reach) if reach is not None else None}


def liability_matching(needs, a_ret, years=(5, 10, 20, 25)):
    """負債配合投資組合：退休後前 N 年的提領缺口放獨立的安全池（定存／現金，不折現，假設實質報酬約 0），
    其餘為風險組合，風險組合照退休配置持有。"""
    rows = []
    for n in years:
        lmp = float(sum(max(0.0, x) for x in needs[:n]))
        rows.append({"years": n, "lmp": lmp, "share": lmp / a_ret if a_ret > 0 else None, "risk": a_ret - lmp})
    return {"rows": rows}


def cfg_retire_at(cfg, x):
    """改成 x 歲退休的輸入：勞保年資依提早／延後的年數增減（未滿 15 年改為一次領），一次領不早於退休。
    勞退的請領年齡由 labor_pension() 自動取 max(60, 退休年齡, claim_age)。"""
    c = copy.deepcopy(cfg)
    c["retire_age"] = x
    li = c.get("labor_insurance")
    if li:
        li["years"] = max(0.0, li["years"] - (cfg["retire_age"] - x))
        if li["years"] < LI_MIN_ANNUITY_YEARS:
            li["mode"] = "lump_sum"
        if li.get("mode") == "lump_sum":
            li["claim_age"] = max(li["claim_age"], x)
    return c


def early_retire_scan(cfg, mu, cov, w_acc, w_ret, a0, r_acc, r_ret, plan_end, stress_end, z):
    """提早退休：逐一假設在 age … plan_end−1 歲退休（退休後停止投入），算退休時資產、目標、缺口、
    退休後每月最多可花多少（確定性、花到規劃終點剛好用完），以及退休年齡在原規劃 +5 年內的蒙地卡羅成功率。
    z 的期數 = stress_end − age，與退休年齡無關，所以各退休年齡共用同一組亂數。"""
    age, monthly = cfg["age"], cfg.get("monthly_contribution", 0.0)
    rows = []
    for x in range(age, plan_end):
        c = cfg_retire_at(cfg, x)
        inc, lumps, *_ = income_by_age(c, plan_end)
        n_acc = x - age
        acc = project_rows(a0, monthly, r_acc, age, n_acc)
        proj = acc[-1]["end"] if acc else a0
        target = max(0.0, pv_due(needs_vector(c, x, plan_end, inc, lumps), r_ret))
        lo, hi = 0.0, max(1e6, proj)  # 二分搜尋每月可花：目標現值 ≤ 退休時資產
        for _ in range(50):
            mid = (lo + hi) / 2
            if pv_due(needs_vector({**c, "monthly_expense": mid}, x, plan_end, inc, lumps), r_ret) <= proj:
                lo = mid
            else:
                hi = mid
        row = {"age": x, "projected": proj, "target": target, "gap": target - proj, "max_monthly": lo}
        if x <= cfg["retire_age"] + 5:
            ns = needs_vector(c, x, stress_end, inc, lumps)
            dep = simulate(mu, cov, w_acc, w_ret, a0, monthly, ns, n_acc, z)[1]
            row.update(success_plan=success_rate(dep, plan_end - x), success_stress=success_rate(dep, len(ns)))
        rows.append(row)
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


def simulate_two_pool(mu, cov, w_ret, w_safe, a_ret, needs, z_ret, n_years, gap=None):
    """退休後兩個池子分開運作（水桶策略）。退休時把前 n_years 年的提領缺口放獨立的安全池（權重 w_safe，
    通常是定存／現金；None 或全 0 表示實質報酬 0），其餘放風險池，風險池照退休配置 w_ret 每年再平衡。每年年初：
    1. 前一年風險池上漲時，賣風險池把安全池補到「未來 n_years 年的缺口」；下跌時不賣
    2. 生活費從安全池付（一次領的金額存入安全池）；安全池不夠才賣風險池
    3. 兩池各自成長
    gap 為未扣一次領的每年缺口，用來算安全池目標（省略時用 needs）；一次領照 needs 存入安全池。
    回傳耗盡的期數（未耗盡為 -1）。"""
    k = len(mu)
    L = np.linalg.cholesky(cov + np.eye(k) * 1e-12)
    gross = np.exp(mu + z_ret @ L.T)
    g_safe = gross @ (w_safe / w_safe.sum()) if w_safe is not None and w_safe.sum() > 0 else np.ones(gross.shape[:2])
    g_risk = gross @ w_ret
    pos = np.maximum(needs if gap is None else gap, 0.0)
    tgt = [float(pos[t:t + n_years].sum()) for t in range(len(needs))]
    safe = np.minimum(a_ret, tgt[0] if tgt else 0.0)
    risk = a_ret - safe
    dep = np.full(len(a_ret), -1)
    for t, n in enumerate(needs):
        if t > 0:
            move = np.where(g_risk[:, t - 1] > 1, np.minimum(np.maximum(tgt[t] - safe, 0.0), risk), 0.0)
            safe, risk = safe + move, risk - move
        safe = safe - n
        risk = risk - np.maximum(-safe, 0.0)
        safe = np.maximum(safe, 0.0)
        dep[(risk < 0) & (dep < 0)] = t
        risk = np.maximum(risk, 0.0)
        safe, risk = safe * g_safe[:, t], risk * g_risk[:, t]
    return dep


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
    fpr = four_pct_rule(cfg, inc, ra, plan_end, cross, projected, lumps)
    cls = {a["symbol"]: a.get("asset_class", "") for a in cfg["assets"]}
    gap_stress = needs_vector(cfg, ra, stress_end, inc)  # 未扣一次領：安全池目標用
    lmp = liability_matching(gap_stress, projected)
    # 兩池分開運作 vs 每年再平衡：同一組亂數；最差 25% = 退休後前 5 年年化報酬（再平衡組合）最低的四分之一
    n_seq = min(5, g_ret.shape[1])
    worst = (np.log(g_ret[:, :n_seq]).mean(axis=1) <= np.percentile(np.log(g_ret[:, :n_seq]).mean(axis=1), 25)
             if n_seq else np.ones(paths, bool))
    # 安全池獨立持有定存／現金（tw_cash）；沒有現金標的時視為實質報酬 0
    w_safe = np.array([1.0 if asset_group(cls.get(s, "")) == "現金" else 0.0 for s in syms])
    stock_w = float(sum(w for s, w in zip(syms, w_ret) if asset_group(cls.get(s, "")) == "股票"))
    pool_rows = [{"years": None, "dep": dep}] + [
        {"years": n, "dep": simulate_two_pool(mu, cov, w_ret, w_safe, a_ret, needs_stress, z[:, n_acc:, :], n, gap_stress)}
        for n in (5, 10, 20, 25)]
    pos_needs = np.maximum(gap_stress, 0.0)
    for x in pool_rows:
        dp = x.pop("dep")
        si = (None if x["years"] is None
              else min(1.0, float(pos_needs[:x["years"]].sum()) / projected) if projected > 0 else None)
        x.update(safe_init=si, stock_init=stock_w * (1 - (si or 0.0)),
                 success_plan=success_rate(dp, T_plan), success_stress=success_rate(dp, len(needs_stress)),
                 worst_plan=success_rate(dp[worst], T_plan), worst_stress=success_rate(dp[worst], len(needs_stress)))

    early = early_retire_scan(cfg, mu, cov, w_acc, w_ret, a0, r_acc, r_ret, plan_end, stress_end, z)
    first = lambda ok: next((x["age"] for x in early if ok(x)), None)
    early_ages = {"deterministic": first(lambda x: x["gap"] <= 0),
                  "mc_plan": first(lambda x: x.get("success_plan", 0) >= target_sr),
                  "mc_stress": first(lambda x: x.get("success_stress", 0) >= target_sr)}

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
        "four_percent_rule": {"target": fpr["target"], "annual_gap": fpr["annual_gap"], "bridge": fpr["bridge"],
                              "reached_at_retirement": projected + fpr["lumps"] >= fpr["target"],
                              "reach_age": fpr["reach_age"], "reach_target": fpr["reach_target"]},
        "liability_matching": {f"{x['years']}y": x["lmp"] for x in lmp["rows"]},
        "two_pool_mc": [{"strategy": "rebalance" if x["years"] is None else f"two_pool_{x['years']}y",
                         "success_plan": x["success_plan"], "success_to_95": x["success_stress"]} for x in pool_rows],
        "early_retirement": {"earliest_deterministic": early_ages["deterministic"],
                             "earliest_mc_plan": early_ages["mc_plan"],
                             "earliest_mc_stress": early_ages["mc_stress"]},
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
            "acc_rows": acc_rows, "fv_factor": fv_factor, "cross_rows": cross, "early_rows": early,
            "four_pct": fpr, "lmp": lmp, "pool_rows": pool_rows,
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
    # 連續年齡超過 15 個時改用數值軸，由 Mermaid 自動挑刻度，避免標籤擠成一團
    if len(xs) > 15 and all(isinstance(x, int) for x in xs) and list(xs) == list(range(xs[0], xs[0] + len(xs))):
        x_axis = f'    x-axis "年齡" {xs[0]} --> {xs[-1]}'
    else:
        x_axis = f"    x-axis [{', '.join(str(x) for x in xs)}]"
    L = ["```mermaid",
         '%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "' + ", ".join(palette or CHART_PALETTE) + '"}}}}%%',
         "xychart-beta", f'    title "{title}"', x_axis,
          f'    y-axis "{unit}"']
    for kind, vals in series:
        L.append(f"    {kind} [{', '.join(f'{v / scale:.1f}' for v in vals)}]")
    return L + ["```", ""]


def calc_sections(cfg, res, charts):
    """報告中「計算過程」：每個關鍵數字的公式、代入值與逐年明細。"""
    e, d, r, c = res["inputs_echo"], res["deterministic"], res["returns"], res["_calc"]
    i, ra, age, end = e["inflation"], e["retire_age"], e["age"], e["plan_end_age"]
    L = ["## 10. 計算過程", "",
         "### 10.1 規劃期", "",
         f"- 規劃終點 = ⌈目前年齡 {age} + 平均餘命 {c['life_years']} + 長壽緩衝 {c['buffer_years']}⌉"
         f" = ⌈{age + c['life_years'] + c['buffer_years']:.2f}⌉ = **{end} 歲**",
         f"- 退休年數 $T$ = {end} − {ra} = **{e['retirement_years']} 年**（{ra}～{end - 1} 歲每年年初提領一次）",
         f"- 壓力測試另算到 {e['stress_end_age']} 歲", "",
         "### 10.2 各資產實質報酬", ""]
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
    L += ["", "### 10.3 投資組合實質報酬率", "",
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
        L += [f"### 10.{n} 勞保老年給付（一次請領，存入投資組合）", "",
              f"- 給付月數 = 前 15 年每年 1 個月 + 超過 15 年部分每年 2 個月 = {min(y, 15):g} + 2 × {max(0.0, y - 15):.2f}"
              f" = {li['months_raw']:.2f}，上限 45 個月 → **{li['months']:.2f} 個月**",
              f"- 一次領 = 平均月投保薪資 {fmt(li['salary'])} × {li['months']:.2f} = **{fmt(li['lump'])} 元**，"
              f"{li['claim_age']} 歲請領當年存入投資組合",
              "- 一次請領老年給付限 98 年前有勞保年資者；平均月投保薪資以退保前 3 年計；60 歲後年資未計", ""]
        n += 1
    elif li:
        pick = "A" if li["formula_a"] >= li["formula_b"] else "B"
        L += [f"### 10.{n} 勞保老年年金", "",
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
        L += [f"### 10.{n} 勞退個人專戶", "",
              f"- 實質報酬 = (1 + {pc(lp['return_nominal'])}) ÷ (1 + {pc(i)}) − 1 = {pc2(lp['real_return'])}",
              f"- 每年提繳 = 月提繳工資 {fmt(lp['monthly_wage'])} × {pc(lp['rate'])} × 12 = {fmt(lp['yearly_contrib'])}（只到退休前）",
              "- 年末餘額 = 年初 + 年初 × 實質報酬 + 提繳", "",
              "| 年齡 | 年初餘額 | 收益 | 提繳 | 年末餘額 |", "|--:|--:|--:|--:|--:|"]
        for x in lp["rows"]:
            L.append(f"| {x['age']} | {fmt(x['start'])} | {fmt(x['growth'])} | {fmt(x['contrib'])} | {fmt(x['end'])} |")
        L += ["", *charts.add("pension", "勞退專戶累積（萬元）", [x["age"] for x in lp["rows"]] + [lp["claim_age"]],
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
    L += [f"### 10.{n} 退休金目標", "",
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
    L += ["每年須由投資組合提領 $C_t$（長條，已扣除一次領存入）與其折現值（折線）；年生活費與固定收入的對照見第 3 節：", "",
          *charts.add("need_pv", "每年提領 C_t（藍柱）與現值（橘線）（萬元）", [x["age"] for x in rows],
                 [("bar", [x["need"] for x in rows]), ("line", [x["pv"] for x in rows])]),
          f"- 不計勞保、勞退等固定收入時，每年都要提領 {fmt(12 * cfg['monthly_expense'])}，"
          f"目標為 {fmt(d['target_without_pension'])} 元", ""]
    n += 1

    acc = c["acc_rows"]
    L += [f"### 10.{n} 退休時預估資產", "",
          f"以退休前實質報酬 $r$ = {pc2(r['accumulation_real'])} 逐年複利，每月投入 "
          f"{fmt(cfg.get('monthly_contribution', 0))}（實質，隨通膨調整）在年底加入：", "",
          "$A_{k+1} = A_k (1 + r) + 12 \\times \\text{每月投入}$", ""]
    if acc:
        L += ["| 年齡 | 年初資產 | 報酬 | 投入 | 年末資產 |", "|--:|--:|--:|--:|--:|"]
        L += [f"| {x['age']} | {fmt(x['start'])} | {fmt(x['growth'])} | {fmt(x['contrib'])} | {fmt(x['end'])} |" for x in acc]
        L += ["", *charts.add("accumulation", "退休前資產累積（萬元）", [x["age"] for x in acc] + [ra],
                         [("bar", [x["start"] for x in acc] + [d["projected_at_retirement"]])])]
    L += [f"- {ra} 歲退休時資產 = **{fmt(d['projected_at_retirement'])}**", ""]
    n += 1

    gap = d["gap"]
    L += [f"### 10.{n} 缺口與每月需再多投入", "",
          f"- 缺口 = 目標 {fmt(d['target'])} − 退休時資產 {fmt(d['projected_at_retirement'])} = {fmt(gap)}"
          + ("（負數代表已達標）" if gap <= 0 else ""),
          f"- 每月需再多投入 = max(0, 缺口) ÷ (12 × 年金終值因子)，年金終值因子 = $\\dfrac{{(1+r)^n - 1}}{{r}}$"
          f" = {c['fv_factor']:.3f}（$n$ = {e['years_to_retire']} 年）",
          f"- = {fmt(max(0.0, gap))} ÷ (12 × {c['fv_factor']:.3f}) = **{fmt(d['extra_monthly_needed'])} 元／月**", ""]
    n += 1

    dr = c["draw_rows"]
    L += [f"### 10.{n} 退休後資產走勢（確定性）", "",
          f"每年年初先提領 $C_t$，剩餘部分以 $r$ = {pc2(r['retirement_real'])} 成長：$A_{{t+1}} = (A_t - C_t)(1+r)$"
          f"，算到 {e['stress_end_age']} 歲", "",
          "| 年齡 | 年初資產 | 提領 $C_t$（負數為存入） | 報酬 | 年末資產 |", "|--:|--:|--:|--:|--:|"]
    L += [f"| {x['age']} | {fmt(x['start'])} | {fmt(x['withdraw'])} | {fmt(x['growth'])} | {fmt(x['end'])} |" for x in dr]
    if lp and lp["mode"] == "monthly" and e["stress_end_age"] > end and not (d["depletion_age"] and d["depletion_age"] <= end):
        L += ["", f"- {end} 歲起勞退已在規劃期內攤完、不再計入，提領金額因此改變"]
    L += ["", *charts.add("drawdown", "退休後年初資產（萬元）", [x["age"] for x in dr], [("line", [x["start"] for x in dr])]),
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


CLASS_NAMES = {
    "tw_equity": "台股市值型股票", "tw_single_stock": "台股個股", "us_large_cap": "美國大型股",
    "us_small_cap": "美國小型股", "global_equity": "全球股票", "dev_ex_us_equity": "美國以外國際股票",
    "em_equity": "新興市場股票", "us_treasury_short": "美國短期公債", "us_treasury_intermediate": "美國中期公債",
    "us_treasury_long": "美國長期公債", "us_aggregate_bond": "美國總體投資級債券", "us_ig_corp": "美國投資級公司債",
    "us_high_yield": "美國高收益債", "gold": "黃金", "tw_cash": "新台幣現金／定存",
}


def light(x):
    """綜合評估燈號：✅ ≥ 85%、🟡 70～85%、⚠️ 30～70%、❌ < 30%。"""
    return "—" if x is None else "✅" if x >= 0.85 else "🟡" if x >= 0.70 else "⚠️" if x >= 0.30 else "❌"


def gap_txt(g):
    """缺口 = 目標 − 資產；正數為不足、負數為餘裕。"""
    return "—" if g is None else f"➖ 不足 {fmt(g)}" if g > 0.5 else f"➕ 餘裕 {fmt(-g)}"


class Charts(dict):
    """收集報告中每張 Mermaid 圖的資料（另存進結構化數據檔），並回傳 Markdown。"""

    def add(self, key, title, xs, series, unit="萬元", scale=1e4, palette=None):
        if len(xs) < 2:
            return []
        self[key] = {"title": title, "unit": unit, "x": [str(x).strip('"') for x in xs],
                     "series": [{"type": k, "values": [round(v / scale, 1) for v in vals]} for k, vals in series]}
        return chart(title, xs, series, unit, scale, palette)


def summarize(name, res):
    d, m, er = res["deterministic"], res["monte_carlo"], res["early_retirement"]
    return {"name": name, "target": d["target"], "projected": d["projected_at_retirement"],
            "extra_det": d["extra_monthly_needed"], "extra_mc": m["extra_monthly_needed"],
            "success_plan": m["success_plan"], "success_stress": m["success_to_95"],
            "earliest": [er["earliest_deterministic"], er["earliest_mc_plan"], er["earliest_mc_stress"]]}


def sensitivity(cfg, res):
    """固定的情境敏感度：每個情境只改一個條件重跑。退休年齡 ±2 年取自各年齡退休試算（同一組亂數）。"""
    exp, mon, ra = cfg["monthly_expense"], cfg.get("monthly_contribution", 0.0), cfg["retire_age"]
    rows = [summarize("基準", res)]
    sc = [(f"生活費 −10%（每月 {fmt(exp * 0.9)}）", {"monthly_expense": exp * 0.9}),
          (f"生活費 +10%（每月 {fmt(exp * 1.1)}）", {"monthly_expense": exp * 1.1})]
    if mon > 0 and ra > cfg["age"]:
        sc += [("每月投入歸零", {"monthly_contribution": 0.0}),
               (f"每月投入 +50%（每月 {fmt(mon * 1.5)}）", {"monthly_contribution": mon * 1.5})]
    for name, kw in sc:
        rows.append(summarize(name, run({**cfg, **kw})))
    if any(a.get("gordon") for a in cfg["assets"]):
        c2 = copy.deepcopy(cfg)
        for a in c2["assets"]:
            if a.get("gordon"):
                a["gordon"]["real_growth"] -= 0.005
        rows.append(summarize("股票實質報酬 −0.5%（股息成長率 −0.5%）", run(c2)))
    early = {x["age"]: x for x in res["_calc"]["early_rows"]}
    for dx in (-2, 2):
        x = early.get(ra + dx)
        if x and "success_plan" in x:
            rows.append({"name": f"{ra + dx} 歲退休（{'提早' if dx < 0 else '延後'} 2 年）", "target": x["target"],
                         "projected": x["projected"], "extra_det": None, "extra_mc": None,
                         "success_plan": x["success_plan"], "success_stress": x["success_stress"], "earliest": None})
    return rows


def derive(cfg, res):
    """把 run() 的結果整理成報告各節與結構化數據檔共用的表格資料。"""
    e, d, m, c = res["inputs_echo"], res["deterministic"], res["monte_carlo"], res["_calc"]
    f, er = c["four_pct"], res["early_retirement"]
    ra, end = e["retire_age"], e["plan_end_age"]
    fv = c["fv_factor"]
    extra = lambda need: (max(0.0, need) / (12 * fv)) if fv > 0 else (None if need > 0 else 0.0)
    A = d["projected_at_retirement"]
    cross = {x["age"]: x for x in c["cross_rows"]}
    cr = cross.get(ra)
    cross_need = (max(0.0, cr["expense"] - cr["fixed"]) / cr["rate"]) if cr and cr["rate"] > 0 else None
    standards = [
        {"name": "確定性退休金目標", "need": d["target"], "have": A, "reach_age": er["earliest_deterministic"],
         "extra": d["extra_monthly_needed"], "principal": True, "volatility": False,
         "meaning": f"每年都拿到平均報酬時，花到 {end} 歲剛好用完"},
        {"name": f"蒙地卡羅 {pc(m['target_success'])} 撐到 {end} 歲", "need": None, "have": A,
         "reach_age": er["earliest_mc_plan"], "extra": m["extra_monthly_needed"], "principal": True, "volatility": True,
         "meaning": f"模擬的市場走勢中，至少 {pc(m['target_success'])} 錢夠用到 {end} 歲"},
        {"name": f"蒙地卡羅 {pc(m['target_success'])} 撐到 {e['stress_end_age']} 歲", "need": None, "have": A,
         "reach_age": er["earliest_mc_stress"], "extra": None, "principal": True, "volatility": True,
         "meaning": f"同上，但要撐到 {e['stress_end_age']} 歲（防長壽）"},
        {"name": "25 倍法則（4% 法則）", "need": f["target"], "have": A + f["lumps"], "reach_age": f["reach_age"],
         "extra": extra(f["target"] - A - f["lumps"]), "principal": True, "volatility": None,
         "meaning": "投資資產 ≥ 每年須自行負擔的生活費 × 25（美國歷史回測 30 年）"},
        {"name": "跨越點（只花收益）", "need": cross_need, "have": cr["assets"] if cr else None,
         "reach_age": d["crossover_age"], "extra": None, "principal": False, "volatility": False,
         "meaning": "投資收益 + 固定收入 ≥ 生活費，本金不動"},
    ]

    inc = res["income"]
    income = []
    for k, name in (("labor_insurance", "勞保"), ("labor_pension", "勞退")):
        v = inc.get(k)
        if v:
            lump = v["mode"] == "lump_sum"
            income.append({"name": name + ("一次領" if lump else "月領"), "type": "一次領" if lump else "月領",
                           "start": v["claim_age"], "end": None if k == "labor_insurance" or lump else end,
                           "monthly": None if lump else v["monthly"], "amount": v.get("lump") if lump else None})
    for o in cfg.get("other_income", []):
        if not o.get("lump"):
            income.append({"name": o.get("name", "其他收入"), "type": "每年收入", "start": o["start_age"],
                           "end": o.get("end_age"), "monthly": o["annual"] / 12, "amount": None})
    for k, v in inc.items():
        if k.startswith("other_lump_"):
            income.append({"name": v["name"] + "一次領", "type": "一次領", "start": v["claim_age"], "end": None,
                           "monthly": None, "amount": v["lump"]})
    segs = []  # 固定收入相同的連續年齡合併成一段
    for x in c["target_rows"]:
        fx = min(x["expense"], sum(x["parts"].values()))
        if segs and abs(segs[-1]["fixed"] - fx) < 0.5:
            segs[-1]["to"] = x["age"]
        else:
            segs.append({"from": x["age"], "to": x["age"], "fixed": fx, "expense": x["expense"],
                         "need": x["expense"] - fx})
    lump_total = f["lumps"]
    bridge = {"years": max(0, f["full_income_age"] - ra), "from": ra, "to": f["full_income_age"] - 1,
              "total": f["bridge"]} if f["full_income_age"] > ra else None

    cls = {a["symbol"]: a.get("asset_class", "") for a in cfg["assets"]}
    rebalance = [{"symbol": a["symbol"], "class": CLASS_NAMES.get(cls[a["symbol"]], cls[a["symbol"]]),
                  "group": asset_group(cls[a["symbol"]]), "w_now": a["weight_now"], "before": A * a["weight_now"],
                  "w_ret": a["weight_retire"], "after": A * a["weight_retire"],
                  "adjust": A * (a["weight_retire"] - a["weight_now"])}
                 for a in res["returns"]["assets"] if a["weight_now"] or a["weight_retire"]]

    reach = f["reach_age"]
    early = [{**x, "light": light(x["success_plan"]), "gap_txt": gap_txt(x["gap"]),
              "four_pct": reach is not None and x["age"] >= reach}
             for x in c["early_rows"] if "success_plan" in x]
    return {"standards": standards, "income": income, "segments": segs, "lump_total": lump_total,
            "lump_years": lump_total / f["annual_gap"] if f["annual_gap"] > 0 and lump_total else None,
            "bridge": bridge, "rebalance": rebalance, "early": early, "sensitivity": sensitivity(cfg, res),
            "cross_need_at_retire": cross_need, "charts": Charts()}


def yn(v):
    return "—" if v is None else "是" if v else "否"


def summary_section(cfg, res, rd):
    e, d, m, r = res["inputs_echo"], res["deterministic"], res["monte_carlo"], res["returns"]
    f, er, a = res["_calc"]["four_pct"], res["early_retirement"], res["_calc"]["mc_analysis"]
    ra, end, stress, ts = e["retire_age"], e["plan_end_age"], e["stress_end_age"], m["target_success"]
    pr = res["_calc"]["pool_rows"]
    best = max(pr[1:], key=lambda x: (x["success_stress"], x["success_plan"])) if len(pr) > 1 else None
    cr = a["cross"]
    fail = cr["reach_fail"] + cr["miss_fail"]
    age_or = lambda x, none: f"{x} 歲" if x is not None else none
    rows = [
        ("規劃期", f"{e['age']} 歲 → {ra} 歲退休 → 規劃終點 {end} 歲（退休 {e['retirement_years']} 年）；壓力測試到 {stress} 歲", "10.1"),
        ("實質報酬：退休前／退休後", f"{pc2(r['accumulation_real'])}／{pc2(r['retirement_real'])}", "1.3"),
        ("退休金目標（確定性；已扣固定收入" + ("與一次領" if rd["lump_total"] else "") + "，是自己要準備的淨額）",
         fmt(d["target"]), "10"),
        ("退休時預估資產（確定性）", fmt(d["projected_at_retirement"])
         + (f"；另有一次領合計 {fmt(rd['lump_total'])} 元在請領當年存入" if rd["lump_total"] else ""), "10"),
        ("缺口（目標 − 退休時資產）", gap_txt(d["gap"]), "10"),
        ("每月需再多投入：確定性／蒙地卡羅 " + pc(ts), f"{fmt(d['extra_monthly_needed'])}／{fmt(m['extra_monthly_needed'])}", "10、9"),
        (f"蒙地卡羅成功率：撐到 {end} 歲／{stress} 歲", f"{pc(m['success_plan'])}／{pc(m['success_to_95'])}", "9.2"),
        ("退休時資產（蒙地卡羅 P10／P50／P90）",
         "／".join(fmt(m["assets_at_retirement"][p]) for p in ("P10", "P50", "P90")), "9.1"),
        (f"最早退休：確定性／{pc(ts)} 撐到 {end} 歲／{pc(ts)} 撐到 {stress} 歲",
         "／".join(age_or(x, "—") for x in (er["earliest_deterministic"], er["earliest_mc_plan"], er["earliest_mc_stress"])), "5"),
        ("25 倍法則門檻（預計退休年齡）", f"{fmt(f['target'])}（年缺口 {fmt(f['annual_gap'])} × 25"
         + (f" + 過渡期缺口 {fmt(f['bridge'])}" if f["bridge"] else "") + "）", "8.1"),
        ("25 倍法則：預計退休時達標／達標年齡",
         f"{'已達標' if res['four_percent_rule']['reached_at_retirement'] else '未達標'}／"
         + (f"{f['reach_age']} 歲（該年齡門檻 {fmt(f['reach_target'])}）" if f["reach_age"] else "規劃期內達不到"), "8.1"),
        ("跨越點（被動收入 ≥ 生活費）", age_or(d["crossover_age"], f"{stress} 歲前不會跨越"), "7"),
        ("確定性資產耗盡年齡", age_or(d["depletion_age"], f"{stress} 歲前不會耗盡"), "10"),
        ("固定收入空窗期", (f"{rd['bridge']['from']}～{rd['bridge']['to']} 歲（{rd['bridge']['years']} 年），"
                       f"多出的缺口合計 {fmt(rd['bridge']['total'])}") if rd["bridge"] else "無", "3"),
        ("一次領可支付幾年缺口", f"約 {rd['lump_years']:.1f} 年（{fmt(rd['lump_total'])} ÷ 年缺口 {fmt(f['annual_gap'])}）"
         if rd["lump_years"] else "—", "3"),
        ("安全池金額：5 年／10 年", f"{fmt(res['liability_matching'].get('5y'))}／{fmt(res['liability_matching'].get('10y'))}", "8.2"),
        (f"撐到 {stress} 歲：每年再平衡／最佳兩池", f"{pc(pr[0]['success_stress'])}／"
         + (f"安全池 {best['years']} 年 {pc(best['success_stress'])}" if best else "—"), "8.3"),
    ]
    if a["seq"]:
        rows.append((f"退休後前 {a['seq_years']} 年報酬最差 25%：撐到 {stress} 歲", pc(a["seq"][0]["success_stress"]), "9.4"))
    if fail > 0:
        rows.append(("規劃期失敗路徑中，退休時其實已達標的比例",
                     f"{pc(cr['reach_fail'] / fail)}（主要風險在{'退休後' if cr['reach_fail'] >= cr['miss_fail'] else '退休前累積不足'}）", "9.3"))
    rd["summary"] = [{"item": k, "value": v, "section": s} for k, v, s in rows]
    L = ["## 0. 關鍵數字總表", "", "| 指標 | 數值 | 出處 |", "|---|---|--:|"]
    L += [f"| {k} | {v} | §{s} |" for k, v, s in rows]
    return L + [""]


def inputs_section(cfg, res):
    e, d, r = res["inputs_echo"], res["deterministic"], res["returns"]
    pa, pr = res["_calc"]["port_acc"], res["_calc"]["port_ret"]
    A = d["projected_at_retirement"]
    L = ["## 1. 輸入與假設", "", "### 1.1 使用者基本資料", "", "| 項目 | 內容 |", "|---|---|"]
    profile = cfg.get("profile") or [
        {"item": "目前年齡", "value": f"{e['age']} 歲"},
        {"item": "預計退休年齡", "value": f"{e['retire_age']} 歲"},
        {"item": "退休後每月生活費", "value": f"{fmt(cfg['monthly_expense'])} 元"},
        {"item": "退休前每月投入", "value": f"{fmt(cfg.get('monthly_contribution', 0))} 元"},
        {"item": "目前資產", "value": "、".join(f"{a['symbol']} {fmt(a['value'])} 元" for a in cfg["assets"] if a.get("value"))}]
    L += [f"| {p['item']} | {p['value']} |" for p in profile]
    L += ["", "### 1.2 資產與預期報酬", "",
          f"目前資產合計 {fmt(d['current_assets'])} 元；通膨 {pc(e['inflation'])}。算法見 §10.2", "",
          "| 標的 | 類別 | 大類 | 市值 | 目前權重 | 退休後權重 | 實質報酬（幾何） | 波動率 | 算法 |",
          "|---|---|---|--:|--:|--:|--:|--:|---|"]
    cfa = {a["symbol"]: a for a in cfg["assets"]}
    for a in r["assets"]:
        c = cfa[a["symbol"]]
        cl = c.get("asset_class", "")
        L.append(f"| {a['symbol']} | {CLASS_NAMES.get(cl, cl)} | {asset_group(cl)} | {fmt(c.get('value', 0))} | "
                 f"{pc(a['weight_now'])} | {pc(a['weight_retire'])} | {pc2(a['real_geometric'])} | {pc(a['vol'])} | "
                 f"{'高登方程式' if c.get('gordon') else '目前殖利率'} |")

    def bad_year(p):  # 一年報酬的第 5 百分位（約 20 年一遇的壞年）
        return (1 + p["real"]) * math.exp(-1.645 * p["vol"]) - 1
    ba, br = bad_year(pa), bad_year(pr)
    L += ["", "### 1.3 投資組合報酬與風險（實質，已扣波動拖累）", "",
          "| | 目前配置 | 退休配置 |", "|---|--:|--:|",
          f"| 長期年化報酬 | {pc2(pa['real'])} | {pc2(pr['real'])} |",
          f"| 年波動率 | {pc2(pa['vol'])} | {pc2(pr['vol'])} |",
          f"| 約 20 年一遇的壞年（單年報酬第 5 百分位） | {pc(ba)} | {pc(br)} |",
          f"| 以退休時資產 {fmt(A)} 元計，壞年約虧損 | {fmt(-A * ba)} | {fmt(-A * br)} |", ""]
    return L


def standards_section(res, rd):
    e = res["inputs_echo"]
    ra = e["retire_age"]
    L = ["## 2. 達標標準總對照表", "",
         f"「所需資產」「預估資產」都是 {ra} 歲退休時；達標年齡假設持續工作與投入；蒙地卡羅以機率衡量，沒有單一所需金額", "",
         f"| 標準 | {ra} 歲退休所需資產 | {ra} 歲預估資產 | 達標年齡 | {ra} 歲退休每月需再多存 | 動用本金 | 考慮市場波動 | 意思 |",
         "|---|--:|--:|--:|--:|:-:|:-:|---|"]
    for s in rd["standards"]:
        L.append(f"| {s['name']} | {fmt(s['need'])} | {fmt(s['have'])} | "
                 f"{'—' if s['reach_age'] is None else str(s['reach_age']) + ' 歲'} | {fmt(s['extra'])} | "
                 f"{yn(s['principal'])} | {yn(s['volatility'])} | {s['meaning']} |")
    return L + ["", "- 25 倍法則的預估資產含一次領；25 倍法則「考慮市場波動」為歷史回測，不是模擬", ""]


def income_section(res, rd):
    e, f = res["inputs_echo"], res["_calc"]["four_pct"]
    rows = res["_calc"]["target_rows"]
    L = ["## 3. 固定收入與一次領", "", "| 來源 | 方式 | 起始年齡 | 結束年齡 | 每月金額 | 一次領金額 |", "|---|---|--:|--:|--:|--:|"]
    for x in rd["income"]:
        endt = "終身" if x["end"] is None and x["type"] != "一次領" else "—" if x["end"] is None else f"{x['end'] - 1}"
        L.append(f"| {x['name']} | {x['type']} | {x['start']} | {endt} | {fmt(x['monthly'])} | {fmt(x['amount'])} |")
    L += ["", f"退休後每年由投資組合提領的缺口（到規劃終點 {e['plan_end_age']} 歲；一次領不算固定收入）：", "",
          "| 年齡 | 年生活費 | 固定收入 | 投資組合提領（缺口） | 缺口佔生活費 |", "|---|--:|--:|--:|--:|"]
    L += [f"| {s['from']}{'' if s['from'] == s['to'] else '～' + str(s['to'])} | {fmt(s['expense'])} | {fmt(s['fixed'])} | "
          f"{fmt(s['need'])} | {pc(s['need'] / s['expense'] if s['expense'] else 0)} |" for s in rd["segments"]]
    L += ["", *rd["charts"].add("expense_vs_fixed", "年生活費（橘柱）vs 固定收入（藍柱）（萬元）", [x["age"] for x in rows],
                                [("bar", [x["expense"] for x in rows]),
                                 ("bar", [min(x["expense"], sum(x["parts"].values())) for x in rows])],
                                palette=["#F28E2B", "#4E79A7"])]
    b = rd["bridge"]
    L.append(f"- 空窗期：{b['from']}～{b['to']} 歲（{b['years']} 年）固定收入尚未全部開始，多出的缺口合計 {fmt(b['total'])}"
             if b else f"- 空窗期：無（{e['retire_age']} 歲退休時固定收入都已開始）")
    if rd["lump_years"]:
        L.append(f"- 一次領合計 {fmt(rd['lump_total'])}，在請領當年存入投資組合；"
                 f"約可支付 {rd['lump_years']:.1f} 年缺口（÷ 全部固定收入開始後的年缺口 {fmt(f['annual_gap'])}，不計報酬）")
    return L + [""]


def rebalance_section(res, rd):
    e, d = res["inputs_echo"], res["deterministic"]
    A = d["projected_at_retirement"]
    L = ["## 4. 退休時配置轉換", "",
         f"以確定性預估的 {e['retire_age']} 歲退休時資產 {fmt(A)} 元計算（假設退休前每年再平衡、維持目前比例；"
         + (f"一次領 {fmt(rd['lump_total'])} 元另依退休配置投入）" if rd["lump_total"] else "不含一次領）"), "",
         "| 標的 | 類別 | 大類 | 目前權重 | 退休時調整前金額 | 退休後權重 | 調整後金額 | 買賣（+買入／−賣出） |",
         "|---|---|---|--:|--:|--:|--:|--:|"]
    groups = {}
    for x in rd["rebalance"]:
        g = groups.setdefault(x["group"], [0.0, 0.0])
        g[0] += x["w_now"]
        g[1] += x["w_ret"]
        adj = x["adjust"]
        L.append(f"| {x['symbol']} | {x['class']} | {x['group']} | {pc(x['w_now'])} | {fmt(x['before'])} | {pc(x['w_ret'])} | "
                 f"{fmt(x['after'])} | {'—' if abs(adj) < 0.5 else ('+' if adj > 0 else '−') + fmt(abs(adj))} |")
    L += [f"| **合計** | | | 100.0% | {fmt(A)} | 100.0% | {fmt(A)} | |", "",
          "| 大類 | 目前 | 退休後 |", "|---|--:|--:|"]
    L += [f"| {g} | {pc(v[0])} | {pc(v[1])} |" for g, v in groups.items()]
    return L + [""]


def early_section(cfg, res, rd):
    e, m, c = res["inputs_echo"], res["monte_carlo"], res["_calc"]
    ra, end, stress = e["retire_age"], e["plan_end_age"], e["stress_end_age"]
    early = {x["age"]: x for x in c["early_rows"]}
    in_plan = [x for x in c["cross_rows"] if x["age"] in early and x["age"] <= ra + 10]
    L = ["## 5. 各年齡退休試算", "",
         f"假設在該年齡退休：退休後停止每月投入；勞保年資依提早／延後的年數增減；勞退最早 {LP_MIN_CLAIM_AGE} 歲領；"
         f"規劃終點固定 {end} 歲。燈號依撐到 {end} 歲成功率：✅ ≥ 85%、🟡 70～85%、⚠️ 30～70%、❌ < 30%；"
         "「25 倍」✓ 表示持續工作到該年齡時已達 25 倍法則門檻", "",
         f"| 退休年齡 | 退休時資產 | 退休金目標 | 不足／餘裕 | 每月最多可花 | 撐到 {end} 歲 | 撐到 {stress} 歲 | 燈號 | 25 倍 |",
         "|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|"]
    for x in rd["early"]:
        mark = "（預計）" if x["age"] == ra else ""
        L.append(f"| {x['age']}{mark} | {fmt(x['projected'])} | {fmt(x['target'])} | {x['gap_txt']} | {fmt(x['max_monthly'])} | "
                 f"{pc(x['success_plan'])} | {pc(x['success_stress'])} | {x['light']} | {'✓' if x['four_pct'] else ''} |")
    L += ["", *rd["charts"].add("early_max_monthly", "提早退休：生活費（橘線）、該年齡退休每月可花（綠線）（萬元／月）",
                                [x["age"] for x in in_plan],
                                [("line", [x["expense"] / 12 for x in in_plan]),
                                 ("line", [early[x["age"]]["max_monthly"] for x in in_plan])],
                                unit="萬元／月", palette=CHART_PALETTE[1:]),
          f"- 每月最多可花：該年齡退休、花到 {end} 歲剛好用完的每月生活費（確定性）；圖只畫到 {ra + 10} 歲"
          "（再往後剩餘年數太少，數值會暴衝）", ""]
    return L


def sensitivity_section(res, rd):
    e, m = res["inputs_echo"], res["monte_carlo"]
    end, stress, ts = e["plan_end_age"], e["stress_end_age"], pc(m["target_success"])
    L = ["## 6. 情境敏感度", "", "每個情境只改一個條件重跑；退休年齡 ±2 年取自第 5 節（每月需再多投入與最早退休不適用）", "",
         f"| 情境 | 退休金目標 | 退休時資產 | 每月需再多投入（確定性／{ts}） | 撐到 {end} 歲 | 撐到 {stress} 歲 | "
         f"最早退休（確定性／{ts} 到 {end} 歲／{ts} 到 {stress} 歲） |", "|---|--:|--:|--:|--:|--:|--:|"]
    for x in rd["sensitivity"]:
        ext = "—" if x["earliest"] is None else "／".join("—" if a is None else str(a) for a in x["earliest"])
        L.append(f"| {x['name']} | {fmt(x['target'])} | {fmt(x['projected'])} | "
                 f"{'—' if x['extra_det'] is None and x['extra_mc'] is None else fmt(x['extra_det']) + '／' + fmt(x['extra_mc'])} | "
                 f"{pc(x['success_plan'])} | {pc(x['success_stress'])} | {ext} |")
    return L + [""]


def crossover_section(cfg, res, rd):
    e, d = res["inputs_echo"], res["deterministic"]
    rows, ca, ra = res["_calc"]["cross_rows"], d["crossover_age"], e["retire_age"]
    exp_m = cfg["monthly_expense"]
    shown = [x for x in rows if x["age"] <= (min(ca + 3, e["stress_end_age"]) if ca else e["stress_end_age"])]
    L = ["## 7. 跨越點", "",
         f"被動收入 = 年初投資資產 × 實質報酬 + 固定收入；假設持續工作、每月投入 {fmt(cfg.get('monthly_contribution', 0))}、不提領；"
         f"{ra} 歲前用目前配置報酬 {pc2(res['returns']['accumulation_real'])}，之後用退休配置 {pc2(res['returns']['retirement_real'])}；"
         "一次領在請領當年存入", "",
         *rd["charts"].add("crossover", "跨越點：被動收入（藍柱）、生活費（橘線）（萬元／月）", [x["age"] for x in rows],
                           [("bar", [(x["invest"] + x["fixed"]) / 12 for x in rows]),
                            ("line", [x["expense"] / 12 for x in rows])], unit="萬元／月"),
         f"| 年齡 | 年初投資資產{'（含一次領）' if rd['lump_total'] else ''} | 每月投資收益 | 每月固定收入 | 每月被動收入合計 | 佔生活費 |",
         "|--:|--:|--:|--:|--:|--:|"]
    for x in shown:
        if (x["age"] - e["age"]) % 5 == 0 or x["age"] in (ra, ca, shown[-1]["age"]):
            tot = (x["invest"] + x["fixed"]) / 12
            mark = "（預計退休）" if x["age"] == ra else "（跨越點）" if x["age"] == ca else ""
            L.append(f"| {x['age']}{mark} | {fmt(x['assets'])} | {fmt(x['invest'] / 12)} | {fmt(x['fixed'] / 12)} | "
                     f"{fmt(tot)} | {pc(tot / exp_m)} |")
    L += ["", f"- 跨越點：{f'{ca} 歲' if ca else str(e['stress_end_age']) + ' 歲前不會跨越'}",
          f"- {ra} 歲就跨越所需投資資產 = (年生活費 − 固定收入) ÷ 退休後實質報酬 = {fmt(rd['cross_need_at_retire'])}", ""]
    return L


def four_pct_section(cfg, res, rd):
    e = res["inputs_echo"]
    f, lm, pr = res["_calc"]["four_pct"], res["_calc"]["lmp"], res["_calc"]["pool_rows"]
    ra, end, stress = e["retire_age"], e["plan_end_age"], e["stress_end_age"]
    have = f["projected"] + f["lumps"]
    cls = {a["symbol"]: a.get("asset_class", "") for a in cfg["assets"]}
    risky_txt = "、".join(f"{a['symbol']} {pc(a['weight_retire'])}" for a in res["returns"]["assets"] if a["weight_retire"] > 0)
    has_cash = any(asset_group(c) == "現金" for c in cls.values())
    L = ["## 8. 25 倍法則與資產重組", "", "### 8.1 25 倍法則（4% 法則，William Bengen）", "",
         "$$\\text{門檻} = 25 \\times (\\text{年生活費} - \\text{年固定收入}) + \\text{過渡期缺口}$$", "",
         f"- 年生活費 = {fmt(f['annual_expense'])}",
         f"- 年固定收入 = {fmt(f['fixed_income'])}（{f['full_income_age']} 歲起所有月領與每年收入都開始後；一次領不計入）",
         f"- 年缺口 = {fmt(f['annual_expense'])} − {fmt(f['fixed_income'])} = {fmt(f['annual_gap'])}",
         f"- 25 倍 = 25 × {fmt(f['annual_gap'])} = {fmt(f['base'])}"]
    if f["bridge"] > 0:
        L.append(f"- 過渡期缺口：{ra}～{f['full_income_age'] - 1} 歲固定收入尚未全部開始，多出的缺口合計 {fmt(f['bridge'])}（實質、不折現）")
    L += [f"- **{ra} 歲退休門檻 = {fmt(f['target'])}**；{ra} 歲預估資產" + ("（含一次領）" if f["lumps"] else "")
          + f" = {fmt(have)} → " + ("已達標" if have >= f["target"] else f"不足 {fmt(f['target'] - have)}"),
          "- 達標年齡（持續工作、每月投入）：" + (f"{f['reach_age']} 歲（該年齡退休的門檻 {fmt(f['reach_target'])}，"
                                       "門檻隨退休年齡變動）" if f["reach_age"] else f"{end} 歲前達不到"), "",
          "### 8.2 安全池（負債配合組合）與風險池", "",
          "安全池 = 退休後前 N 年每年須自行負擔的缺口加總（已扣固定收入，實質、不折現），只放"
          + ("定存／現金" if has_cash else "現金（沒有現金標的，假設實質報酬 0）")
          + f"；風險池照退休配置持有並每年再平衡：{risky_txt}"
          + (f"。安全池金額沒有扣一次領；佔比的分母是 {ra} 歲資產（不含一次領），一次領 {fmt(f['lumps'])} 在請領當年另外存入安全池"
             if f["lumps"] else ""), "",
          f"| 安全池涵蓋 | 安全池金額 | 佔 {ra} 歲資產 | 風險池金額 |", "|--:|--:|--:|--:|"]
    for x in lm["rows"]:
        risk = fmt(x["risk"]) if x["risk"] >= 0 else f"不足 {fmt(-x['risk'])}"
        L.append(f"| {x['years']} 年 | {fmt(x['lmp'])} | {pc(x['share'])} | {risk} |")
    L += ["", *rd["charts"].add("safe_pool", "退休時資產（橘柱）vs 安全池所需金額（藍柱）（萬元）",
                                [f'"{x["years"]} 年"' for x in lm["rows"]],
                                [("bar", [f["projected"]] * len(lm["rows"])), ("bar", [x["lmp"] for x in lm["rows"]])],
                                palette=["#F28E2B", "#4E79A7"]),
          "### 8.3 蒙地卡羅：兩池分開運作 vs 每年再平衡", "",
          "同一組市場走勢。兩池：生活費從安全池付；前一年風險池上漲才賣風險池把安全池補到未來 N 年缺口，下跌不賣；"
          "安全池用完才賣風險池。「最差 25%」＝退休後前 5 年報酬最差的四分之一路徑", "",
          f"| 提領方式 | 退休時安全池比例 | 退休時股票比例 | 撐到 {end} 歲 | 撐到 {stress} 歲 | 最差 25%：撐到 {end} 歲 | 最差 25%：撐到 {stress} 歲 |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for x in pr:
        name = "每年再平衡（退休配置）" if x["years"] is None else f"兩池：安全池 {x['years']} 年"
        si = "—" if x["safe_init"] is None else pc(x["safe_init"])
        L.append(f"| {name} | {si} | {pc(x['stock_init'])} | {pc(x['success_plan'])} | {pc(x['success_stress'])} | "
                 f"{pc(x['worst_plan'])} | {pc(x['worst_stress'])} |")
    best = max(pr[1:], key=lambda x: (x["success_stress"], x["success_plan"]))
    base = pr[0]
    wins = [x["years"] for x in pr[1:] if x["worst_stress"] > base["worst_stress"]]
    L += ["", f"- 撐到 {stress} 歲：兩池最佳為安全池 {best['years']} 年 {pc(best['success_stress'])}，每年再平衡 "
          f"{pc(base['success_stress'])} → " + ("兩池較好" if best["success_stress"] > base["success_stress"] else "每年再平衡較好"),
          "- 最差 25% 情境：" + (f"安全池 {'、'.join(map(str, wins))} 年比每年再平衡撐得久" if wins else "兩池都沒有比每年再平衡撐得久"),
          "- 兩池退休時的股票比例 = （1 − 安全池比例）× 退休配置的股票比例", ""]
    return L


def mc_section(res, rd):
    e, d, m = res["inputs_echo"], res["deterministic"], res["monte_carlo"]
    a, mp = res["_calc"]["mc_analysis"], res["_calc"]["mc_pcts"]
    ra, end, stress = e["retire_age"], e["plan_end_age"], e["stress_end_age"]
    tgt, cr, fs = d["target"], a["cross"], a["fail_stress"]
    L = ["## 9. 蒙地卡羅模擬", "",
         f"{m['paths']:,} 條路徑；各資產年對數報酬為多變量常態（含相關係數）；退休前用目前配置、{ra} 歲起用退休配置，每年再平衡；"
         f"退休後每年初先提領。成功＝規劃期內每年生活費都付得出", "",
         "### 9.1 退休時資產分布", "",
         f"| 百分位 | {ra} 歲資產 | 與退休金目標 {fmt(tgt)} 相比 |", "|---|--:|--:|"]
    for p in (5, 10, 25, 50, 75, 90):
        v = a["ret_pcts"][p]
        L.append(f"| P{p} | {fmt(v)} | {'+' if v >= tgt else '−'}{fmt(abs(v - tgt))}"
                 + (f"（{v / tgt:.2f} 倍）" if tgt > 0 else "") + " |")
    L += ["", f"- 退休時達到目標的機率：{pc(m['prob_reach_target'])}", "",
          "### 9.2 各年齡還有錢的機率", "", "| 年齡 | 還有錢的機率 |", "|--:|--:|"]
    for x in a["survival"]:
        if (x["age"] - ra) % 5 == 0 or x["age"] in (end, stress):
            mark = "（規劃終點）" if x["age"] == end else "（壓力測試）" if x["age"] == stress else ""
            L.append(f"| {x['age']}{mark} | {pc(x['rate'])} |")
    L += ["", *rd["charts"].add("survival", "各年齡還有錢的機率（%）", [x["age"] for x in a["survival"]],
                                [("line", [x["rate"] for x in a["survival"]])], unit="%", scale=0.01),
          "### 9.3 退休時達標 vs 規劃期成功", "",
          "| | 規劃期成功 | 規劃期失敗 |", "|---|--:|--:|",
          f"| 退休時達標 | {pc(cr['reach_ok'])} | {pc(cr['reach_fail'])} |",
          f"| 退休時未達標 | {pc(cr['miss_ok'])} | {pc(cr['miss_fail'])} |", ""]
    if a["seq"]:
        L += ["### 9.4 報酬順序風險", "",
              f"| 退休後前 {a['seq_years']} 年報酬 | 年化實質報酬區間 | 撐到 {end} 歲 | 撐到 {stress} 歲 |", "|---|--:|--:|--:|"]
        L += [f"| {s['name']} | {pc(s['lo'])} ～ {pc(s['hi'])} | {pc(s['success_plan'])} | {pc(s['success_stress'])} |"
              for s in a["seq"]]
        L.append("")
    L += ["### 9.5 失敗路徑的耗盡年齡", ""]
    if fs["pcts"]:
        L += [f"- 算到 {stress} 歲有 {pc(fs['rate'])} 的路徑耗盡；耗盡年齡 P10 {fs['pcts'][10]:.0f}／"
              f"中位數 {fs['pcts'][50]:.0f}／P90 {fs['pcts'][90]:.0f} 歲",
              "- 規劃期內失敗路徑的耗盡年齡中位數：" + (f"{m['failed_depletion_age_median']:.0f} 歲"
                                                if m["failed_depletion_age_median"] else "—（規劃期內沒有失敗路徑）"), ""]
    else:
        L += [f"- 算到 {stress} 歲，所有路徑都沒有耗盡", ""]
    L += ["### 9.6 各年初資產分布", "",
          *rd["charts"].add("mc_percentiles", "蒙地卡羅：各年初資產（藍線 P10、橘線 P50、綠線 P90）（萬元）",
                            [x["age"] for x in mp], [("line", [x[p] for x in mp]) for p in ("P10", "P50", "P90")])]
    return L


def report(cfg, res, rd=None):
    rd = rd or derive(cfg, res)
    L = [f"# 退休金試算計算底稿（{cfg.get('as_of', '')}）", "",
         "> 金額皆為今天的購買力（實質新台幣）。本檔只列數據與算式；同樣的數據另存結構化數據檔（--json）。", ""]
    body = (inputs_section(cfg, res) + standards_section(res, rd) + income_section(res, rd) + rebalance_section(res, rd)
            + early_section(cfg, res, rd) + sensitivity_section(res, rd) + crossover_section(cfg, res, rd)
            + four_pct_section(cfg, res, rd) + mc_section(res, rd) + calc_sections(cfg, res, rd["charts"]))
    L += summary_section(cfg, res, rd) + body
    L += ["## 11. 資料來源", "", "| 項目 | 數值 | 來源 |", "|---|---|---|"]
    L += [f"| {s['item']} | {s['value']} | {s['source']} |" for s in cfg.get("sources", [])]
    L += ["", "## 12. 假設與限制", ""] + [f"- {n}" for n in cfg.get("notes", [])] + [
        "- 未計所得稅與基金內扣費用" + ("（美國掛牌股票 ETF 已依殖利率扣除 30% 股利預扣稅拖累，見 §10.2 的 drag 欄）"
                                     if any(x.get("drag", 0) > 0 for x in cfg["assets"]) else ""),
        "- 股票預期報酬用高登方程式，假設估值維持現狀；若未來估值下修，實際報酬會更低",
        "- 未計匯率風險：海外資產以新台幣計價，視同本地資產",
        "- 報酬分布假設為對數常態，真實市場的極端事件（肥尾）可能更常發生", ""]
    return "\n".join(L)


def to_json(cfg, res, rd):
    """結構化數據檔：報告各節的表格、圖表資料與 run() 的完整結果。"""
    out = {"as_of": cfg.get("as_of"), "profile": cfg.get("profile"), **rd, "result": res}
    conv = lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o)
    return json.dumps(out, ensure_ascii=False, indent=2, default=conv)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--report", help="計算底稿 Markdown 的輸出路徑")
    ap.add_argument("--json", help="結構化數據檔的輸出路徑")
    a = ap.parse_args()
    with open(a.input, encoding="utf-8") as f:
        cfg = json.load(f)
    res = run(cfg)
    if a.report or a.json:
        rd = derive(cfg, res)
        md = report(cfg, res, rd)  # 產生報告時一併收集圖表資料
        if a.report:
            with open(a.report, "w", encoding="utf-8") as f:
                f.write(md)
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                f.write(to_json(cfg, res, rd))
    res.pop("_calc")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
