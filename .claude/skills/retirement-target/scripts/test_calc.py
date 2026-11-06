# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pytest"]
# ///
"""執行：uv run --with numpy --with pytest pytest test_calc.py -q"""
import math

import numpy as np
import pytest

import calc


def base_cfg(**kw):
    cfg = {
        "age": 40, "retire_age": 60, "remaining_life_years": 25, "longevity_buffer_years": 5,
        "stress_end_age": 95, "monthly_expense": 50000, "monthly_contribution": 20000,
        "inflation": 0.02,
        "assets": [{"symbol": "A", "value": 1_000_000, "expected_return": 0.0404, "vol": 0.0}],
        "mc": {"paths": 500, "seed": 1},
    }
    cfg.update(kw)
    return cfg


def test_fisher_real_return():
    res = calc.run(base_cfg())
    assert res["returns"]["accumulation_real"] == pytest.approx(1.0404 / 1.02 - 1)


def test_gordon_real_return():
    a = {"symbol": "S", "value": 1_000_000, "vol": 0.0, "drag": 0.003,
         "gordon": {"div_yield": 0.01, "real_growth": 0.015}, "ltcma_return": 0.075}
    cfg = base_cfg(assets=[a])
    res = calc.run(cfg)
    assert res["returns"]["accumulation_real"] == pytest.approx(0.01 + 0.015 - 0.003)
    rep = calc.report(cfg, res)
    assert "高登方程式" in rep and "J.P. Morgan LTCMA" in rep


def test_zero_return_is_simple_sum():
    cfg = base_cfg(inflation=0.0, assets=[{"symbol": "A", "value": 1_000_000, "expected_return": 0.0, "vol": 0.0}])
    d = calc.run(cfg)["deterministic"]
    # 40 歲 + 25 年餘命 + 5 年 = 70 歲，退休 10 年
    assert d["target"] == pytest.approx(10 * 12 * 50000)
    assert d["projected_at_retirement"] == pytest.approx(1_000_000 + 20 * 12 * 20000)


def test_target_matches_annuity_due_formula():
    res = calc.run(base_cfg())
    r, T, C = res["returns"]["retirement_real"], 10, 12 * 50000
    expected = C * (1 - (1 + r) ** -T) / r * (1 + r)
    assert res["deterministic"]["target"] == pytest.approx(expected)


def test_zero_vol_monte_carlo_equals_deterministic():
    res = calc.run(base_cfg())
    d, m = res["deterministic"], res["monte_carlo"]
    assert m["assets_at_retirement"]["P50"] == pytest.approx(d["projected_at_retirement"])
    ok = d["depletion_age"] is None or d["depletion_age"] >= res["inputs_echo"]["plan_end_age"]
    assert m["success_plan"] == (1.0 if ok else 0.0)


def test_extra_monthly_closes_gap():
    cfg = base_cfg(monthly_expense=120000)
    d = calc.run(cfg)["deterministic"]
    assert d["gap"] > 0
    r = calc.run(cfg)["returns"]["accumulation_real"]
    fv = calc.project_rows(1_000_000, 20000 + d["extra_monthly_needed"], r, 40, 20)[-1]["end"]
    assert fv == pytest.approx(d["target"])


def test_mc_extra_reaches_target_success():
    cfg = base_cfg(monthly_expense=120000,
                   assets=[{"symbol": "A", "value": 1_000_000, "expected_return": 0.07, "vol": 0.18}])
    m = calc.run(cfg)["monte_carlo"]
    assert m["success_plan"] < 0.85
    cfg["monthly_contribution"] = 20000 + m["extra_monthly_needed"] + 1
    assert calc.run(cfg)["monte_carlo"]["success_plan"] >= 0.85


def test_labor_insurance_formula():
    li = {"avg_insured_salary": 45800, "years": 35, "claim_age": 60, "legal_age": 65}
    # 兩式擇優取 45800*35*1.55%，提前 5 年減 20%
    x = calc.labor_insurance_detail(li)
    assert x["monthly"] == pytest.approx(45800 * 35 * 0.0155 * 0.8)
    assert x["formula_a"] == pytest.approx(45800 * 35 * 0.00775 + 3000)
    # 年資未滿 15 年不能月領
    li.update(avg_insured_salary=30000, years=10, claim_age=65)
    with pytest.raises(ValueError):
        calc.labor_insurance_detail(li)


def test_labor_insurance_lump_sum():
    li = {"avg_insured_salary": 45800, "years": 28, "claim_age": 60, "mode": "lump_sum", "pre2009_years": True}
    # 15 + 2 × 13 = 41 個月
    assert calc.labor_insurance_detail(li)["lump"] == pytest.approx(45800 * 41)
    li["years"] = 40  # 上限 45 個月
    assert calc.labor_insurance_detail(li)["months"] == 45
    li["pre2009_years"] = False
    with pytest.raises(ValueError):
        calc.labor_insurance_detail(li)


def test_pension_reduces_target():
    cfg = base_cfg(labor_insurance={"avg_insured_salary": 45800, "years": 30, "claim_age": 65, "legal_age": 65})
    d = calc.run(cfg)["deterministic"]
    assert d["target"] < d["target_without_pension"]


def test_retirement_allocation_and_correlation():
    cfg = base_cfg(
        assets=[{"symbol": "EQ", "value": 1_000_000, "expected_return": 0.07, "vol": 0.18},
                {"symbol": "BD", "value": 0, "expected_return": 0.04, "vol": 0.06}],
        correlation={"symbols": ["EQ", "BD"], "matrix": [[1, 0.2], [0.2, 1]]},
        retirement_allocation={"EQ": 0.4, "BD": 0.6})
    r = calc.run(cfg)["returns"]
    assert r["assets"][0]["weight_now"] == 1.0
    assert r["assets"][1]["weight_retire"] == pytest.approx(0.6)
    assert r["retirement_real"] < r["accumulation_real"]


def test_volatility_drag():
    w, mu, cov = np.array([1.0]), np.array([math.log(1.05)]), np.array([[0.04]])
    assert calc.portfolio_return_detail(w, mu, cov)["real"] == pytest.approx(0.05)


def test_mc_analysis_consistent():
    cfg = base_cfg(monthly_expense=90000,
                   assets=[{"symbol": "A", "value": 1_000_000, "expected_return": 0.07, "vol": 0.18}])
    res = calc.run(cfg)
    a, m, e = res["_calc"]["mc_analysis"], res["monte_carlo"], res["inputs_echo"]
    assert sum(a["cross"].values()) == pytest.approx(1.0)
    assert a["cross"]["reach_ok"] + a["cross"]["reach_fail"] == pytest.approx(m["prob_reach_target"])
    rates = [x["rate"] for x in a["survival"]]
    assert all(x >= y for x, y in zip(rates, rates[1:]))
    surv = {x["age"]: x["rate"] for x in a["survival"]}
    assert surv[e["plan_end_age"]] == pytest.approx(m["success_plan"])
    assert surv[e["stress_end_age"]] == pytest.approx(m["success_to_95"])
    assert len(a["seq"]) == 4 and a["seq"][0]["success_stress"] <= a["seq"][-1]["success_stress"]
    assert "### 重點解讀" in calc.report(cfg, res)


def test_mc_report_zero_vol():
    res = calc.run(base_cfg())
    assert "## 蒙地卡羅模擬" in calc.report(base_cfg(), res)


def test_report_retirement_portfolio():
    cfg = base_cfg(
        assets=[{"symbol": "EQ", "asset_class": "us_large_cap", "value": 1_000_000, "expected_return": 0.07, "vol": 0.18},
                {"symbol": "BD", "asset_class": "us_aggregate_bond", "value": 0, "expected_return": 0.04, "vol": 0.06}],
        retirement_allocation={"EQ": 0.4, "BD": 0.6})
    rep = calc.report(cfg, calc.run(cfg))
    assert "## 退休後的投資組合" in rep
    assert "| 股票 | 100.0% | 40.0% |" in rep and "| 債券 | 0.0% | 60.0% |" in rep
    assert "為什麼退休要換配置" in rep


def test_four_pct_rule_without_income():
    res = calc.run(base_cfg())
    f = res["four_percent_rule"]
    assert f["target"] == pytest.approx(25 * 12 * 50000) and f["bridge"] == 0


def test_four_pct_rule_with_bridge():
    # 60 歲退休、65 歲起勞保月領：門檻 = 25 × (年生活費 − 勞保) + 60～64 歲每年的勞保金額
    li = {"avg_insured_salary": 45800, "years": 30, "claim_age": 65, "legal_age": 65}
    res = calc.run(base_cfg(labor_insurance=li))
    li_year = 12 * calc.labor_insurance_detail(li)["monthly"]
    f = res["four_percent_rule"]
    assert f["annual_gap"] == pytest.approx(12 * 50000 - li_year)
    assert f["bridge"] == pytest.approx(5 * li_year)
    assert f["target"] == pytest.approx(25 * (12 * 50000 - li_year) + 5 * li_year)


def test_other_income_lump():
    # 一次性收入（例如保單領回）在當年存入投資組合，不算成每年的固定收入
    lump = {"name": "保單", "lump": True, "age": 61, "amount": 300000}
    cfg = base_cfg(inflation=0.0, assets=[{"symbol": "A", "value": 1_000_000, "expected_return": 0.0, "vol": 0.0}],
                   other_income=[lump])
    res = calc.run(cfg)
    assert res["deterministic"]["target"] == pytest.approx(10 * 12 * 50000 - 300000)
    f = res["four_percent_rule"]
    assert f["annual_gap"] == pytest.approx(12 * 50000) and f["target"] == pytest.approx(25 * 12 * 50000)
    assert "保單一次領" in calc.report(cfg, res)


def test_other_income_lump_before_retirement_deferred():
    # 退休前領到的一次性收入，延到退休當年才存入（不計中間的報酬，保守）
    cfg = base_cfg(inflation=0.0, assets=[{"symbol": "A", "value": 1_000_000, "expected_return": 0.0, "vol": 0.0}],
                   other_income=[{"name": "保單", "lump": True, "age": 55, "amount": 300000}])
    assert calc.run(cfg)["deterministic"]["target"] == pytest.approx(10 * 12 * 50000 - 300000)


def test_liability_matching():
    cfg = base_cfg(
        assets=[{"symbol": "EQ", "asset_class": "us_large_cap", "value": 1_000_000, "expected_return": 0.07, "vol": 0.18},
                {"symbol": "BD", "asset_class": "us_aggregate_bond", "value": 0, "expected_return": 0.04, "vol": 0.06}],
        retirement_allocation={"EQ": 0.4, "BD": 0.6})
    res = calc.run(cfg)
    lm, A = res["_calc"]["lmp"], res["deterministic"]["projected_at_retirement"]
    assert lm["rows"][0]["lmp"] == pytest.approx(5 * 12 * 50000)
    assert lm["rows"][0]["risk"] == pytest.approx(A - 5 * 12 * 50000)
    rep = calc.report(cfg, res)
    assert "25 倍法則（4% 法則）" in rep and "負債配合投資組合" in rep
    assert "風險池**照你的退休配置**持有並每年再平衡：EQ 40.0%、BD 60.0%" in rep


def test_two_pool_zero_return():
    # 零報酬零波動：兩池總額只減不增，撐過 n 年 ⇔ 起始資產 ≥ 前 n 年缺口加總；一次領（負數）存入安全池
    mu, cov = np.zeros(2), np.zeros((2, 2))
    w, w_safe = np.array([0.5, 0.5]), np.array([0.0, 1.0])
    needs = np.array([100.0, 100.0, -50.0, 100.0, 100.0])
    z = np.zeros((2, len(needs), 2))
    dep = calc.simulate_two_pool(mu, cov, w, w_safe, np.array([350.0, 349.0]), needs, z, 2)
    assert dep[0] == -1 and dep[1] == 4


def test_two_pool_independent_pools():
    # 安全池只放現金（實質 0），風險池照退休配置 EQ/BD 各半：EQ 每年 +10%、BD 0
    mu, cov = np.log(np.array([1.1, 1.0, 1.0])), np.zeros((3, 3))
    w_ret, w_safe = np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.0, 1.0])
    needs = np.array([100.0, 100.0])
    z = np.zeros((1, 2, 3))
    # 起始 195：安全池 100 付第 1 年；風險池 95 照 50/50 成長 5% → 99.75 < 100，第 2 年耗盡
    # （若風險池只放股票會成長到 104.5 而撐過，可分辨風險池有照退休配置）
    assert calc.simulate_two_pool(mu, cov, w_ret, w_safe, np.array([195.0]), needs, z, 1)[0] == 1
    # 起始 196：風險池 96 → 100.8，撐過；沒有現金標的（None）時安全池視為實質 0，結果相同
    assert calc.simulate_two_pool(mu, cov, w_ret, None, np.array([196.0]), needs, z, 1)[0] == -1


def test_two_pool_in_report():
    cfg = base_cfg(
        assets=[{"symbol": "EQ", "asset_class": "us_large_cap", "value": 1_000_000, "expected_return": 0.07, "vol": 0.18},
                {"symbol": "BD", "asset_class": "us_aggregate_bond", "value": 0, "expected_return": 0.04, "vol": 0.06}],
        retirement_allocation={"EQ": 0.4, "BD": 0.6})
    res = calc.run(cfg)
    tp = res["two_pool_mc"]
    assert tp[0]["strategy"] == "rebalance" and len(tp) == 5
    assert tp[0]["success_plan"] == pytest.approx(res["monte_carlo"]["success_plan"])
    assert "### 蒙地卡羅：兩個池子分開運作" in calc.report(cfg, res)
    pr = res["_calc"]["pool_rows"]
    assert pr[0]["safe_init"] is None and pr[0]["stock_init"] == pytest.approx(0.4)
    assert pr[1]["stock_init"] == pytest.approx(0.4 * (1 - pr[1]["safe_init"]))


def test_two_pool_target_ignores_lump():
    # 安全池目標應以「未扣一次領」的缺口計算；一次領另外存入安全池，不能讓 N 年安全池少放一年
    # 風險池每年 −50%、安全池實質 0；第 1 年缺口 100 但有一次領 50 → needs[0] = 50
    mu, cov = np.log(np.array([0.5, 1.0])), np.zeros((2, 2))
    w_ret, w_safe = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    needs, gap = np.array([50.0, 100.0, 100.0]), np.array([100.0, 100.0, 100.0])
    z = np.zeros((1, 3, 2))
    # 目標 = gap 前 2 年 200：安全池 200、風險池 200 → 撐過
    assert calc.simulate_two_pool(mu, cov, w_ret, w_safe, np.array([400.0]), needs, z, 2, gap)[0] == -1
    # 舊算法（目標用扣掉一次領的 needs，只有 150）→ 第 3 年耗盡
    assert calc.simulate_two_pool(mu, cov, w_ret, w_safe, np.array([400.0]), needs, z, 2)[0] == 2


def test_two_pool_safe_init_matches_liability_matching_with_lump():
    cfg = base_cfg(
        assets=[{"symbol": "EQ", "asset_class": "us_large_cap", "value": 1_000_000, "expected_return": 0.07, "vol": 0.18},
                {"symbol": "CASH", "asset_class": "tw_cash", "value": 0, "expected_return": 0.017, "vol": 0.005}],
        other_income=[{"name": "保單", "lump": True, "age": 60, "amount": 3_000_000}])
    res = calc.run(cfg)
    pr = {x["years"]: x["safe_init"] for x in res["_calc"]["pool_rows"]}
    A = res["deterministic"]["projected_at_retirement"]
    assert pr[5] == pytest.approx(min(1.0, 5 * 12 * 50000 / A))


