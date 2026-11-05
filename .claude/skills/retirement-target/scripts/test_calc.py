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
