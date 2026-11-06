# calc.py 輸入 JSON 格式

所有金額為新台幣，以**今天的購買力**表示。報酬率、通膨用小數（0.075 = 7.5%）。

```json
{
  "as_of": "2026-11-05",
  "profile": [
    {"item": "性別／出生年", "value": "男，1986 年"},
    {"item": "目前年齡／預計退休年齡", "value": "40 歲／60 歲"},
    {"item": "持股", "value": "0050 2,000,000 元、VOO 1,000,000 元"}
  ],
  "age": 40,
  "retire_age": 60,
  "remaining_life_years": 40.2,
  "longevity_buffer_years": 5,
  "stress_end_age": 95,
  "monthly_expense": 60000,
  "monthly_contribution": 20000,
  "inflation": 0.02,

  "assets": [
    {"symbol": "0050", "asset_class": "tw_equity", "value": 2000000, "vol": 0.198, "drag": 0,
     "gordon": {"div_yield": 0.025, "real_growth": 0.015}, "ltcma_return": 0.071, "hist_return": 0.15},
    {"symbol": "VOO", "asset_class": "us_large_cap", "value": 1000000, "vol": 0.154, "drag": 0.0032,
     "gordon": {"div_yield": 0.0105, "real_growth": 0.015}, "ltcma_return": 0.075, "hist_return": 0.151},
    {"symbol": "BND", "asset_class": "us_aggregate_bond", "value": 0, "expected_return": 0.042, "vol": 0.052, "drag": 0,
     "ltcma_return": 0.05},
    {"symbol": "CASH", "asset_class": "tw_cash", "value": 500000, "expected_return": 0.017, "vol": 0.005}
  ],
  "correlation": {"symbols": ["0050", "VOO", "BND"], "matrix": [[1, 0.64, 0.30], [0.64, 1, 0.44], [0.30, 0.44, 1]]},
  "retirement_allocation": {"0050": 0.2, "VOO": 0.2, "BND": 0.5, "CASH": 0.1},

  "labor_insurance": {"avg_insured_salary": 45800, "years": 35, "claim_age": 65, "legal_age": 65, "mode": "monthly", "pre2009_years": true},
  "labor_pension": {"balance": 800000, "monthly_wage": 70000, "rate": 0.06, "return_nominal": 0.03, "claim_age": 60, "mode": "monthly"},
  "other_income": [{"name": "房租", "annual": 120000, "start_age": 60, "end_age": 80},
                   {"name": "保單領回", "lump": true, "age": 65, "amount": 1000000}],

  "mc": {"paths": 10000, "seed": 42, "target_success": 0.85},

  "sources": [{"item": "通膨率", "value": "2.0%", "source": "max(主計總處 20 年平均 1.4%, 2%)"}],
  "notes": ["00679B 上市未滿 10 年，波動率以 117 個月計算"]
}
```

## 欄位說明

| 欄位 | 必填 | 說明 |
|---|---|---|
| `profile[]` |  | 報告開頭「使用者基本資料」表：`{"item", "value"}`，列出使用者在步驟 1 提供的所有個人資料（含選填項目與職業、保險年資等額外資料），依使用者原意、不做推算；省略時由年齡、生活費、投入、持股自動產生簡表 |
| `age`、`retire_age` | ✅ | 整數年齡 |
| `remaining_life_years` | ✅ | 生命表上「目前年齡」的平均餘命 |
| `longevity_buffer_years` |  | 預設 5；規劃終點 = ceil(age + 餘命 + buffer) |
| `stress_end_age` |  | 預設 95；蒙地卡羅另外報告活到這個年齡的成功率 |
| `monthly_expense` | ✅ | 退休後每月生活費（今天的物價） |
| `monthly_contribution` |  | 退休前每月投入（實質，隨通膨調整） |
| `inflation` | ✅ | 通膨率 |
| `assets[]` | ✅ | `value` 可為 0，用來放「退休後才會買」的標的 |
| `assets[].gordon` | 股票 ✅ | 高登方程式：`{"div_yield": 起始股息殖利率, "real_growth": 實質股息成長率}`，實質報酬 = div_yield + real_growth − drag。有這個欄位時不看 `expected_return` |
| `assets[].expected_return` | 債券、現金 ✅ | **名目**預期報酬 = 目前殖利率（債券 SEC yield 或 `div_yield_ttm`、現金為定存利率），以費雪方程式換成實質 |
| `assets[].ltcma_return`、`assets[].hist_return` |  | 名目的 J.P. Morgan LTCMA 預期報酬、`fetch_market.py` 的 `hist_return_annual`；只在報告中當對照，不用於計算 |
| `assets[].vol` | ✅ | 年化波動率（對數報酬） |
| `assets[].drag` |  | 從預期報酬扣除的拖累：美國掛牌股票 ETF 的股利預扣稅 = 殖利率 × 30%（例：VOO 1.05% × 30% ≈ 0.0032），債券 ETF 為 0；規則見 defaults.md |
| `correlation` |  | 缺少的配對視為 0；現金不必放 |
| `retirement_allocation` |  | 退休後的權重，鍵名須是 `assets` 裡的 symbol；省略時沿用目前配置 |
| `labor_insurance` |  | 勞保；`legal_age` 依出生年查 defaults.md（只有月領用到） |
| `labor_insurance.mode` |  | `monthly`（預設，月領年金）或 `lump_sum`（一次領，`claim_age` 當年存入投資組合，不可早於退休） |
| `labor_insurance.pre2009_years` |  | 98 年前是否有勞保年資；年資滿 15 年要選 `lump_sum` 時必須為 `true` |
| `labor_pension` |  | 勞退個人專戶；請領年齡至少 60 且不早於退休 |
| `labor_pension.mode` |  | `monthly`（預設，在規劃期內攤成月領）或 `lump_sum`（請領年齡當年把專戶餘額全數存入投資組合） |
| `other_income[]` |  | 每年的固定收入：`annual` 為實質年額，`start_age` 起、`end_age` 不含，省略表示終身 |
| `other_income[].lump` |  | `true` 表示一次性收入（保單領回、資產出售等）：`age` 當年把 `amount`（實質）存入投資組合，不算成每年的固定收入；`age` 早於退休時延到退休當年存入（不計中間的報酬） |
| `sources[]`、`notes[]` |  | 原樣寫進報告 |
