# 內建預設值與資料來源

只有在執行時 WebSearch 查不到最新資料時才用這份表，並在報告中標註「內建預設值（建表日期 2026-11-05）」。

## 預期報酬的算法

依伯恩斯坦《投資金律》，用「基本報酬率」而不是歷史平均報酬：歷史報酬裡包含估值膨脹的投機報酬，無法期待它重演。

| 類別 | 算法 | JSON |
|---|---|---|
| 股票（`tw_equity`、`tw_single_stock`、`us_*_cap`、`global_equity`、`dev_ex_us_equity`、`em_equity`） | **高登方程式**：實質報酬 = 起始股息殖利率 + 實質股息成長率 − drag | `gordon: {div_yield, real_growth}` |
| 債券（`us_treasury_*`、`us_aggregate_bond`、`us_ig_corp`、`us_high_yield`） | **目前殖利率**當名目報酬，以費雪方程式換成實質 | `expected_return` = 殖利率 |
| 現金（`tw_cash`） | 台灣一年期定存利率（WebSearch 台銀牌告利率） | `expected_return` |
| 黃金（`gold`） | 不配息，高登方程式不適用，沿用 JPM LTCMA | `expected_return` |

- **股息殖利率**：用 `fetch_market.py` 的 `div_yield_ttm`（近 12 個月配息 ÷ 最新收盤價）。抓不到時用「股利預扣稅拖累」一節的預設殖利率表
- **實質股息成長率 `real_growth`**：各市場統一 **1.5%**（美國近百年實質股息成長率約 1.5～2.0%，取下緣）
- **債券殖利率**：優先 WebSearch 該 ETF 的 30 日 SEC yield（較接近到期殖利率）；查不到用 `div_yield_ttm`；再查不到用下表的 JPM 數字
- 下表的 JPM LTCMA 數字填進 `ltcma_return`，只在報告中當對照

## 各資產類別：J.P. Morgan LTCMA 預期報酬（名目、幾何／複利、10～15 年，對照用）與預設波動率

下表「JPM 2027」為 2026-10-19 發布值；標示「估計」者 JPM 新聞稿未公布，是依相近類別推估的。
波動率應優先用 `fetch_market.py` 抓的標的歷史值；下表只在歷史資料不足 24 個月時使用。
USD 計價的報酬直接視為新台幣報酬（第一版不處理匯率風險）。

| asset_class | 說明／常見標的 | JPM 預期報酬 | 來源 | 預設波動率 |
|---|---|--:|---|--:|
| `tw_equity` | 台股大盤 ETF：0050、006208、00850 | 7.1% | 估計：用 JPM 2027 新興市場股票（台、韓佔 EM 指數 51%） | 18% |
| `tw_single_stock` | 台股個股：2330、2317… | 7.1% | 同上（不預測個別公司超額報酬） | 30% |
| `us_large_cap` | VOO、SPY、IVV、00646 | 7.5% | JPM 2027 U.S. large cap equities | 15% |
| `us_small_cap` | VB、IWM | 7.5% | 估計：比照美股大型股 | 20% |
| `global_equity` | VT、ACWI | 7.5% | JPM 2027 Global equities | 15% |
| `dev_ex_us_equity` | VEA、VXUS、EFA | 7.5% | 估計：JPM 2027 稱已開發市場首度超越新興市場，取全球股票值 | 16% |
| `em_equity` | VWO、EEM | 7.1% | JPM 2027 Emerging markets equities | 19% |
| `us_treasury_short` | VGSH、SHY | 4.0% | 估計：JPM 2026 U.S. intermediate Treasuries 為 4.0%，短債取不高於此 | 1.5% |
| `us_treasury_intermediate` | VGIT、IEF | 4.6% | JPM 2027 U.S. intermediate Treasuries | 5% |
| `us_treasury_long` | TLT、00679B、00795B | 4.9% | 估計：JPM 2026 U.S. long Treasuries | 13% |
| `us_aggregate_bond` | BND、AGG | 5.0% | 估計：介於中期公債 4.6% 與投資級債 6.1% 之間 | 5.5% |
| `us_ig_corp` | LQD、00751B、00720B | 6.1% | JPM 2027 U.S. investment grade credit | 8% |
| `us_high_yield` | HYG、JNK | 6.7% | JPM 2027 U.S. high yield credit | 9% |
| `gold` | GLD、00635U | 5.5% | JPM 2027 gold | 15% |
| `tw_cash` | 新台幣活存、定存 | 1.7% | 估計：台灣一年期定存利率 | 0.5% |

## 股利預扣稅拖累（`drag`）

台灣居民持有美國掛牌的 ETF，配息會被美國預扣 30% 稅，這部分要從預期報酬扣掉。依各 ETF 的實際殖利率設定，取到小數第 4 位：

| 標的 | `drag` | 理由 |
|---|---|---|
| 美國掛牌的股票類 ETF（VOO、VXUS、VWO、VT…）、REIT、黃金 | `div_yield_ttm × 0.30` | 股利全額預扣 30%；不配息的（如 GLD）殖利率為 0，drag 自然為 0 |
| 美國掛牌的債券類 ETF（`us_treasury_*`、`us_aggregate_bond`、`us_ig_corp`、`us_high_yield`） | `0` | 配息大多屬「利息相關股利」（interest-related dividends），依美國稅法 IRC 871(k) 對非居民免扣繳 |
| 台灣掛牌的海外 ETF（00646、00679B…） | `0` | 已在基金內扣稅，反映在淨值裡 |

- `div_yield_ttm` 來自 `fetch_market.py`：近 12 個月配息 ÷ 最新收盤價
- 債券 ETF 例外：若使用者的券商（部分複委託）實際上仍對債券 ETF 配息扣 30%，改用 `div_yield_ttm × 0.30`
- `div_yield_ttm` 為 null（抓取失敗）時，用下表的預設殖利率（高登方程式的 `div_yield` 也用它；drag = 預設殖利率 × 0.30），並在 `notes` 註明：

| asset_class | 預設殖利率 |
|---|--:|
| `tw_equity` | 3.0% |
| `tw_single_stock` | 2.5% |
| `us_large_cap` | 1.2% |
| `us_small_cap` | 1.5% |
| `global_equity` | 1.8% |
| `dev_ex_us_equity` | 3.0% |
| `em_equity` | 2.8% |

- 每個 drag 都記到 `sources`，例如「VOO drag｜0.32%｜殖利率 1.05% × 30%（fetch_market.py）」

## 通膨率

規則：取 **主計總處 CPI 近 20 年年增率平均** 與 **2%** 兩者中較高者。

已知參考值（查不到時使用）：2008～2020 年平均約 1.0%、1983～2007 年約 1.8%、2021～2022 年約 2.5%、2024 年 2.18%、2025 年 1.66%。近 20 年平均約 1.3～1.5%，因此**預設採 2%**。

## 平均餘命

用 **內政部最新簡易生命表**，查使用者「目前年齡、性別」的**平均餘命**（不是出生時的平均壽命）。
114 年簡易生命表：出生時平均壽命 男 77.93 歲、女 84.75 歲。
查不到該年齡的餘命時：`remaining_life_years = max(出生時平均壽命 − 目前年齡, 15)`，並在報告註明這會低估年長者的餘命。

## 勞保老年年金

- 月領 = max(平均月投保薪資 × 年資 × 0.775% + 3,000, 平均月投保薪資 × 年資 × 1.55%)
- 平均月投保薪資：最高 60 個月平均，**上限 45,800 元**（2026 年；最低 29,500 元）
- 法定請領年齡（`legal_age`）：民國 46 年以前出生 60 歲、47 年 61、48 年 62、49 年 63、50 年 64、**51 年（1962）以後 65 歲**
- 提前請領每年減 4%、延後每年增 4%，最多 5 年
- 年資未滿 15 年只能一次請領（老年一次金），不適用年金公式：設 `mode: "lump_sum"`
- 勞保年金累計 CPI 漲幅達 5% 時會調整，所以視為實質金額

### 勞保一次請領（`mode: "lump_sum"`）

- **一次請領老年給付**（舊制）：限 98 年 1 月 1 日前有勞保年資者選擇，核付後不得變更；條件之一為年資滿 1 年、男性年滿 60 歲（女性 55 歲）退職，並須離職退保
- 給付月數：前 15 年每滿 1 年 1 個月，超過 15 年的部分每滿 1 年 2 個月，**最高 45 個月**（60 歲後繼續工作的年資最多計 5 年，合計最高 50 個月；第一版不計）
- 一次領 = 平均月投保薪資（退保前 3 年平均）× 給付月數
- **老年一次金**：年資未滿 15 年時，每滿 1 年給 1 個月（平均月投保薪資以最高 60 個月平均）
- 98 年後才初次加保、年資滿 15 年者只能月領年金
- 來源：勞動部〈給付標準〉https://www.mol.gov.tw/1607/28162/28472/28474/28478/29187/post 、〈請領條件〉https://www.mol.gov.tw/topic/3078/3301/19471/

## 勞退新制個人專戶

- 雇主強制提繳 6%（`rate: 0.06`）；自提最多 6%，有自提時填 `rate: 0.12`
- 月提繳工資上限 150,000 元
- 60 歲起可請領；年資滿 15 年可月領，否則一次領
- 預設報酬率 `return_nominal: 0.03`（保守值，法定保證收益為二年期定存利率）

## 來源網址

- J.P. Morgan 2027 LTCMA 新聞稿：https://am.jpmorgan.com/us/en/asset-management/institutional/about-us/media/press-releases/jp-morgan-releases-2027-long-term-capital-market-assumptions/
- J.P. Morgan LTCMA 主頁：https://am.jpmorgan.com/us/en/asset-management/institutional/insights/portfolio-insights/ltcma/
- 2026 年勞保、勞退分級表：https://www.ctee.com.tw/news/20260121701103-430104
- 內政部 114 年簡易生命表：https://www.moi.gov.tw/News_Content.aspx?n=4&s=340439
- 2025 年 CPI 年增率：https://wantrich.chinatimes.com/news/20260107900712-420101
