# retirement-target

Claude Code skill：計算退休金目標、退休缺口與每月需再多存多少（台灣、新台幣）。

輸入你的年齡、預計退休年齡、每月生活費、持股和勞保勞退狀況，就能算出你需要多少退休金、現在的存法夠不夠、每月還要多存多少，以及錢撐到 95 歲的機率。

## 方法出處

本 skill 的方法大量參考威廉．伯恩斯坦（William J. Bernstein）的《投資金律》（*The Four Pillars of Investing*），例如：

- 股票預期報酬用高登方程式估「基本報酬率」（股息殖利率＋實質股息成長率），不把估值膨脹的投機報酬算進去
- 退休資產分成「負債配合組合」（生活費安全池）與「風險組合」（成長風險池），也就是報告中的雙水庫策略

但並非完全照書：另外加入台灣的勞保、勞退與公保規則、蒙地卡羅模擬，並實際比較雙水庫與每年再平衡的成功率。試算結果是本 skill 自己的計算，不代表原書作者的看法。

## 範例結果

以下是 `examples/個人資料_小資上班族.txt` 產生的報告節錄。

**使用者基本資料**：skill 從個人資料整理出的現況

![使用者基本資料](docs/images/profile.png)

**結論**：退休金目標、預估累積、缺口，以及每月要多存多少

![結論](docs/images/conclusion.png)

**錢能撐到幾歲**：蒙地卡羅模擬 10,000 種市場走勢後，各年齡還有錢的機率

![錢能撐到幾歲](docs/images/survival.png)

**每年提領金額**：退休後每年要從投資組合提領多少，以及折算成今天的現值（65 歲起勞保年金開始給付，提領金額隨之下降）

![每年提領](docs/images/withdrawal.png)

**各資產實質報酬**：股票用高登方程式、債券與現金用目前殖利率，換算扣除通膨後的實質報酬

![各資產實質報酬](docs/images/asset-returns.png)

## 安裝

1. 將本 repo clone 下來，在 repo 根目錄啟動 Claude Code（skill 位於 `.claude/skills/retirement-target`）。
2. （選用）複製 `.env.sample` 為 `.env` 並填入 `FUGLE_API_KEY`，用於抓取台股資料；未設定時會自動跳過 Fugle。
3. 安裝 Python 相依套件：`pip install numpy pandas requests yfinance fugle-marketdata pytest`

## 使用

在 Claude Code 中輸入「算退休金」「退休試算」等，或直接提供個人資料檔。

範本請見 `examples/`：

- `examples/個人資料_公務員.txt`：公務人員（公保＋退撫新制）
- `examples/個人資料_小資上班族.txt`：私人企業上班族（勞保＋勞退新制）

勞保投保年資、平均月投保薪資、勞退個人專戶累積金額等資料，可登入[勞保局 e 化服務系統](https://edesk.bli.gov.tw/)查詢，再照範本格式填入。

## 結構

```
.claude/skills/retirement-target/
├── SKILL.md                 # skill 流程說明
├── reference/
│   ├── input_schema.md      # calc.py 輸入格式
│   └── defaults.md          # 預設值、勞保勞退規則
└── scripts/
    ├── fetch_market.py      # 抓市場資料
    ├── calc.py              # 計算與產生報告
    └── test_calc.py         # 測試
```
