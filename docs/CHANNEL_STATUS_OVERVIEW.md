# WhiteSearch 四通道現況總覽

> **最後更新：2026-09-10（HEAD = `018b022`）**
> 這是一份**索引與現況總表**，不重述細節。每一項都連到記錄它的既有文件。
> 文件內引用的 commit hash 全部以 `git cat-file` 驗證存在。

## 範圍聲明

**這份文件記錄的是四條資料通道在 forward-model 一致性與推論校準上的工程現況。
不構成任何白洞訊號偵測或未偵測的科學宣稱，也不包含任何天文物理結論。**

WhiteSearch 是 candidate ranking engine（候選訊號排序引擎），不是白洞證明器。
除 GW 通道的 GWOSC 路徑外，所有校準數字都來自 mock 模擬資料。

## 文件索引

| 文件 | 範圍 |
|---|---|
| `docs/GW_LIKELIHOOD_STABILIZATION_POSTMORTEM.md` | GW likelihood 前五輪穩定化（`41ce4a7`；taper 引入於 `c7dc6e0`） |
| `docs/BH_RINGDOWN_SBC_COVERAGE_REPORT.md` | `bh_ringdown` 校準驗證 |
| `docs/BOUNCE_PREFLIGHT_AUDIT.md` | `bounce` 逐輪原始證據（Part A–J） |
| `docs/BOUNCE_SBC_COVERAGE_REPORT.md` | `bounce` 工作線敘事總結與最終定性 |
| `docs/RADIO_PREFLIGHT_AUDIT.md` | radio 通道稽核（R.1–R.15） |
| `docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` | xray + image 通道稽核（X.0–X.11） |
| `docs/calibration/*.csv` | 各輪原始數表 |

---

## 一、四通道現況總表

### GW（`bounce`、`bh_ringdown`）

**成熟度：已校準，有已定性但未驗證的殘留。** 唯一有真實資料路徑、且跑過完整規模
SBC/coverage campaign 的通道。

| 已完成的修正 | commit |
|---|---|
| `bh_ringdown` 模型規格對齊 likelihood、先驗映射修正、SBC 判定確定性化 | `01ba53e` |
| `discrete_uniform` → `bilby.DiscreteValues` | `4f8c7c6` |
| `bounce` 四項規格對齊（取樣維度交集、爆發移出 GW、天線投影、`M` 先驗） | `fdf8870` |
| coverage 對未取樣參數改為 fail-closed | `13bb132` |
| 爆發時序重新參數化（`log10_dt_bounce_s`） | `5e2e94c` |
| **taper 依資料 provenance 分流（mock vs 真實）** | `88c5f77` |

診斷輪（未改 production）：`2f2fcbb`（假說一否證）、`372ece1`（假說二否證）、
`55ca4fa`（Fisher 分析）、`794b8cb`（假說三否證＋找到 taper 根因）、
`6325cd1`（簡併脊定性）。驗證輪：`99168ba`、`450de4f`、`d4bce4e`、`8bdc46c`、
`17b2570`、`7bb2655`。

**最終校準（N=100, L=100, nlive=250, 8 維，`17b2570`）**：KS 檢定 8 個參數全部
通過 Bonferroni 校正；`log10_A_bounce` / `log10_dt_bounce_s` / `D_L` / `i`
在三個信賴水準上都無顯著偏差。

**已知限制：**

| 項目 | 分類 |
|---|---|
| `M` 90% coverage 0.75（−3.46σ）、`M`/`eps_f` 的 20-bin χ² 未通過；已定性為等 `f_rd` 簡併脊上的 posterior 偏窄約 1.4 倍 | **需要更多驗證才能判斷**（見待決策 G-1） |
| GW150914 / GW170814 未因 taper 修正重跑；真實路徑不變靠「同一段程式碼＋regression test」保證 | **已定性但選擇不修**（本環境無快取 strain） |
| `bh_ringdown` 的 mock 校準數字未用新 taper 邏輯重跑；推測風險低（無連續到達時間參數）但**未驗證** | **需要更多驗證才能判斷** |
| 樣板接受範圍 `f_rd ∈ [25.0, 1945.6] Hz` 與內積遮罩 `[20, 1700] Hz` 不對稱 | **需要你做一個設計決定**（本輪 100 筆無樣本落入，但那是資料巧合） |

**真實資料路徑：✅ 有。** GWOSC，已在 GW150914 / GW170814 驗證。

---

### radio（`pbh_tunneling`、`magnetar`、`grb_frb`）

**成熟度：mock 路徑內部一致，沒有真實資料路徑；`pbh_tunneling` 在自身先驗下幾乎沒有訊號。**
從未跑過任何 SBC/campaign。

| 已完成的修正 | commit |
|---|---|
| 命名統一（`log10_W_int_ms` / `spectral_index` / `DM`）＋模擬器 fail-closed；接上 `burst_fluence_jy_ms()`；逐模型 `parameter_names` 分支 | `824392e` |
| fluence 公式物理修正（刪除死的 `W_obs_ms`、Δν 改用 CHIME 400 MHz、補上觀測頻寬的 (1+z)）＋跨通道 fail-closed | `d23564f` |

稽核：`919e1d3`。

**已知限制：**

| 項目 | 分類 |
|---|---|
| `pbh_tunneling` 先驗中位 fluence 比 CHIME 門檻低約 6.80 dex；已查證**不是**單位或公式錯誤，主導者是 `log10_eta_r` 的 `uniform(-10, 0)` | **需要你做一個設計決定**（見待決策 R-1） |
| `log10_f_pbh` / `log10_k_tunnel` 是率／壽命參數，單一爆發 likelihood 結構上約束不了 | **已定性但選擇不修**（已移出取樣維度） |
| `grb_frb` 沒有宣告任何寬度或散射參數，仍不可模擬 | **需要你做一個設計決定** |
| `RadioPreprocessor` 沒有任何呼叫端，不在 likelihood 路徑上 | **需要你做一個設計決定** |
| 次像素脈衝可能整個消失（不只是無法解析）；`W_int` 11.88% 低於格寬 | **已定性但選擇不修** |
| Δν 常數與 `chime.yaml` 的一致性靠測試保障，非 import 時讀取 | 刻意設計（同 GW 慣例） |

**真實資料路徑：❌ 無。** `chime` 分支回傳的是**目錄表 DataFrame**，不是動態頻譜；
餵給 likelihood 會 `KeyError: 'data'`。

---

### image / VLBI（`gr_eternal`、`bh_accretion`）

**成熟度：接線乾淨，但先驗與網格解析度不匹配，待重新參數化。**
從未跑過任何 SBC/campaign。

| 已完成的修正 | commit |
|---|---|
| `uv_coverage` 改為從資料取得（缺少則 fail-closed）；closure phase 不再安靜退化；`null` 接上取樣維度；環半徑可表示範圍推導進程式碼 | `018b022` |

稽核：`d4b3178`。

**`gr_eternal` 是所有稽核過的模型-通道組合中唯一 forward-model 結構完整的一個**：
在環可被表示的區域，7 個取樣參數全部同時影響資料與 lnL，無單邊生效。

**已知限制：**

| 項目 | 分類 |
|---|---|
| 環半徑先驗僅 18.29% 落在可表示範圍，80.98% 低於一個像素而產生**精確為零**的影像 | **需要你做一個設計決定**（見待決策 I-2） |
| 推導出的可表示下界 25.77 μas **高於 M87\* 的 19.67 μas**；Sgr A\* 的 25.80 μas 只是剛好擦過 | **需要你做一個設計決定**（同 I-2） |
| `bh_accretion` 無法建立先驗；三個選項的代價已列出 | **需要你做一個設計決定**（見待決策 I-1） |
| 影像網格對薄環的響應**非單調**（3 px 0.1217、4 px 0.0003、5 px 0.9749），是取樣假影 | **已定性但選擇不修** |
| image 通道沒有 preprocess 模組 | **需要你做一個設計決定** |

**真實資料路徑：⚠️ 格式相容但未接。** `EHTLoader` 的 record 可直接餵進
`VisibilityLikelihood`（實測回傳有限值），但 `loader.py` 的 `eht` 分支未實作
（fail-closed，需 `--allow-mock-fallback`）；且該 record **沒有 `closure_phases` 鍵**，
接上後會被新的 fail-closed 檢查擋下（這是刻意的）。

---

### xray

**成熟度：結構性未實作。整條通道目前無法執行。**
沒有任何模型宣告 `channel = "xray"`；沒有做過任何修正。

稽核：`d4b3178`。**本輪明確不在修正範圍內。**

**已知限制：**

| 項目 | 分類 |
|---|---|
| `XRayBurstLikelihood.parameter_names` 是不看 `model_name` 的固定清單，四個被允許的模型**全部**拋 `ValueError`；registry 裡沒有任何模型宣告 `log10_fluence_erg_cm2` | **需要你做一個設計決定**（見待決策 X-1） |
| `log10_eta_gamma` 在 likelihood 清單裡但模擬器從未讀取 | **需要你做一個設計決定**（同 X-1） |
| `pbh_tunneling` 在 xray 上 10 個參數全部零變化（模擬器 `params.get` 吃預設值） | **需要你做一個設計決定**（同 X-1） |
| `log10_fluence_jy_ms`（Jy·ms）與 `log10_fluence_erg_cm2`（erg/cm²）是不同物理量，缺轉換 | **需要你做一個設計決定**（同 X-1） |
| `T90` 先驗 49.96% 短於一個時間格、16.30% 長於整段資料、24.60% 被 t=0 截斷 | **需要你做一個設計決定** |
| `chandra.py` / `xmm.py` 存在且被匯出，但 `loader.py` 沒有分支 | **需要你做一個設計決定** |
| `XRayPreprocessor` 沒有任何呼叫端 | **需要你做一個設計決定** |

**真實資料路徑：❌ 無。** `heasarc` 分支未實作（fail-closed 正確），
且 `HEASARCLoader` 與 `XRayBurstLikelihood` 需要的 counts 陣列之間沒有轉接層。

---

## 二、待決策事項清單

格式：**編號｜通道｜要決定什麼｜分類｜細節出處**

### G-1｜GW｜`bounce` 的 `M`/`eps_f` 簡併脊殘留要不要驗證

- **要決定的**：是否投入一輪對照 campaign，驗證提高 `nlive`（或加大 walk 步數）
  能否消除沿等 `f_rd` 脊方向的 posterior 偏窄。
- **分類**：**需要更多驗證才能判斷**。已排除的：頻帶邊界效應；已確立的：
  `f_rd` 本身校準完美（90% coverage 0.900，偏差 +0.00σ）。
- **細節**：`docs/BOUNCE_PREFLIGHT_AUDIT.md` Part J、
  `docs/BOUNCE_SBC_COVERAGE_REPORT.md`「已知限制」第 1 項。

### R-1｜radio｜`pbh_tunneling` 先驗尾端集中，SBC 要不要分層抽樣

- **要決定的**：跑 radio SBC 時，是否需要分層抽樣（或調整先驗），
  以免絕大多數注入落在無訊號區、rank 均勻只是因為 posterior = prior。
- **分類**：**需要你做一個設計決定**（改先驗＝改模型；分層抽樣＝改驗證程序，
  兩者性質不同）。
- **相關事實**（修正後）：SNR ≥ 1 的比例 **2.210%**、SNR ≥ 8 的比例 **0.785%**
  （修正前分別是 1.355% / 0.465%）。
- **細節**：`docs/RADIO_PREFLIGHT_AUDIT.md` R.6、R.13、R.15.5–R.15.6。

### I-1｜image｜`bh_accretion` 要走 A / B / C 哪一條

- **要決定的**：
  - **A** 交集分支 `[M, a_star, D_L, i]` — 三個幾何參數落到模擬器預設（R.4 靜默預設模式），
    `bh_accretion` 變成「凍結三參數的 `gr_eternal`」，不是物理上不同的對立假說。
  - **B** 讓模型補宣告三個幾何參數 — 與 `gr_eternal` 完全相同，ln BF 恆等於 0。
  - **C** 擴充 `ImageShadowSimulator`，讓 `log10_mdot_edd` / `jet_power_frac`
    真正驅動亮度分布 — 唯一物理上有意義，但是實質建模工作。
- **分類**：**需要你做一個設計決定**。
- **細節**：`docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` X.11.4。

### I-2｜image｜環半徑要不要重新參數化

- **要決定的**：是否把先驗從 `(M, D_L)` 改到資料真正約束的比值上
  （`r ∝ M/D_L`），規模比照 `bounce` 的 B3-2 爆發時序重新參數化。
- **分類**：**需要你做一個設計決定**。已確立的事實：獨立收窄邊際先驗**做不到**——
  要 ~100% 可表示需要 `dex(M) + dex(D_L) ≤ 1.204`（質量與距離各只能跨 4 倍），
  而 M87\* 與 Sgr A\* 質量差 3.19 dex、距離差 3.31 dex 卻環半徑相近。
- **細節**：`docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` X.6、X.11.1。

### X-1｜xray｜整條通道要不要投入實作

- **要決定的**：(a) 是否投入；若投入，(b) `XRayBurstLikelihood.parameter_names`
  要怎麼設計逐模型分支（比照 `RadioBurstLikelihood` 在 `824392e` 的做法），
  (c) Jy·ms → erg/cm² 的 fluence 單位／物理轉換要放在哪一層，
  (d) 是否要有一個原生 `channel = "xray"` 的模型。
- **分類**：**需要你做一個設計決定**。
- **細節**：`docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` X.1–X.3、X.9。

---

## 三、資源事實（不含建議，僅供估算）

> 這一節**只列已量測或可直接推算的事實**，不替任何選項背書、不排優先順序。

### G-1 若要驗證簡併脊假說

- **已量測的基準**：上一輪 N=100、L=100、`nlive=250`、8 維、
  `likelihood_mode=full`、4 shard × 25 平行，**實際 wall clock 5.4 小時**
  （各 shard 16434 / 19279 / 16434 / 18472 s），4 核心，0 失敗（`17b2570`）。
- **`nlive` 加倍的成本**：dynesty 的 likelihood 呼叫數大致隨 `nlive` 線性成長，
  所以 `nlive=500` 的同規模對照**至少**是 5.4 h 的兩倍，即 **≳ 11 小時**；
  實務上可能更高（每次迭代的 bound 更新成本也隨 `nlive` 增加）。
  **這是外推，不是量測。**
- **已知的估算風險**：上一輪用 8 筆注入外推得到 2.2–2.8 h，實際 5.4 h，
  **外推偏樂觀約 2 倍**。
- **環境風險**：本容器曾多次在 10–20 分鐘內被回收；上一輪靠
  `resume=True` + `check_point_delta_t=90` 才完成，該次連續運行 5.4 h 未被回收。

### R-1 若要跑 radio SBC

- **沒有任何 radio campaign 的耗時量測**，無法給出可靠估計。
- **可直接推算的**：以修正後 SNR ≥ 8 的比例 0.785% 計，N=100 的 campaign
  期望只有 **約 0.8 筆**注入帶有可偵測訊號；N=1000 約 8 筆。

### I-2 若要跑 image SBC

- **沒有任何 image campaign 的耗時量測。**
- **可直接推算的**：目前先驗下 **18.29%** 的注入的環可被表示，
  N=100 期望約 18 筆有訊號、約 81 筆的影像精確為零。

### X-1 若要實作 xray

- **目前 0% 可執行**：四個被允許的模型全部無法建立先驗，
  所以「要花多久」在做出 X-1 的設計決定之前無法估算。

---

## 四、跨通道橫向事實

- **`effective_parameter_names()` 的交集機制在四條通道都正確 fail-closed**
  （`fdf8870` 引入）：該擋的都擋下來了（`grb_frb` on radio、`bh_accretion` 與
  `null` on image、四個模型 on xray），該過的也過了。
- **`params.get(key, default)` 的靜默預設是重複出現最多次的失效模式**：
  GW（`3.086e22` vs `MPC_M`）、radio（R.4，三組命名）、xray（X.2，十個參數）、
  image（`bh_accretion` 的三個幾何參數）都命中過。radio 已改為 fail-closed
  （`824392e`），其餘未改。
- **「算了但沒用」的死程式碼**出現兩次：`bounce` 的爆發成分（B.3）、
  radio `burst_fluence_jy_ms()` 的 `W_obs_ms`（R.15.1，判定為應刪除而非應接線）。
- **「先驗跨度遠大於儀器可表示範圍」出現四次**，嚴重度遞增：
  GW `M`（B.5，14.06% 被拒）→ radio `W_int`（R.7，11.88% 次格寬）→
  xray `T90`（X.3，49.96% 次格寬）→ image 環半徑（X.6，**80.98%** 次像素）。
- **preprocess 模組的接線狀況不一致**：GW 有接（`gw_observation.py`），
  radio 與 xray 的 preprocessor 沒有任何呼叫端，image 根本沒有模組。
