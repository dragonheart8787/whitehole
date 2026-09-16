# WhiteSearch 四通道現況總覽

> **最後更新：2026-09-11（前次修正 HEAD = `c292336`）**
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
| `docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` | xray + image 通道稽核（X.0–X.12） |
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
| `M` 90% coverage 0.75（−3.46σ）、`M`/`eps_f` 的 20-bin χ² 未通過；已定性為等 `f_rd` 簡併脊上的 posterior 偏窄約 1.4 倍 | **已定性、已評估兩種加強手段、決定不再繼續投入**（見待決策 G-1） |
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

**成熟度：模型層與先驗層已到位，但可見度取樣有一個阻斷性錯誤。**
從未跑過任何 SBC/campaign，**目前也還不該跑**（見 I-4）。

| 已完成的修正 | commit |
|---|---|
| `uv_coverage` 改為從資料取得（缺少則 fail-closed）；closure phase 不再安靜退化；`null` 接上取樣維度；環半徑可表示範圍推導進程式碼 | `018b022` |
| `D_L` 改為逐目標已知常數（`target` 欄位、metadata→context、fail-closed）；`gr_eternal` 取樣維度 7→6；M 先驗改為逐目標、出處進註解；`bh_accretion` 實作為真正不同的假說（吸積率驅動亮度+厚度、噴流足點打破方位角對稱）；四個幾何參數不再走靜默預設 | 本輪 |

稽核：`d4b3178`、X.12。

**已知限制：**

| 項目 | 分類 |
|---|---|
| `_compute_visibilities()` 的 Gλ→rad⁻¹ 換算多乘了波長，16 條基線全部落在 uv 原點的格子裡，**可見度只帶總流量、不帶結構**；`position_angle` 因此改影像 ~100% 但只改可見度 4.14e−4 | **阻斷性，已回報未修**（見待決策 I-4） |
| shipped 的 200 μas / 128 px 網格只表示得了 M87\* 質量先驗的 **13.29%**、Sgr A\* 的 **52.39%**；下界 25.77 μas 高於 M87\* 的 19.66 μas | **需要你做一個設計決定**（見待決策 I-3） |
| `cli.py` 的預設 `n_pixels: 64` 與 `eht.yaml` 的 `128` 不一致；64 px 下兩個目標都是 0% 可表示 | **需要你做一個設計決定**（同 I-3） |
| 影像網格對薄環的響應**非單調**（3 px 0.1217、4 px 0.0003、5 px 0.9749），是取樣假影 | **已定性但選擇不修** |
| `gr_eternal.summary_stats` 走 `photon_ring_radius_m()`（Chan+2015 外光子軌道），模擬器走 `_shadow_radius_muas()`（Bardeen 近似），兩者在 a\* ≠ 0 時不同 | **新發現，已記錄未修** |
| image 通道沒有 preprocess 模組 | **需要你做一個設計決定** |

**已解除的限制：** 環半徑先驗的**先驗側**問題已解決——固定 `D_L` 後環半徑的先驗
展寬從 7.301 dex 降到 0.523 dex（M87\*）/ 0.155 dex（Sgr A\*），在可表示的網格上
先驗抽樣不再產生任何一張精確為零的影像。剩下的是網格側，見 I-3。

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

### G-1｜GW｜`bounce` 的 `M`/`eps_f` 簡併脊殘留 —— **已收尾**

- **分類**：**已定性、已評估兩種加強手段、決定不再繼續投入。**
  這條調查線有明確結論，不是「未解決」也不是「待驗證」。
- **已定性的內容**：殘留的 under-coverage 是等 `f_rd` 簡併脊上的 posterior
  偏窄（`M` 的區間只有應有寬度的 0.699 倍，需要約 **43%** 的增寬）。
  資料真正約束的組合 `f_rd` 本身**校準完美**（90% coverage 0.900，偏差 +0.00σ、
  rank KS p = 0.948），`M` 與 `eps_f` 只是那條脊在座標軸上的投影。
- **已排除的手段一：頻帶邊界效應**（Part J）。100 筆真值的 `f_rd` 離 likelihood
  接受下界最近也有 0.139 dex，分層後差異僅 +1.16σ（Fisher p = 0.356）、
  三分位不單調、逐筆相關性 p = 0.15–0.83 全不顯著。
- **已排除的手段二：加大隨機游走鏈長**（`65f5a10`，Part K）。12 筆定向 pilot，
  `nact` 從 2 提到 8（平均接受步數 4 → 16）：

  | | `base` | `nact8` |
  |---|---|---|
  | `in90_M`（9 筆配對） | 1/9 | 2/9 |
  | `w90(M)` 中位數增幅 | — | **+3.6%**（Wilcoxon p = 0.0273） |
  | ncall 中位數 | 4.63×10⁵ | **1.91×10⁶（4.26×）** |
  | 耗時中位數 | 610 s | **3618 s** |

  增寬方向確實是系統性的（p = 0.0273），但 **3.6% 的效應量遠不及 43% 的缺口**，
  而代價是 4.26 倍。`in90_M` 由 1/9 變 2/9，在此樣本數下不可與雜訊區分。
- **旁證：加長鏈長讓最難取樣的樣本明顯惡化。** `nact8` 只完成 9/12——
  `20260944` 與 `20260956` 分別跑到 **4 小時 39 分**與 **6 小時 01 分**、
  效率掉到 0.1%、`nc` 長期卡在 maxmcmc 上限 5001 仍未收斂而被中止
  （`20260973` 未開始）。同樣兩筆在 `base` 下是分鐘級。
- **最終判斷：不建議繼續投入資源驗證 `nlive` 加倍。** 理由是
  `nact` 是**更直接**針對「脊上探索能力」的旋鈕（它直接決定隨機游走的鏈長），
  連它都沒有帶來與代價相稱的改善；`nlive` 加倍是**更間接**的手段
  （增加活點數不直接改善單一鏈在強相關方向上的移動能力），成本卻更高
  （§三 的外推是 ≳ 11 小時）。
  **要說清楚的是：`nlive` 加倍從未被實測**（原計畫的對照組在縮減 pilot 範圍時
  被移除），所以「預期不會更有效」是根據機制的推論，不是量測結果。
- **細節**：`docs/BOUNCE_PREFLIGHT_AUDIT.md` Part J、**Part K**、
  `docs/BOUNCE_SBC_COVERAGE_REPORT.md`「已知限制」第 1 項、
  `docs/calibration/bounce_ridge_sampler_pilot.csv`。

### R-1｜radio｜`pbh_tunneling` 先驗尾端集中，SBC 要不要分層抽樣

- **要決定的**：跑 radio SBC 時，是否需要分層抽樣（或調整先驗），
  以免絕大多數注入落在無訊號區、rank 均勻只是因為 posterior = prior。
- **分類**：**需要你做一個設計決定**（改先驗＝改模型；分層抽樣＝改驗證程序，
  兩者性質不同）。
- **相關事實**（修正後）：SNR ≥ 1 的比例 **2.210%**、SNR ≥ 8 的比例 **0.785%**
  （修正前分別是 1.355% / 0.465%）。
- **細節**：`docs/RADIO_PREFLIGHT_AUDIT.md` R.6、R.13、R.15.5–R.15.6。

### I-1｜image｜`bh_accretion` 要走 A / B / C 哪一條 —— **已實作（選 C）**

- **決定**：走 **C**（擴充 `ImageShadowSimulator`），作法比照 GW 通道
  `bh_ringdown` 的 `log10_A`——宣告一條唯象標度律，不做第一原理 GRMHD。
- **實作內容**：`log10_mdot_edd` **同時**決定峰值面亮度與環厚度
  （輻射低效吸積流在低吸積率下幾何厚、趨近 Eddington 轉薄），
  `jet_power_frac` 在投影自轉軸方向的環段上加亮（足點方位角半寬 ~34°，
  `f_jet = 1` 時 4 倍對比）。兩條標度律只寫在
  `ring_emission_from_params()` 一處，simulator 與 likelihood 共用。
- **取樣維度**：`M, a_star, i, position_angle, log10_mdot_edd, jet_power_frac`（6）。
  與 `gr_eternal` 的交集只有四個幾何參數。
- **與 `gr_eternal` 的實測差異**（相同真值、同環半徑／厚度／峰值亮度）：
  影像最大相對差 **89.8%**；對 180° 旋轉的不對稱度 **0.473** vs **0.0**（精確）；
  同一筆資料上的 lnL 差 **6.32e6**。
- **未達成的部分（如實記錄）**：逐參數活性測試要求每個宣告的參數都同時改變
  simulator 輸出與 lnL。六個參數中**五個達成**，`position_angle` 沒有——
  它改影像約 100%，改可見度只有 4.14e−4。**原因不在這個模型，在 I-4。**
- **細節**：`docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` X.12.3。

### I-2｜image｜環半徑要不要重新參數化 —— **已實作（`D_L` context 化）**

- **決定**：不做比值重新參數化，改成**把 `D_L` 當成逐目標已知常數**。
  VLBI 影像約束的是角半徑，距離不是這條通道量的東西；把它從取樣向量拿掉
  之後，`r ∝ M / D_L` 的不可辨識方向就不存在了。
- **機制**：比照 GW 通道 `GWLikelihood._parse_data` 的
  `meta.get(..., context.get(...))`（observation metadata 優先、context 次之），
  唯一差別是**沒有第三層預設值**——缺 `target` 直接 raise。
  新增 `target` 欄位而非重用 `EHTLoader` 既有的 `source`
  （後者在 GW/radio 上代表資料來源證跡，語意衝突）。
- **常數與出處**：M87\* 16.8 Mpc（EHT 2019 ApJL 875 L6）、
  Sgr A\* 0.008178 Mpc（GRAVITY 2019 A&A 625 L10）；
  質量先驗 `[3.0e9, 1.0e10]` 與 `[3.5e6, 5.0e6]`，寬度照文獻上的**獨立測量**訂
  （Gebhardt+2011 / Walsh+2013；Do+2019），**不是照網格可表示範圍反推的**。
- **成果**：環半徑的先驗展寬 7.301 dex → **0.523 dex**（M87\*）/
  **0.155 dex**（Sgr A\*）；在可表示的網格上，先驗抽樣不再產生任何一張
  精確為零的影像（原本 80.98%）。
- **未達成的部分（如實記錄）**：可表示比例**沒有接近 100%**——
  shipped 網格下 M87\* **13.29%**、Sgr A\* **52.39%**。這是網格的限制，
  不是先驗的，所以**沒有為了讓數字好看而收窄先驗**。轉為 I-3。
- **細節**：`docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` X.12.1、X.12.2。

### I-3｜image｜成像網格要不要重新設定（FoV 或 `n_pixels`）

- **要決定的**：`configs/instruments/eht.yaml` 的 `imaging.fov_muas = 200.0`
  對一個 20–31 μas 的環而言約大了一個數量級；可表示下界
  `8.245 px × pixel` 因此落在 25.77 μas，高於 M87\* 的環。兩條可行的槓桿：

  | 槓桿 | M87\* | Sgr A\* |
  |---|---|---|
  | 固定 FoV = 200 μas，需要的 `n_pixels` | ≥ **364** | ≥ **152** |
  | 固定 `n_pixels` = 128，可用的 FoV 半寬 | **[30.24, 70.42] μas** | **[31.06, 168.76] μas** |

  **FoV = 50 μas / 128 px 下兩個目標都是 100% 可表示**，且先驗中最大的環
  （31.06 μas）仍安穩落在視野內。順帶要決定 `cli.py` 的 `n_pixels: 64`
  與 yaml 的 `128` 誰是對的（64 px 下兩個目標都是 0%）。
- **分類**：**需要你做一個設計決定**。推導已進程式碼
  （`mass_representable_range_msun` / `mass_prior_representable_fraction` /
  `required_n_pixels_for_prior` / `required_fov_muas_for_prior`），
  數字全部由測試鎖住；**沒有自行改動 shipped 設定**。
- **細節**：`docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` X.12.2。

### I-4｜image｜`_compute_visibilities()` 的 Gλ → rad⁻¹ 換算 —— **阻斷性**

- **問題**：`uv_rad = uv_coverage * 1e9 * wavelength_m`。以 Gλ 為單位的基線
  本來就是「每弧度幾個週期」，乘上波長得到的是**基線物理長度（公尺）**。
  230 GHz 下等於把每條基線縮小 767 倍，16 條全部落進 uv 原點所在的那一個
  FFT 格子（間距 2.06e9 rad⁻¹，落點 < 8.5e6）。
- **後果（已量測）**：模型可見度在**每條基線上都等於影像總流量**
  （165.68 vs 總流量 165.70 Jy，相對散布 1.3e−3）。正確取樣下同一張影像的
  `|V|` 應該散布在 39.7–163.7 Jy。`position_angle` 0.7→1.9 的
  `max|ΔV|/max|V|`：現行 **4.14e−4**，正確取樣 **0.571**（逐基線 33–2385 σ）。
- **意義**：**image 通道的可見度目前不帶任何影像結構資訊，只帶總流量。**
  其他五個參數之所以還「活著」，是因為它們都會改變總流量，
  不是因為通道量到了環的形狀。
- **影響範圍**：同樣的換算也在 `dataio/eht.py::_mock_eht_data()`，
  所以不是「改一行」；修正會改變 image 通道**每一個**既有數字。
- **分類**：**阻斷性問題，依規則回報後停下，未在本輪修正。**
  在這個修好之前，image 通道的 SBC 不會是有意義的校準。
- **細節**：`docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` X.12.5。

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

### G-1 若要驗證簡併脊假說（依 §二 G-1 的最終判斷，已決定不再執行；數字保留供日後參考）

- **已量測的基準**：上一輪 N=100、L=100、`nlive=250`、8 維、
  `likelihood_mode=full`、4 shard × 25 平行，**實際 wall clock 5.4 小時**
  （各 shard 16434 / 19279 / 16434 / 18472 s），4 核心，0 失敗（`17b2570`）。
- **`nlive` 加倍的成本**：dynesty 的 likelihood 呼叫數大致隨 `nlive` 線性成長，
  所以 `nlive=500` 的同規模對照**至少**是 5.4 h 的兩倍，即 **≳ 11 小時**；
  實務上可能更高（每次迭代的 bound 更新成本也隨 `nlive` 增加）。
  **這是外推，不是量測。**
- **另一個旋鈕的實測，可作為代價遞增的參考（不是 `nlive` 本身的驗證）**：
  Part K 的 `nact=8` pilot 實測 ncall 中位數上升 **4.26 倍**、耗時中位數
  610 s → **3618 s**，且 12 筆裡有 2 筆在 4.6–6.0 小時後仍未收斂。
  **這量的是鏈長而非活點數**，不能直接換算成 `nlive` 加倍的成本；
  它的用處是顯示 **dynesty 在這個簡併幾何上，加強取樣的代價成長得比線性更快**，
  最難的樣本尤其明顯。依 G-1 的最終判斷，這項對照**不再規劃執行**。
- **已知的估算風險**：上一輪用 8 筆注入外推得到 2.2–2.8 h，實際 5.4 h，
  **外推偏樂觀約 2 倍**。
- **環境風險**：本容器曾多次在 10–20 分鐘內被回收；上一輪靠
  `resume=True` + `check_point_delta_t=90` 才完成，該次連續運行 5.4 h 未被回收。

### R-1 若要跑 radio SBC

- **沒有任何 radio campaign 的耗時量測**，無法給出可靠估計。
- **可直接推算的**：以修正後 SNR ≥ 8 的比例 0.785% 計，N=100 的 campaign
  期望只有 **約 0.8 筆**注入帶有可偵測訊號；N=1000 約 8 筆。

### I-3 / I-4 若要跑 image SBC

- **沒有任何 image campaign 的耗時量測。**
- **先決條件**：**I-4 未修之前不建議跑**——可見度只帶總流量，
  跑出來的 SBC 校準的是「總流量模型」，不是影像模型。
- **可直接推算的**（先驗側已修好之後）：shipped 的 200 μas / 128 px 網格下，
  M87\* 有 **13.29%** 的注入環可被表示（N=100 約 13 筆）、
  Sgr A\* **52.39%**（約 52 筆）。若採 FoV = 50 μas / 128 px，兩者都是 100%。

### X-1 若要實作 xray

- **目前 0% 可執行**：四個被允許的模型全部無法建立先驗，
  所以「要花多久」在做出 X-1 的設計決定之前無法估算。

---

## 四、跨通道橫向事實

- **`effective_parameter_names()` 的交集機制在四條通道都正確 fail-closed**
  （`fdf8870` 引入）：該擋的都擋下來了（`grb_frb` on radio、`bh_accretion` 與
  `null` on image、四個模型 on xray），該過的也過了。
- **第三方設定「看似傳入、實際無效」出現兩次，都在 bilby 2.8.2 的 dynesty 介面**
  （Part K.2）：`sample='rwalk'` 下 `walks` 完全不被讀取（鏈長由 `nact` 決定），
  而使用者傳入的 `maxcall` 會被 bilby 覆蓋成自己的 checkpoint 分塊大小。
  **兩項均已修正（見 Part L）**：`DEFAULT_DYNESTY_KWARGS` 的
  `walks: 32` 換成 `nact: 2`（bilby 自己的預設值，所以行為不變、既有校準結果
  仍可重現），逾時改用不依賴 bilby 的 `_BudgetGuard`（在 likelihood 內部檢查
  牆鐘與呼叫數，超出即拋 `SamplingBudgetExceeded`）。
  教訓已寫成測試：驗證設定是否生效要看**行為**，不能看「鍵是否出現在 kwargs 裡」
  ——`AcceptanceTrackingRWalk` 物件**確實有** `walks` 屬性，值卻恆為 dynesty 的
  預設 25，與傳入值無關。修正 commit 是 **`c292336`**；該 commit message 當時寫
  「完整測試結果另行回報」，現已確認：**完整套件 294 passed, 1 skipped,
  1 deselected**（基準 `d4b3178` 是 281 passed，差額 +13 即新增的 13 項測試），
  **無非預期 regression**。
- **`params.get(key, default)` 的靜默預設是重複出現最多次的失效模式**：
  GW（`3.086e22` vs `MPC_M`）、radio（R.4，三組命名）、xray（X.2，十個參數）、
  image（`bh_accretion` 的三個幾何參數）都命中過。radio 已改為 fail-closed
  （`824392e`）；image 的四個幾何參數也已改為明確要求（X.12.3）；xray 未改。
- **「算了但沒用」的死程式碼**出現兩次：`bounce` 的爆發成分（B.3）、
  radio `burst_fluence_jy_ms()` 的 `W_obs_ms`（R.15.1，判定為應刪除而非應接線）。
- **「先驗跨度遠大於儀器可表示範圍」出現四次**，嚴重度遞增：
  GW `M`（B.5，14.06% 被拒）→ radio `W_int`（R.7，11.88% 次格寬）→
  xray `T90`（X.3，49.96% 次格寬）→ image 環半徑（X.6，**80.98%** 次像素）。
  image 的這一項已處理：把 `D_L` 變成逐目標已知常數之後先驗側歸零，
  殘留的 13.29% / 52.39% 是**網格**表示不了先驗，不是先驗太寬（X.12.2、I-3）。
- **「單位換算錯誤」是第三次出現的類型**：GW 的 `3.086e22` vs `MPC_M`
  （相對差 1.0449e−4，偏了 D_L）、radio 的 Jy·ms → erg/cm²（X-1(c) 仍未決），
  以及 image 的 Gλ → rad⁻¹（**767 倍**，I-4）。前兩者量級小或已知未接，
  image 這一次的後果是整條通道的可見度失去結構資訊。
- **preprocess 模組的接線狀況不一致**：GW 有接（`gw_observation.py`），
  radio 與 xray 的 preprocessor 沒有任何呼叫端，image 根本沒有模組。
