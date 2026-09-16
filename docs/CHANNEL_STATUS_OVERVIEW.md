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
| `docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` | xray + image 通道稽核（X.0–X.14） |
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

**成熟度：forward model 與先驗皆已到位，但 SBC 在這個環境跑不完。**
仍**沒有任何 coverage 證據**——第一次 campaign 嘗試因取樣成本受阻，見 I-6。

| 已完成的修正 | commit |
|---|---|
| `uv_coverage` 改為從資料取得（缺少則 fail-closed）；closure phase 不再安靜退化；`null` 接上取樣維度；環半徑可表示範圍推導進程式碼 | `018b022` |
| `D_L` 改為逐目標已知常數（`target` 欄位、metadata→context、fail-closed）；`gr_eternal` 取樣維度 7→6；M 先驗改為逐目標、出處進註解；`bh_accretion` 實作為真正不同的假說；四個幾何參數不再走靜默預設 | `ad4f43a` |
| uv 基線尺度修正（Gλ 即角頻率，不再乘波長）＋改用直接 DFT 取樣；mock 路徑同步修正；成像 FoV 200→50 μas；`n_pixels` 改為單一讀取點；Kerr 陰影半徑統一為對精確臨界曲線的擬合；`bh_accretion` 亮度歸一化修正 100 倍 | 本輪 |

稽核：`d4b3178`、X.12、X.13。

**修正後的狀態（量測值）：**

| 指標 | 值 |
|---|---|
| 先驗抽樣的環可被表示 | **100%**（兩模型 × 兩目標；曾是 18.29% → 13.29%） |
| 可見度取樣對解析 Hankel 變換的偏差 | **0.76%**（曾是「每條基線都等於總流量」） |
| 宣告參數全部同時影響資料與 lnL | **兩個模型皆全部**（`position_angle` 從 4.14e−4 變成 0.392） |
| 先驗抽樣 SNR ≥ 8 的比例 | `gr_eternal` **81.2%**、`bh_accretion` **66.0%**（GW 通道是 0.785%） |
| 總流量真值是否在先驗內 | 是，兩目標都在 `log10_mdot_edd ≈ −3` |

**已知限制：**

| 項目 | 分類 |
|---|---|
| **SBC 跑不完**：先驗預測 SNR 橫跨 0.2–3.2e5，巢狀取樣成本隨資訊量爆炸。實測單筆 SNR 158 要 3292 s / 1.3e6 次 likelihood 呼叫；N=100 估計約 145 小時 | **阻塞性，已回報未修**（見待決策 I-6） |
| `_compute_closure_phases()` 的三元組**不閉合**（`_default_eht_uv()` 是一串基線不是台站陣列），所以它是自洽的相位組合、不是具增益不變性的 closure phase。不造成推論偏差，但名不副實 | **已定性，回報未修**（見待決策 I-5） |
| **從未跑過任何 SBC**——上表是「條件已具備」，不是「已校準」；第一次嘗試已執行但未完成（I-6） | 待執行 |
| 影像網格對薄環的響應**非單調**（3 px 0.1217、4 px 0.0003、5 px 0.9749），是取樣假影；現行網格下兩個目標都遠在下界之上，實務上不觸發 | **已定性但選擇不修** |
| image 通道沒有 preprocess 模組 | **需要你做一個設計決定** |
| `null` 模型無法注入到 image 通道（模擬器沒有 null 分支，`M` 缺失即 raise）。這是 `cda3003` 之前就有的行為，不是本輪引入 | **已定性，未修** |

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

### I-1｜image｜`bh_accretion` 要走 A / B / C 哪一條 —— **已完成**

- **決定**：走 **C**（擴充 `ImageShadowSimulator`），比照 GW 通道
  `bh_ringdown` 的 `log10_A`——宣告唯象標度律，不做第一原理 GRMHD。
- `log10_mdot_edd` **同時**決定峰值亮度與環厚度；`jet_power_frac` 在投影
  自轉軸方向加亮環段。兩條標度律只寫在 `ring_emission_from_params()` 一處。
- **逐參數活性：六個宣告參數全部同時改變 simulator 輸出與 lnL。**
  上一輪 `position_angle` 未達成（4.14e−4），原因是 I-4，已隨之解決（0.392）。
- 與 `gr_eternal` 在相同真值下的差異：影像最大相對差 89.8%、
  180° 旋轉不對稱度 0.473 vs 0.0、同一筆資料的 lnL 差 387。
- **後續修正**：亮度歸一化 `LOG10_I0_EDD` 原為 2.0，使整個先驗的總流量
  （1.87–2.4e4 Jy）都在 M87\* 實測 0.5–1.2 Jy 之上——**真值落在先驗外**。
  這是上一輪引入的算術錯誤（環面積算錯，且錨定峰值而非積分），
  已改為 0.0，先驗改跨 0.019–235 Jy。詳見 X.13.5。
- **細節**：`docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` X.12.3、X.13.5。

### I-2｜image｜環半徑要不要重新參數化 —— **已完成**

- **決定**：不做比值重新參數化，改成把 `D_L` 當成逐目標已知常數。
- 環半徑的先驗展寬 7.301 dex → 0.523 dex（M87\*）/ 0.155 dex（Sgr A\*）。
- 上一輪殘留的「可表示比例未達 100%」已由 I-3 解決（見下）。
- **細節**：`docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` X.12.1、X.12.2。

### I-3｜image｜成像網格要不要重新設定 —— **已完成**

- **決定**：`configs/instruments/eht.yaml` 的 `imaging.fov_muas` 200.0 → **50.0**
  （`n_pixels` 維持 128）。200 μas 半寬對 20–25 μas 的環大了約一個數量級。

  | 網格 | 像素 | 可表示下界 | M87\* | Sgr A\* |
  |---|---|---|---|---|
  | 200 μas / 64 px（cli.py 舊值） | 6.25 μas | 51.53 μas | 0% | 0% |
  | 200 μas / 128 px（yaml 舊值） | 3.125 μas | 25.77 μas | 7.92% | 34.23% |
  | **50 μas / 128 px** | **0.781 μas** | **6.44 μas** | **100%** | **100%** |

- **先驗完全沒有動**，仍由文獻上的獨立質量測量決定；改的是分析設定。
- `n_pixels` 不一致同時處理：新增 `dataio.eht.eht_imaging_config()` 作為唯一
  讀取點，`cli.py` 不再另寫一份；找不到設定檔時 fail-closed。
- **細節**：`docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` X.13.6。

### I-4｜image｜`_compute_visibilities()` 的 Gλ → rad⁻¹ 換算 —— **已修正**

- **問題**：以 Gλ 為單位的基線本身就是角頻率，原本又乘了一次波長，
  等於把每條基線縮小 767 倍，16 條全部落進 uv 原點的格子，模型可見度在
  每條基線上都等於總流量。同樣的換算也在 `dataio/eht.py::_mock_eht_data()`。
- **修正**：兩處都改為 `uv * 1e9`；並把取樣方式從「FFT + 內插」改成
  **在要求的 (u,v) 點上直接做 DFT**——因為 FFT 網格間距是 `1/(2·FoV)`，
  讓影像視野悄悄決定 uv 精度，在 I-3 要求的 50 μas 視野下誤差達 8.96%。
  直接 DFT 把兩者解耦，對解析解的偏差 0.76%，成本約 1.2 ms。
- **外部物理檢查**（非內部前後比較）：點源條紋相位、V(0,0) = 總流量、
  42 μas 環的第一極小在 3.74 Gλ（理論 3.76、EHT 2019 觀測 ~3.4）、
  最長基線相關流量落在幾十 mJy。
- **`position_angle` 修正前後**：`max|ΔV|/max|V|` 9.59e−4 → **0.570**（×595）、
  相位 8.67e−4 → **2.80 rad**（×3225）。診斷曾預測 0.571，實測 0.570。
- **細節**：`docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` X.13.1、X.13.2。

### I-5｜image｜closure phase 的三元組不閉合

- **問題**：`_compute_closure_phases()` 取可見度陣列的連續三個元素當三角形，
  但 `_default_eht_uv()` 是一串基線而非台站陣列，**沒有任何一組滿足
  u_ij + u_jk = u_ik**。所以它是相位組合而不是 closure phase，
  不具備台站增益不變性。
- **不造成推論偏差**：simulator 與 likelihood 用同一個函式，是資料的自洽
  統計量。但若在報告中宣稱使用 closure phase，這一點必須說清楚。
- **要決定的**：是否從 `configs/instruments/eht.yaml` 的台站座標推導 uv 覆蓋，
  讓真正的三角形存在。
- **分類**：**需要你做一個設計決定**。順帶已修：wrap 的 off-by-π
  （`closure % 2π − π` 把零閉合映到 −π）。
- **細節**：`docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` X.13.4。

### I-6｜image｜SBC 的取樣成本 —— **第一次 campaign 未能完成**

- **做了什麼**：`gr_eternal`、M87\*、6 維、`use_closure_phases=False`
  （amplitude-only，明確聲明，因 I-5 未修）、dynesty `bound='live'` /
  `sample='rwalk'` / `nact=2` / `nlive=250` / L=100。
  `nact` 生效已用行為驗證（bilby 印 "An average of 4 steps will be accepted"，
  且完成的 result JSON 內 `sampler_kwargs` 含 `nact: 2`）。
- **順帶修好**：bilby 的 `check_point_delta_t` 預設 600 s 比一個 shard 還長，
  所以從來沒寫出過 resume 檔；改為 45 s 之後 resume 實測生效。
- **量到的成本**：唯一走完全程的中等注入（SNR 158）要
  **3292 s / 4574 次迭代 / 約 1.3e6 次 likelihood 呼叫**，且明顯減速
  （`nc` 54 → 1081，效率 3.9% → 0.4%）。SNR ≳ 1500 的注入全部逾時。
  likelihood 本身只要 1.755 ms/呼叫，**瓶頸是呼叫次數不是單次成本**。
- **原因**：`log10_brightness` 是 6 dex 的自由振幅，先驗預測 SNR 橫跨
  0.2–3.2e5；NS 迭代數 ≈ `nlive × H`，而 H 隨 ln(SNR) 成長。
  N=100 的期望總成本約 **145 小時**，尾部很重。
- **為什麼沒有「設上限、用跑完的筆數回報」**：收斂與否幾乎完全由 SNR 決定，
  任何時間上限都只會留下最安靜的注入——那些注入的後驗幾乎就是先驗，
  rank 自然均勻。那會產生一份**看起來校準良好但純屬選擇效應**的報告。
  因此本輪**不回報任何 rank / KS p 值 / coverage 數字**。
- **要決定的**：如何讓這條通道的 SBC 成本可負擔。可能的方向（未執行、
  未替任何一個背書）：縮小 `log10_brightness` 先驗或改為以總流量參數化、
  降低 `nlive`、換 `bound`/`sample` 策略、或接受在別的機器上長跑。
  **這些都會改動先驗或取樣設定，屬於設計決定，不是機械修正。**
- **分類**：**需要你做一個設計決定**。
- **細節**：`docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md` X.14。
  可續跑的 harness 在 `scripts/run_image_sbc.py`，
  原始數字在 `docs/calibration/image_gr_eternal_sbc_cost_probe.csv`。

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

### image SBC

- **已量測**（X.14，不再是推算）：單筆 SNR 158 的注入需 **3292 s /
  1.3e6 次 likelihood 呼叫**；SNR ≳ 1500 的注入在 120–240 s 內遠未收斂。
- **外推**：先驗 SNR 中位數 581 → 單筆約 87 分；95 百分位 5.9e5 → 約 6 小時。
  **N=100 約 145 小時**，N=20 約 29 小時。
- **likelihood 單次成本 1.755 ms**（`_compute_visibilities` 占 1.126 ms）。
  即使完全消除，單筆仍約 1900 s——瓶頸是呼叫次數。
- **在本執行環境不可行**：容器在 turn 之間暫停，只能以 ≤ 600 s 的前景
  片段推進。

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
- **「同一個數字寫在兩個地方」出現一次**：`cli.py` 的 `n_pixels: 64` 對上
  `eht.yaml` 的 128，而且兩者不只是不同、是對「這條通道能不能用」有相反
  的答案（64 px 下兩個目標都是 0% 可表示）。已改為單一讀取點並 fail-closed。
- **「算了但沒用」的死程式碼**出現兩次：`bounce` 的爆發成分（B.3）、
  radio `burst_fluence_jy_ms()` 的 `W_obs_ms`（R.15.1，判定為應刪除而非應接線）。
- **「先驗跨度遠大於儀器可表示範圍」出現四次**，嚴重度遞增：
  GW `M`（B.5，14.06% 被拒）→ radio `W_int`（R.7，11.88% 次格寬）→
  xray `T90`（X.3，49.96% 次格寬）→ image 環半徑（X.6，**80.98%** 次像素）。
  **image 這一項已完全解決**：`D_L` 變成逐目標已知常數消掉先驗側（X.12.2），
  成像視野縮到與源同尺度消掉網格側（X.13.6），現在是 100% 可表示。
  兩步都沒有動先驗本身——先驗一直是由文獻上的獨立測量決定的。
- **「單位換算錯誤」是第三次出現的類型**：GW 的 `3.086e22` vs `MPC_M`
  （相對差 1.0449e−4，偏了 D_L）、radio 的 Jy·ms → erg/cm²（X-1(c) 仍未決），
  以及 image 的 Gλ → rad⁻¹（**767 倍**，I-4，已修）。
- **對稱的尺度／單位錯誤不會被 forward-model 一致性檢查抓到，需要獨立的
  物理現實檢查。** 這是這個專案到目前為止最重要的一課，案例是 I-4：
  `simulators/image_shadow.py::_compute_visibilities()` 與
  `dataio/eht.py::_mock_eht_data()` 用了**同一個**錯誤的 Gλ → rad⁻¹ 換算，
  所以模型與資料互相比較時**完全自洽**——注入／回收自洽、simulator 與
  likelihood 自洽、SBC 也會通過——但兩邊都錯，可見度只帶總流量、不帶任何
  影像結構。這一類錯誤對「把兩邊對起來」的檢查是隱形的，只有拿**外部**
  基準才看得出來：這次用的是高斯環的解析 Hankel 變換（第一極小應在
  2.405/(2π r₀) = 3.76 Gλ，實測 3.74）、點源的條紋相位、以及 EHT 實測的
  相關流量量級。同一輪裡另外兩個問題也是靠同一種檢查才發現的——
  `bh_accretion` 的亮度歸一化讓整個先驗都在 M87\* 實測流量之上（X.13.5），
  以及 Kerr 陰影半徑的兩個公式**都**偏離精確臨界曲線（X.13.3）。
  **教訓：forward-model 一致性是必要條件，不是充分條件。每一個把物理量
  轉成儀器量的換算，都要另外對一個不是本專案產生的數字。**
- **「可偵測」與「可取樣」是兩件事，readiness 評估要分開看。**
  X.13.8 判定 image 通道具備校準條件時，依據之一是先驗抽樣 SNR ≥ 8 的比例
  81.2%（對比 GW 的 0.785%）——那只看了可偵測性。極高的 SNR 對偵測是好事，
  對巢狀取樣卻是成本來源：資訊量 H 越大、需要壓縮的先驗體積越多，
  迭代數 ≈ `nlive × H`。GW 通道的 campaign 跑得動，部分正是因為它的注入
  大多很安靜。**判準應補上：先驗預測的資訊量分布要落在取樣器負擔得起的
  範圍內**（X.14.5）。
- **preprocess 模組的接線狀況不一致**：GW 有接（`gw_observation.py`），
  radio 與 xray 的 preprocessor 沒有任何呼叫端，image 根本沒有模組。
