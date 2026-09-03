# `bh_ringdown` 推論機制校準驗證報告（SBC / Coverage）

> **文件狀態（2026-09-03 更新）**：第 1–9 節是**第一輪（修正前）**的原始紀錄，內容保持不變。
> 第一輪發現的 (A)、(B)、(D) 與工具面 (E1) 已於 commit `01ba53e` 修正，並用同規模的
> SBC/coverage 重跑驗證——**修正後三個參數（`M`、`a_star`、`log10_A`）全部校準良好**。
> 修正內容、重跑數字與並排比較見 **§10**。發現 (C)（第五輪 taper 在 mock 路徑失效）**尚未處理**，
> 仍然成立。

## 範圍聲明

**這份文件記錄的是 WhiteSearch 推論機制（bilby + dynesty + 現有 priors）在 `bh_ringdown` 模型
上的統計校準品質驗證，不構成任何白洞訊號偵測或未偵測的科學宣稱，也不包含任何天文物理結論。**

WhiteSearch 是 candidate ranking engine（候選訊號排序引擎），不是白洞證明器。本文件量測的是
「給定先驗與模擬器，推論出來的 posterior 與 evidence 是否自我一致」這件純統計/工程的事情。所有
資料皆為 mock 模擬資料，未使用任何真實 GWOSC 觀測資料。

第一輪工作**沒有修改任何 production 模組**。第 1–9 節的數字都是在既有程式碼原封不動的狀態下
量到的；發現的問題一律照 fail-closed 原則如實回報，未自行調整先驗、模型或判定門檻。
第二輪（§10）才動手修正，且修正的是模型規格與先驗映射，不是判定門檻。

---

## 1. 為什麼是現在做

`docs/GW_LIKELIHOOD_STABILIZATION_POSTMORTEM.md` 記錄的五輪修正
（`c4d07b1` → `b34c813` → `84fdba1` → `7cb1308` → `c7dc6e0`）解決了 pipeline 在真實 GWOSC
資料上系統性膨脹統計顯著性的問題。在 likelihood 本身還有系統性 bug 的情況下跑 SBC/coverage
只會校準到一個錯的 likelihood；likelihood 穩定之後，才是第一次可以真正驗證推論機制本身的時間點。

---

## 2. 驗證工具現況（讀程式碼確認）

| 模組 | 目前實作 | 既有測試涵蓋 |
|---|---|---|
| `validation/sbc.py` | `SBCRunner.run()`：`model.sample_prior(rng_i)` → `simulator.simulate()` → `runner.run()` → `compute_sbc_rank(true, thinned_posterior)`；`SBCResult.__post_init__` 自動算均勻性（`ks_2samp` 對隨機均勻樣本）。 | 只有 `tests/test_inference.py::test_sbc_smoke`：N=3、`force_toy=True`、model=`bounce`，只斷言 `ranks` 是 dict 且含 `"M"`。**從未斷言任何校準品質**。 |
| `validation/ppc.py` | 從 posterior 抽 `n_replicates` 組參數重新模擬，對 channel-specific summary statistics 算 Bayesian p-value；GW 的統計量是 `peak_strain` / `rms_strain` / `snr_proxy`。 | 只有 `test_calibration_report.py` 的檔案存在性 smoke test。 |
| **coverage** | **沒有獨立檔案**。實作在 `injection.py::InjectionRecoveryResult._compute_coverage`（單一 `ci_level`，建構時決定，預設 0.90）；報表端在 `calibration_report.py::_evaluate_coverage` / `_plot_coverage`（PASS 條件 `0.8 ≤ cov ≤ 1.0`）。 | 只有 `test_injection_recovery_smoke`（N=3、`force_toy=True`），不檢查覆蓋率數值。 |
| `validation/injection.py` | 完整 injection/recovery campaign，保留 posteriors / evidences / CI，同時計算 coverage 與 SBC ranks（用全長 posterior，rank 分母隨每次 run 變動）。 | 同上。 |
| `validation/calibration_report.py` | 固定 artifact 契約（`index.md` / `report.json` / `coverage.csv` / `sbc/` / `ppc/` / `prior_audit.csv` / `mock_vs_real/`）。 | `test_calibration_report.py` 只檢查檔案存在與 `overall ∈ {PASS, FAIL}`。 |

**關鍵現況**：CLI `calibrate` 的兩個 profile 是 `quick`（`n_sbc=6`, `nlive=30`, **`force_toy=True`**）
與 `standard`（`n_sbc=12`, `nlive=50`），且 `--model` 預設 `bounce`、`--likelihood-mode` 預設 `mf`。
換句話說，**`bh_ringdown` + `full` likelihood + dynesty 這條路徑在本輪之前從未被任何校準檢查實際跑過**，
而 N=6～12 的 SBC 在統計上也不可能判定 rank 均勻性。

---

## 3. 執行設定

| 項目 | 值 | 理由 |
|---|---|---|
| 模型 | `bh_ringdown`（`StandardBHRingdown`），5 個參數 `M, a_star, log10_A, D_L, i` | 任務指定；likelihood 已穩定的那條 GW 路徑 |
| Likelihood | `GWLikelihood("bh_ringdown", use_full_likelihood=True)` | full inner-product 模式（非 `mf` 近似） |
| 模擬器 | `GravitationalWaveSimulator`（第四輪修正過雜訊正規化的版本） | 任務指定使用現有 mock 模擬器 |
| Sampler | bilby 2.8.2 + dynesty 3.1.0，`bound="live"`, `sample="rwalk"`, `walks=32`, `dlogz=0.1` | `BilbyRunner.DEFAULT_DYNESTY_KWARGS`，未更動 |
| `nlive` | **250** | 見下方 §3.1 |
| N（SBC / injection） | **200 / 200** | ≥ 任務要求的 200；`SBCRunner` docstring 建議 ≥1000，本輪未達到，如實記錄 |
| rank 分母 L | 100 | 遠低於實測最小 posterior 樣本數（329），避免 `SBCRunner` 的 `replace=False` 抽樣讓分母失真 |
| Context | `sample_rate=4096, duration=4.0, t_merger=1.0, low_freq_cutoff=20.0` | 與 `calibration_report.py` 內建 context 一致 |
| Seed | 20260901（每個 sim 用 `seed + i`） | `SBCRunner` / `InjectionRecovery` 既有慣例 |

### 3.1 為什麼是 `nlive=250`

不是為了省時間隨便調低，而是先用 6 組配對注入實測 `nlive=250` vs `nlive=500`（production 設定）：

| 指標 | 結果 |
|---|---|
| `max abs(ln Z(250) − ln Z(500))` | **0.164 nats**（6 組配對） |
| `M` 後驗中位數相對差 | median 0.38%，最大 24.5%（僅發生在唯一一組 90% CI 橫跨 8.5–528 M⊙ 的極弱約束案例） |
| `a_star` 後驗中位數相對差 | median 2.1%，最大 4.8% |
| `log10_A` 後驗中位數相對差 | median 0.01%，最大 0.35% |
| posterior 樣本數 @ nlive=250 | 456–854（本次 200 組實跑的最小值為 **329**），皆 ≫ L=100 |
| 單次執行時間中位數 | 20.5 s（250）vs 105.6 s（500） |

`nlive=250` 是 production `nlive=500` 的一半，在 ln Z 與後驗分位數上與 production 設定一致到遠小於
本身統計誤差的程度，且 posterior 樣本數仍遠高於 rank 分母。全程使用 dynesty，**未使用 toy sampler、
未設定 `force_toy`**（`per_simulation.csv` 中 200/200 筆的 `sampler` 皆為 `dynesty`、
`is_approximate_evidence=False`）。

### 3.2 三個 campaign

| Campaign | 內容 | 用途 |
|---|---|---|
| **A `sbc_prod`** | `SBCRunner` N=200, L=100, production 設定（`TAPER_ALPHA=0.1`） | SBC 主結果 |
| **B `cov_prod`** | `InjectionRecovery` N=200, `ci_level=0.90` | Coverage 主結果 |
| **C `sbc_notaper`** | 同 A，但在 driver script 內把 `gw_likelihood.TAPER_ALPHA` 覆寫為 0.0 | **診斷用**，把第五輪 taper 的影響與其他問題分離。**這只是 script 內的執行期覆寫，repo 內的常數未被更動。** |

執行 provenance：A/B/C 皆 **200/200 成功、0 次 dynesty bound fallback、0 次 sampler 例外**。
A 總計 11539 s、C 總計 9936 s、B 總計 11558 s（4 核平行）。

---

## 4. SBC Rank Statistics

Rank 定義沿用 `utils/math_utils.compute_sbc_rank`（真值以下的後驗樣本數），L=100 → rank ∈ {0,…,100}。
下表的 KS 為對 Uniform(0,1) 的單樣本檢定（`ks_1samp`，rank 以 `(r+0.5)/(L+1)` 連續化），
χ² 為 20 個等寬 bin 的適合度檢定。`module ks_2samp p` 是 `SBCResult` 自己算出來的值，一併列出對照。

### Campaign A — production（`TAPER_ALPHA=0.1`）

| 參數 | rank 平均（期望 50.0） | rank 標準差（期望 29.15） | frac rank=0 | frac rank=L | KS p | χ² p | module ks_2samp p | 判定 |
|---|---|---|---|---|---|---|---|---|
| `M` | 38.04 | 30.64 | 0.075 | 0.010 | **1.1e-07** | **8.8e-11** | 4.7e-05 | **FAIL** |
| `a_star` | 45.49 | 30.02 | 0.010 | 0.010 | 0.075 | 0.052 | 0.394 | 邊際（見 §4.1） |
| `log10_A` | 60.10 | 46.97 | **0.325** | **0.530** | **6.5e-52** | **3.1e-302** | 1.5e-26 | **FAIL** |
| `D_L` | 52.69 | 29.85 | 0.005 | 0.010 | 0.221 | 0.948 | 0.030 | **PASS** |
| `i` | 94.79 | 9.53 | 0.000 | **0.640** | **1.8e-104** | **0.0** | 6.6e-43 | **FAIL** |

### Campaign C — 診斷（`TAPER_ALPHA=0.0`）

| 參數 | rank 平均 | rank 標準差 | KS p | χ² p | 判定 |
|---|---|---|---|---|---|
| `M` | 37.58 | 30.97 | 1.3e-07 | 5.2e-15 | FAIL |
| `a_star` | 45.76 | 30.17 | 0.095 | 0.011 | 邊際 |
| `log10_A` | 60.32 | 46.79 | 6.5e-52 | 1.5e-297 | FAIL |
| `D_L` | 52.25 | 29.18 | 0.647 | 0.917 | PASS |
| `i` | 94.86 | 9.65 | 1.6e-104 | 0.0 | FAIL |

**A 與 C 的 rank 統計幾乎完全一致**——即使兩者的 ln Z 差了 7 個數量級
（median ln Z：−9.0×10¹⁰ vs −6723）。這代表第五輪 taper 在 mock 上造成的問題（§6 C）**不是**
rank 非均勻的成因，但它確實摧毀了 evidence 的精度（median `ln_Z_err`：**533.8 nats vs 0.214 nats**）。

### 4.1 Rank 直方圖（20 bins，N=200，均勻期望值每格 10）

```
Campaign A (TAPER_ALPHA=0.1)
M        36 16 13 15  8 10  7  8  9  8  7  8  6  7  9  7  9  7  5  5   → 單調遞減 + 低端尖峰
a_star   23 11 11 11  7  8  9  8 16 10  9  5 11  7  8 14  9  8  8  7   → 大致平坦，僅低端輕微超額
log10_A  68  0  4  1  1  2  0  3  0  2  1  2  0  0  2  0  1  2  1 110  → 極端雙峰（兩端各佔 32.5% / 53.0%）
D_L      10 12  9  7  8 10 12  8 11  7  8  9  8 13  9 10 13 16  9 11   → 平坦
i         0  0  0  0  0  0  0  0  0  0  1  1  2  2  8  6 12 10 15 143  → 全部堆在上界
```

完整的 rank histogram PNG（由既有 `SBCResult.plot()` / `plot_all()` 產生）在
`artifacts/calibration/bh_ringdown_sbc/{sbc_prod,sbc_notaper}/`（該路徑被 `.gitignore` 排除，
只存在於本機）。機器可讀的統計表在 `docs/calibration/bh_ringdown_sbc_rank_statistics.csv`。

---

## 5. Credible Interval Coverage

Campaign B（`InjectionRecovery`，N=200）。90% 一欄直接來自既有的
`InjectionRecoveryResult._compute_coverage`；50% / 68% 是對同一批保留下來的 posterior，
沿用同一個既有 helper `utils/math_utils.compute_credible_interval` 計算，**沒有另寫一套 coverage 邏輯**。
括號內為二項式標準誤。

| 參數 | 名目 50% | 名目 68% | 名目 90% | 方向 |
|---|---|---|---|---|
| `M` | **0.395** (±0.035) | **0.585** (±0.035) | **0.795** (±0.029) | 系統性 under-coverage |
| `a_star` | 0.480 (±0.035) | 0.650 (±0.034) | 0.855 (±0.025) | 輕微 under-coverage |
| `log10_A` | **0.045** (±0.015) | **0.085** (±0.020) | **0.115** (±0.023) | 嚴重 under-coverage |
| `D_L` | 0.485 (±0.035) | 0.680 (±0.033) | **0.905** (±0.021) | 與名目值一致 |
| `i` | **0.075** (±0.019) | **0.125** (±0.023) | **0.260** (±0.031) | 嚴重 under-coverage |

`calibration_report.py::_evaluate_coverage` 的 PASS 條件是 `0.8 ≤ coverage_90 ≤ 1.0`：
`a_star`（0.855）與 `D_L`（0.905）通過，`M`（0.795）、`log10_A`（0.115）、`i`（0.260）不通過。

---

## 6. 發現的問題（照 fail-closed 原則回報，未修正）

以下四項都是在準備與執行本輪驗證時，透過閱讀程式碼並用數值實驗確認的。**本輪沒有修改任何
production 模組**——這些問題全部原封不動留在 repo 裡等待決策。

### (A) `i` 的先驗在「抽真值」與「做推論」兩邊不一致

`models/base.py::ParameterSpec.sample()` 的 `cos_uniform` 分支回傳 `arccos(U(-1,1)) ∈ [0, π]`
（pdf ∝ sin i），`ParameterSpec.log_prior()` 也是同一個定義；但 `to_bilby_prior()` 轉出來的是
`bilby.core.prior.Cosine`，其預設支撐是 **[−π/2, +π/2]**（pdf ∝ cos x）。

- 實測（20000 次抽樣）：**50.3% 的 SBC 真值落在 bilby 先驗的支撐之外**，這些 sim 的 rank 必然頂到上界。
- 這正好對應觀測到的 `i` rank 分布（64% 落在 rank=L，rank 平均 94.79）與 coverage 0.260 @ 90%。
- 同一個 bug 家族：`to_bilby_prior()` 的 `half_normal` 與 `beta` 沒有對應 case，會掉進
  `case _` 變成 `bp.Uniform(0.0, 1.0)`。`bh_ringdown` 不使用這兩種先驗，但其他模型會。

### (B) mock 模擬器與 likelihood 對 `bh_ringdown` 使用的不是同一個 forward model

- `simulators/grav_wave.py` 的 ringdown 振幅是 `h0 = G·M/(c²·D_L)` × antenna pattern(`i`)，
  **完全不使用 `log10_A`**。實測：其他參數固定、只把 `log10_A` 從 −24 改到 −18，
  產生的資料 `max|d₁ − d₂| = 0.0`（完全相同）。
- `likelihoods/gw_likelihood.py::_build_template()` 的 `bh_ringdown` 分支用
  `A_rd = 10**log10_A`，**完全不使用 `D_L` 與 `i`**。實測：只改 `D_L`（400→4000 Mpc）
  或 `i`（0.5→1.5 rad），`ΔlnL = 0.0000e+00`（逐位元相同）。

SBC 的成立前提是資料由 likelihood 所假設的同一個 `p(θ)p(d|θ)` 生成。這裡不是：`log10_A` 的真值與
資料無關，而 posterior 卻被資料裡真正的（由 `M, D_L, i` 決定的）振幅釘得很緊——這正好產生觀測到的
極端雙峰 rank 分布（兩端 32.5% / 53.0%）與 coverage 0.115 @ 90%。

反過來說，`D_L` 完全不進入 likelihood ⇒ 其 posterior 就是先驗，而 `volume_uniform` ↔
`bilby.PowerLaw(alpha=2)` 的對應是正確的 ⇒ **`D_L` 成為一個乾淨的正對照組**：
KS p = 0.221 / 0.647、coverage 0.485 / 0.680 / 0.905，全部貼合名目值。這證明 SBC 的
rank 統計、thinning、coverage 計算與 dynesty 本身都沒有問題。

### (C) 第五輪的 taper 在 mock 資料路徑上讓 per-bin ⟨d|d⟩ 失效

mock 雜訊是由 `gaussian_noise_from_psd()` 用 `irfft` 直接從 PSD 生成的**嚴格週期**訊號，
只在「未加窗」的 rfft 基底上是對角的。一旦套用第五輪引入的 Tukey 窗（`TAPER_ALPHA = 0.1`），
`aligo_psd_analytic()` 未截斷的地震牆項 `(4.49x)^-56` 造成的 **band 內 PSD 動態範圍 1.7×10²¹**
（20 Hz: 1.71×10⁻²⁷ vs 200 Hz: 1.02×10⁻⁴⁸）就會洩漏進整個分析頻帶。

| mock 純雜訊 per-bin ⟨d\|d⟩（理論值 2.0） | `taper_alpha = 0.0` | `taper_alpha = 0.1`（現行） |
|---|---|---|
| aLIGO analytic PSD（模擬器實際使用） | mean **1.992** / median 1.427 | mean **9.45×10⁷** / median 491 |
| 平坦 PSD 1×10⁻⁴⁶（對照組） | mean 1.992 | mean 1.865 |

- 第五輪的合成驗證用的是「白雜訊 + 已知平坦 PSD」，正好是唯一看不到這個問題的組態。
- 真實資料路徑不受影響：`GWPreprocessor` 已對真實 strain 做過 bandpass，帶外功率在時域裡就已經被壓掉了。
- 具體後果（本輪實測）：mock 上 `null lnL = −3.17×10¹¹`（band 只有 6721 個 bin，合理量級是 ~−10⁴）；
  200 組 SBC 的 median `ln_Z_err` 是 **533.8 nats**，而 `TAPER_ALPHA=0.0` 的同一批只有 **0.214 nats**。
- **重要的是**：Campaign A 與 C 的 rank 統計與判定完全一致，所以這一項**不是** rank 非均勻的成因；
  它影響的是 **evidence（ln Z）的精度**，而 evidence 正是 WhiteSearch 排序引擎的核心輸出。

### (D) 先驗支撐與 likelihood 可表示範圍不一致（`M` 失敗的主因）

`_build_template()` 在 `f_rd < low_freq_cutoff` 或 `f_rd > 0.95·nyquist`（= 1945.6 Hz）時回傳
`None` ⇒ `loglike` 回傳 `ll_min`；而 `_band_mask()` 的上界是 `high_freq_cutoff = 1700 Hz`。
`M` 的先驗是 `log_uniform(5, 1000) M⊙`，對應 `f_rd` 從 11.9 Hz 到 6505 Hz。本輪 200 組真值中：

- **25/200（12.5%）** 的注入 `f_rd` 被 `_build_template()` 直接拒絕（lnL = `ll_min`）；
- **9/200（4.5%）** 落在 1700–1945.6 Hz 的「死區」：template 建得出來，但完全落在 band mask 之外，
  likelihood 對它毫無敏感度；
- 只有 **166/200（83%）** 的注入是 likelihood 真正能表示的。

依此條件切開後（同一批 200 組，不重跑）：

| 參數 | 子集 | n | rank 平均 | KS p | coverage 68% | coverage 90% |
|---|---|---|---|---|---|---|
| `M` | 全部 | 200 | 38.04 | 1.1e-07 | 0.585 | 0.795 |
| `M` | **f_rd 在 band 內** | 166 | 41.99 | 1.6e-03 | **0.669** | **0.892** |
| `M` | f_rd 在 band 外 | 34 | 18.71 | 3.2e-13 | 0.176 | 0.324 |
| `a_star` | f_rd 在 band 內 | 166 | 45.46 | **0.114** | **0.687** | 0.861 |
| `D_L` | f_rd 在 band 內 | 166 | 51.89 | **0.570** | **0.681** | **0.916** |
| `log10_A` | f_rd 在 band 內 | 166 | 57.01 | 2.0e-39 | 0.072 | 0.108 |
| `i` | f_rd 在 band 內 | 166 | 94.90 | 3.0e-90 | 0.114 | 0.253 |

只要限制在 likelihood 能表示的注入上，**`M` 的 coverage 就回到名目值**（0.669 vs 0.68、
0.892 vs 0.90），`a_star` 的 KS p 從 0.075 升到 0.114。剩餘的 `M` rank 非均勻性
（KS p = 1.6e-03，rank 平均 42.0 而非 50）是本輪未能完全歸因的殘留項——最可能與 (B) 的振幅
參數化不一致有關（資料裡的振幅 ∝ M，模型裡的振幅是自由的 `log10_A`，兩者對 M–振幅簡併的
處理方式不同），但本文件不對此做臆測性的定量宣稱。

### (E) 較小的工具面問題（一併如實記錄，未修正）

1. **`SBCResult._compute_uniformity()` 的 p-value 不可重現。** 它用
   `ks_2samp(ranks, np.random.uniform(0, L, n))` 對一組**未設種子的全域 RNG** 抽出的均勻樣本做
   two-sample 檢定。實測：同一組 ranks 連算 5 次得到 p = 0.628, 0.628, 0.328, 0.112, 0.866。
   `calibrated = p > 0.05` 因此在門檻附近是擲骰子，且 two-sample 檢定的檢定力低於對已知均勻分布的
   單樣本檢定（本報告表格中 module 欄與 KS 欄的差異即由此而來，例如 `D_L` 在 module 是
   p=0.030 → 判 FAIL，單樣本檢定則是 p=0.221 → PASS）。
2. **`SBCResult` 的 rank 分母固定為 `n_posterior_samples`**，但 `SBCRunner` 的 thinning 是
   `min(L, n_avail)`；若某次 posterior 少於 L，該次的 rank 上界會小於分母而不被察覺。
   本輪最小 posterior 為 329（> L=100），未觸發，但這是個潛在陷阱。
3. **`SBCRunner.run()` 把原始 `context` 而非 `ctx_i` 傳給 `runner.run()`**
   （`ctx_i["rng_seed"] = seed_i` 只餵給模擬器）。目前 likelihood 不從 context 取 RNG，所以沒有
   實際影響，但與同函式內建立 `ctx_i` 的意圖不一致。
4. **`utils/math_utils.compute_credible_interval()` 的 docstring 寫 "highest posterior density"，
   實作是等尾（equal-tailed）分位數區間。** 對本報告的 coverage 沒有影響（等尾區間本來就是合法的
   CI 定義），但文件與實作不符，對多峰後驗會給出不同的區間。
5. `validation/ppc.py` 的 `plot_statistic()` / `plot_all()` 在型別註記使用 `Path`，但模組頂層沒有
   import；只因為 `from __future__ import annotations` 讓註記延後求值才不會出錯。

---

## 7. 結論

**一句話結論：目前的推論機制（bilby + dynesty + 現有 priors）在 `bh_ringdown` 模型上 5 個參數中
只有 2 個（`D_L`、`a_star`）校準良好，`M` 在限制於 likelihood 可表示的注入後也回到名目覆蓋率，
而 `log10_A` 與 `i` 則存在確定性的嚴重系統性偏差——偏差方向是 posterior 系統性過窄且位置錯誤
（coverage 0.115 / 0.260 對名目 0.90），成因不是 sampler 而是先驗映射與 forward-model 定義的不一致。**

補充判讀：

- **不是 dynesty 的問題。** 200/200 成功、0 次 bound fallback、`nlive=250` 與 `nlive=500` 的
  `ln Z` 差異 ≤ 0.164 nats；且 `D_L` 這個「完全由先驗決定」的正對照組校準完美
  （KS p = 0.221–0.647，coverage 0.905 @ 90%），證明 rank 統計、thinning、coverage 計算與
  取樣器本身都是對的。
- **不是第五輪 taper 造成 rank 非均勻。** `TAPER_ALPHA` 0.1 vs 0.0 的兩個 campaign 判定完全一致。
  但 taper 在 mock 上確實把 evidence 精度從 `ln_Z_err` 0.214 nats 惡化到 533.8 nats——對一個以
  Bayes factor 排序為核心輸出的引擎來說，這件事本身需要處理。
- **`M` 的整體 under-coverage（0.795 @ 90%）主要是先驗支撐 vs likelihood 可表示範圍不一致造成的**：
  17% 的先驗抽樣其 QNM 頻率根本落在分析頻帶之外。排除這些之後 coverage = 0.892 @ 90%。
- **`a_star` 在整體上是邊際（KS p = 0.075 / χ² p = 0.052），限制在 band 內後改善到 p = 0.114。**
  以目前 N=200 的統計力，只能說「沒有證據顯示不校準」，不能宣稱已通過。

### 可能原因清單（未動手修，等決策）

| 現象 | 最可能成因 | 對應章節 |
|---|---|---|
| `i` rank 全堆在上界、coverage 0.260 | `ParameterSpec.cos_uniform`（[0,π], ∝sin）與 `bilby.Cosine`（[−π/2,π/2], ∝cos）的先驗映射錯誤 | (A) |
| `log10_A` rank 極端雙峰、coverage 0.115 | 模擬器與 likelihood 的振幅參數化不同（模擬器用 `M,D_L,i`，likelihood 用 `log10_A`） | (B) |
| `M` 整體 under-coverage | 先驗支撐（f_rd 11.9–6505 Hz）超出 likelihood 可表示範圍（band 20–1700 Hz、template 上限 1945.6 Hz），且 `_build_template` 的上限與 `_band_mask` 的上限不一致 | (D) |
| `M` 在 band 內仍有殘留非均勻（KS p=1.6e-3） | 未完全歸因；疑與 (B) 的 M–振幅簡併處理不一致有關 | (D) |
| `ln_Z_err` 高達 533.8 nats | 第五輪 taper 與 mock 週期性雜訊 + 未截斷 aLIGO 地震牆的交互作用造成的頻譜洩漏 | (C) |
| `SBCResult.calibrated` 判定不穩定 | `_compute_uniformity` 使用未設種子的 `ks_2samp` | (E1) |

### 不在本輪範圍內、但需要一併知道的

- 本輪只跑 mock 資料，未使用任何真實 GWOSC 資料，因此**不涉及也不支持任何天文物理宣稱**。
- `SBCRunner` 的 docstring 建議 N ≥ 1000，本輪是 N = 200（任務指定的下限）。以 N=200、L=100，
  KS 檢定對「rank 平均偏離 50 約 8 以上」的偏差才有足夠檢定力；`a_star` 的邊際結果需要更大的 N 才能定論。
- PPC（`validation/ppc.py`）本輪未執行，因為在 (B) 的 forward-model 不一致沒有解決之前，
  posterior predictive 的比對對象本身就不明確。

---

## 8. 產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `docs/BH_RINGDOWN_SBC_COVERAGE_REPORT.md` | 本報告 | 是 |
| `docs/calibration/bh_ringdown_sbc_rank_statistics.csv` | 兩個 campaign × 5 參數的 rank 統計與檢定結果 | 是 |
| `docs/calibration/bh_ringdown_coverage.csv` | 50%/68%/90% coverage 與二項式標準誤 | 是 |
| `docs/calibration/bh_ringdown_coverage_90_native.csv` | `InjectionRecoveryResult.summary()` 原生輸出 | 是 |
| `docs/calibration/bh_ringdown_conditional_on_band.csv` | 依注入 f_rd 是否在 band 內切開的 rank / coverage | 是 |
| `artifacts/calibration/bh_ringdown_sbc/{sbc_prod,sbc_notaper}/rank_hist_*.png` | 由既有 `SBCResult.plot_all()` 產生的 rank 直方圖 | 否（`artifacts/` 在 `.gitignore` 內） |
| `artifacts/calibration/bh_ringdown_sbc/*/per_simulation.csv`, `ranks.json` | 每次 sim 的 provenance 與原始 ranks | 否（同上） |

> 註：任務建議的 `artifacts/calibration/bh_ringdown_sbc_report.md` 路徑在本 repo 的 `.gitignore`
> 中被整個排除（`artifacts/`），所以報告本體改放 `docs/`（與現有 postmortem 一致），
> 圖檔仍依 `calibration_report.py` 的慣例輸出到 `artifacts/calibration/`。

## 9. 對 production 程式碼的改動

**無。** 本輪沒有為了讓驗證流程能跑而修改任何 production 模組——所有 driver script 都在 repo 之外，
Campaign C 的 `TAPER_ALPHA=0.0` 是 script 內的執行期覆寫。既有測試套件在本輪開始前跑過一次確認基準：
**121 passed, 1 skipped, 1 deselected**。

---

# 10. 第二輪：修正後重新驗證

> 本節記錄對第一輪發現 (A)、(B)、(D) 與工具面 (E1) 的修正，以及用**同規模、同設定**重跑的
> SBC/coverage 結果。發現 (C) 依決策**不在本輪範圍內**，未觸碰。
> 本節同樣只描述 `bh_ringdown` 模型規格對齊與先驗映射修正後的推論機制校準品質，
> 不含任何天文物理宣稱。

## 10.1 修正內容（commit `01ba53e`）

| 對應發現 | 檔案 | 改動 | 預期解決什麼 |
|---|---|---|---|
| **(B)** | `models/alternatives.py` | `StandardBHRingdown.parameters()` 移除 `D_L`、`i`，只留 `M, a_star, log10_A`；`summary_stats()` 移除 `D_L_mpc`；docstring 記錄它是「自由振幅 nuisance 參數的 phenomenological ringdown」 | 宣告的參數向量與 likelihood 實際讀的參數一致；不再對 likelihood 完全平坦的兩個方向取樣 |
| **(B)** | `likelihoods/gw_likelihood.py` | `parameter_names` 的非 null／非 bounce 分支回傳 `["M","a_star","log10_A"]` | bilby likelihood 參數與 prior keys 對齊 |
| **(B)** | `simulators/grav_wave.py` | 參數含 `log10_A` 時改用 `A_rd = 10**log10_A`（與 `_build_template()` 的 bh_ringdown 分支同一式子）；bounce 的 `M/D_L/i` 距離+天線路徑完全不動（分支條件是 `log10_A` 是否存在，bounce 只有 `log10_A_bounce`）；metadata 新增 `A_rd`、`amplitude_source` 記錄實際走了哪條路 | **恢復 SBC 的前提**：資料由 likelihood 假設的同一個 `p(θ)p(d\|θ)` 生成 |
| **(D)** | `models/alternatives.py` | 新增 `BAND_LOW_HZ=20.0` / `BAND_HIGH_HZ=1700.0`（取自 `configs/instruments/ligo.yaml` 的 `preprocessing.{low,high}_freq_cutoff`）與 `_m_prior_bounds_for_band()`；`M` prior 從 `log_uniform(5, 1000)` 收窄為 `log_uniform(20, 590)` | 每一筆先驗抽樣的 QNM 頻率都落在 likelihood 看得見的頻帶內 |
| **(A)** | `models/base.py` | `cos_uniform` → `bilby.core.prior.Sine`（支撐 `[0, π]`、密度 ∝ sin）取代錯誤的 `Cosine`（`[-π/2, π/2]`、密度 ∝ cos）；補上 `half_normal` → `HalfNormal`、`beta` → `Beta`；`case _` 改為 **raise**，不再 silent fallback 成 `Uniform(0,1)` | 抽真值與做推論用的是同一個先驗分布 |
| **工具面 (E1)** | `validation/sbc.py` | `_compute_uniformity` 改用 `stats.kstest(u, "uniform")`，`u = (rank + 0.5)/(L + 1)` | 同一組 ranks 的 p 值與 `calibrated` 判定變成確定性 |

`M` 界限的推導（寫在 `_m_prior_bounds_for_band()` 裡，不是硬編報告數字）：`kerr_qnm_frequency`
給 `f = k(a)/M`，`k` 對自旋單調遞增，所以要讓 `a_star ∈ [0, 0.998]` 全域落在頻帶內需要
`M ≥ k(0.998)/f_high = 19.13` 且 `M ≤ k(0)/f_low = 594.88`。取 `[20, 590]` 留安全邊界。

**物理範圍檢查（不為了讓 SBC 過關硬縮）**：20–590 M⊙ 涵蓋恆星質量 BBH 殘骸到 IMBH 區間
（GW150914 殘骸 ≈62 M⊙、GW170814 ≈53 M⊙、GW190521 ≈142 M⊙），沒有窄到不合理。
**但要如實標註一個保守性**：下界 20 M⊙ 是由**最高自旋**（`a_star = 0.998`，f_rd = 1626 Hz）決定的；
低自旋時更輕的殘骸（例如 M = 10、`a_star` = 0 的 f_rd = 1190 Hz）其實仍在頻帶內，卻被這個獨立先驗
一併排除了。要放寬需要 `(M, a_star)` 的**聯合**先驗，現有的獨立 `ParameterSpec` 機制表達不了。
這是已知的保守取捨，不是一個已解決的問題。

## 10.2 重跑設定（與第一輪逐項相同）

`SBCRunner` N=200、L=100、`InjectionRecovery` N=200（`ci_level=0.90`）、dynesty `nlive=250`、
`likelihood_mode=full`、seed=20260901、context 與 §3 相同。`nlive=250` 沿用 §3.1 的理由。
provenance：**200/200 成功、0 次 bound fallback、0 次 sampler 例外**，`sampler` 全部是 dynesty、
`is_approximate_evidence=False`；posterior 樣本數最小 344（≫ L=100）；SBC campaign 總計 10847 s。

先驗抽樣的注入頻率：**f_rd ∈ [23.7, 1093.6] Hz，0/200 落在 [20, 1700] 之外**
（第一輪是 34/200 在頻帶外，其中 25 筆被 `_build_template()` 直接拒絕）。

## 10.3 SBC rank statistics（修正後）

L=100，rank ∈ {0,…,100}；`ks_p` 是模組修正後自己算出來的確定性單樣本 KS 值（本節表格中
`module_ks_p` 與獨立重算的 `ks_p` 完全相同，確認 §10.1 的 (E1) 修正正確）。

| 參數 | rank 平均（期望 50.0） | rank 標準差（期望 29.15） | frac rank=0 | frac rank=L | KS p | χ² p | 判定 |
|---|---|---|---|---|---|---|---|
| `M` | 50.46 | 30.18 | 0.015 | 0.020 | **0.813** | 0.509 | **PASS** |
| `a_star` | 46.87 | 28.63 | 0.010 | 0.005 | **0.133** | 0.114 | **PASS** |
| `log10_A` | 52.35 | 29.26 | 0.010 | 0.015 | **0.421** | 0.760 | **PASS** |

Rank 直方圖（20 bins，N=200，均勻期望每格 10）：

```
M        14  9  8  8 14 10 15  8  9  3  8 14  7  9  9  8 15 10 11 11
a_star   18  4 10 10 17 12  9  9 11 10 11 13 12  5  6 11  6  8 13  5
log10_A   8 13 12  7  6  6  8 14 12 10  6 13 11  8 13  9  9 14  9 12
```

對照第一輪的 `log10_A`（`68 0 4 1 … 1 110`，兩端各 32.5%／53.0%）與 `i`
（`0 0 … 15 143`，64% 頂在上界），極端堆積已經完全消失。

## 10.4 Coverage（修正後）

括號內為二項式標準誤。

| 參數 | 名目 50% | 名目 68% | 名目 90% | `_evaluate_coverage` 判定（`0.8 ≤ cov90 ≤ 1.0`） |
|---|---|---|---|---|
| `M` | 0.455 (±0.035) | 0.640 (±0.034) | **0.895** (±0.022) | PASS |
| `a_star` | 0.490 (±0.035) | 0.690 (±0.033) | **0.905** (±0.021) | PASS |
| `log10_A` | 0.515 (±0.035) | 0.655 (±0.034) | **0.890** (±0.022) | PASS |

九個數字全部落在名目值的 1.3σ 以內（最大偏離是 `M` 的 50%：0.455 vs 0.50，1.3σ）。

## 10.5 修正前後並排

| 參數 | rank 平均 前 → 後 | KS p 前 → 後 | coverage 68% 前 → 後 | coverage 90% 前 → 後 |
|---|---|---|---|---|
| `M` | 38.03 → **50.46** | 1.1e-07 → **0.813** | 0.585 → **0.640** | 0.795 → **0.895** |
| `a_star` | 45.48 → **46.87** | 0.075 → **0.133** | 0.650 → **0.690** | 0.855 → **0.905** |
| `log10_A` | 60.10 → **52.35** | 6.5e-52 → **0.421** | 0.085 → **0.655** | 0.115 → **0.890** |
| `D_L` | （已移除：likelihood 從未讀取） | — | — | — |
| `i` | （已移除：likelihood 從未讀取） | — | — | — |

改善幅度：

- **`log10_A`** 從**確定性失敗**（KS p = 6.5e-52，90% CI 只覆蓋 11.5% 的真值）變成校準良好
  （p = 0.421、覆蓋 89.0%）。這直接對應 (B)：模擬器現在用的振幅公式跟 likelihood 一樣。
- **`M`** 從 KS p = 1.1e-07 變成 0.813；90% coverage 從 0.795（未達 `_evaluate_coverage`
  的 0.8 下限）升到 0.895。第一輪把注入限制在頻帶內時 `M` 的 coverage 是 0.892——
  **修正後的 0.895 與那個條件值一致**，證實 (D) 的診斷正確：問題就是先驗支撐超出可分析頻帶，
  不是別的東西。
- **`a_star`** 從邊際（p = 0.075）變成 p = 0.133。它仍然是三個裡最弱的，但已無不校準的證據。
- **`D_L`／`i`** 已從模型移除，不再需要驗證。第一輪 `D_L` 作為正對照組校準完美這件事，
  是判斷「SBC 工具鏈本身沒問題」的依據，本輪修正沒有動到那條邏輯。

## 10.6 一句話結論（第二輪）

**修正 (A)(B)(D) 之後，`bh_ringdown` 模型的三個自由參數 `M`、`a_star`、`log10_A` 在
SBC rank 均勻性（KS p = 0.813 / 0.133 / 0.421，全部 > 0.05）與 credible interval coverage
（90% 名目下實測 0.895 / 0.905 / 0.890）上都校準良好，沒有任何一個參數仍然校準不良，
也沒有可辨識的系統性 posterior 過窄或過寬。**

## 10.7 仍然成立、未在本輪處理的事項

1. **發現 (C) 完全未觸碰**（依決策）。第五輪 taper 在 mock 路徑上造成的頻譜洩漏依舊：
   本輪 200 組的 median `ln_Z` = −8.91×10¹⁰、median **`ln_Z_err` = 441.8 nats**
   （第一輪是 533.8 nats；差異來自參數數量與先驗範圍改變，不是 (C) 有任何改善）。
   如 §6(C) 所述，這**不影響 posterior 的校準**（本節數字就是證據），但 evidence 的精度
   在 mock 上仍然是壞的——而 evidence 正是排序引擎的核心輸出。
2. **`discrete_uniform` 的 bilby 映射同樣是錯的，本輪刻意未修**。`ParameterSpec.sample()`
   從 `values` 均勻抽樣，`to_bilby_prior()` 卻回傳 `DeltaFunction(values[0])`——等於把
   `bounce` 的 `p_lifetime`（`values=[4, 5]`）在 dynesty 裡釘死在 4。這與 (A) 是同一個
   bug 家族，但 bilby 2.8 的 `Categorical` 只涵蓋 `0..n-1`，對 `[4, 5]` 這種偏移集合沒有
   等價類別；改成 raise 會直接讓 bounce 的 dynesty 路徑停擺，超出本輪範圍。
   目前的處置：程式碼裡標成 `KNOWN MISMATCH`，並在
   `tests/test_prior_mapping.py::test_bilby_prior_matches_sample_prior` 以
   `xfail(strict=True)` 鎖住（若有人修好而未移除 marker，測試會轉紅）。**待決策**。
3. **`M` 先驗下界的保守性**（見 §10.1）：需要 `(M, a_star)` 聯合先驗才能放寬。
4. **N=200 的檢定力限制**：`SBCRunner` docstring 建議 N ≥ 1000。`a_star` 的 p = 0.133
   只能說「沒有證據顯示不校準」，不能宣稱已排除小幅偏差。
5. **PPC 仍未執行**。(B) 修好之後 posterior predictive 的比對對象已經明確，可以做，
   但不在本輪指定範圍內。

## 10.8 測試

`pytest`：**147 passed, 1 skipped, 1 deselected, 1 xfailed**
（第一輪基準 121 passed, 1 skipped, 1 deselected；新增 26 個通過 + 1 個刻意的
`discrete_uniform` xfail，無非預期 regression）。

新增測試檔：

| 檔案 | 內容 |
|---|---|
| `tests/test_prior_mapping.py` | 對 `PriorType` Literal 裡**每一種**類型比對 `sample()` 與 `to_bilby_prior().sample()` 的經驗分布（各 20000 抽樣，兩樣本 KS）；「新增類型必須同步更新本檔」的守門測試；`cos_uniform → Sine` 具名斷言；未知類型必須 raise；每個註冊模型都能建出 bilby priors |
| `tests/test_bh_ringdown_alignment.py` | 模型／likelihood 參數向量一致且無死參數；模擬器注入振幅 == `10**log10_A` 且對它有反應；bounce 的距離／天線路徑未被動到；`M` prior 兩端 × 自旋兩端都在頻帶內、且在解析推導的界限內；500 次先驗抽樣沒有一次被 `_build_template()` 拒絕；頻帶常數與 `configs/instruments/ligo.yaml` 一致 |
| `tests/test_sbc_statistics.py` | 同一組 ranks 連呼叫 5 次 p 值與判定完全相同；不受全域 numpy RNG 影響；正規化分母是 L+1；均勻 ranks 通過、堆在邊界的 ranks 失敗 |

## 10.9 第二輪產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `docs/calibration/bh_ringdown_sbc_rank_statistics_postfix.csv` | 修正後 rank 統計與檢定結果 | 是 |
| `docs/calibration/bh_ringdown_coverage_postfix.csv` | 修正後 50%/68%/90% coverage | 是 |
| `docs/calibration/bh_ringdown_coverage_90_native_postfix.csv` | `InjectionRecoveryResult.summary()` 原生輸出 | 是 |
| `artifacts/calibration/bh_ringdown_sbc/sbc_v2/rank_hist_*.png`, `ranks.json`, `per_simulation.csv` | rank 直方圖與每次 sim 的 provenance | 否（`artifacts/` 在 `.gitignore` 內） |
