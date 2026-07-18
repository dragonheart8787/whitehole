# GW Likelihood/PSD/Preprocessing Stabilization — Engineering Postmortem

## 範圍聲明

**這份文件記錄的是 WhiteSearch pipeline 在 GW 通道上的數值穩定性與統計正確性修正過程，不構成任何白洞訊號偵測或未偵測的科學宣稱。**

WhiteSearch 是一個 candidate ranking engine（候選訊號排序引擎），不是白洞證明器。本文件描述的五輪修正解決的是同一個系統性問題家族——pipeline 在真實 GWOSC 資料上會系統性膨脹統計顯著性（scale corruption / spectral leakage / band-masking bug 等工程問題），而不是任何關於白洞是否存在的判定。文件中出現的 GW150914、GW170814 compare 輸出（ln BF、ln Z 等）僅代表這兩筆真實資料在**目前簡化模型**（H1 單站、簡化 ringdown template，非完整 CBC 波形）下，經過修正後的 pipeline 輸出的數值，用途是驗證 pipeline 本身的一致性與穩定性，不是天文物理結論。

---

## 問題陳述

在這五輪修正之前，WhiteSearch 的 GW likelihood/PSD/preprocessing 鏈在 **mock（模擬）資料**上運作正常，測試套件全數通過。但當同一套 pipeline 被套用到**真實 GWOSC 資料**（GW150914 H1，`--likelihood-mode full --nlive 500` 的 dynesty compare）時，開始出現一系列尺度異常：

- 真實資料上的 log evidence（ln Z）量級遠超合理範圍（例如曾出現 ln BF ~5.7×10¹² 這種明顯是 artifact 而非統計結果的數字）。
- dynesty 的 bound/ellipsoid 取樣在真實資料上不穩定，曾發生 `live` bound 靜默 fallback 到 `multi` 卻未被記錄的情況。
- 兩個候選模型的取樣一度都收斂到 likelihood floor（`LL_MIN`）形成的平原（plateau）上，代表 sampler 找不到任何有意義的似然梯度可以爬。

這些異常最初是在對真實資料執行 `fit`/`compare` 驗證時被發現的——mock 資料的測試套件沒有能力發現這些問題，因為 mock 資料的尺度、視窗函數、PSD 估計方式與真實 GWOSC 資料的特性（真實雜訊 PSD 曲線、真實取樣率與資料長度、真實 DQ flag 結構）系統性不同。這一系列真實資料驗證上的不穩定，觸發了以下五輪逐步深入的調查與修正。

---

## 五輪修正時間軸

> 以下每輪的 commit hash、日期、檔案異動皆以 `git log` / `git show --stat` 實際查證為準；關鍵數字直接引用自對應 commit 的 commit message（作者本人在修正當下記錄的驗證數字）。

### 第一輪 — PSD off-source 估計 + dynesty bound-fallback provenance

- **Commit**: [`c4d07b1`](https://github.com/dragonheart8787/whitehole/commit/c4d07b156f72a49f827a782698178713d098f536) `fix: off-source PSD estimation and dynesty bound-fallback provenance`
- **日期**: 2026-07-04 15:08:17 +0800
- **異動檔案**: `src/whitesearch/dataio/gw_observation.py`、`src/whitesearch/inference/bilby_runner.py`、`src/whitesearch/preprocess/__init__.py`、`src/whitesearch/preprocess/gw_preprocess.py`、`tests/test_gw_preprocess.py`（新增）、`tests/test_inference.py`

**根因**：`GWPreprocessor.prepare_raw()` 原本直接對含有事件訊號的 on-source strain 跑 Welch PSD 估計，再把這個 PSD 拿去做白化與 noise-weighted inner product。訊號功率因此洩漏進 PSD 估計本身，讓 `⟨h|h⟩`、`⟨d|h⟩` 產生偏誤；且 32s 片段在 `fft_length=4.0` 下只有約 15 個 Welch segment，PSD 估計本身變異數偏高，讓 dynesty 要爬的 likelihood surface 更崎嶇。

**修法**：`prepare_raw()` 新增 `event_gps`/`segment_gps_start` 參數；當兩者都提供時，改用「時間平移的 off-source 背景視窗」的 Welch 估計中位數作為 PSD（重用 `make_background()` 的 on-source 排除邏輯），視窗長度會針對短片段（mock）自動縮小；若縮小後仍無法滿足最小視窗數，**fail-closed** 拋出 `PSDEstimationError`，而不是靜默退回 on-source。`quality` 字典新增 `psd_source`（`"off_source"`/`"on_source"`）、`n_off_source_windows`、`off_source_window_duration_s`、以及退回 on-source 時的 `psd_source_reason`。同時修正 `bilby_runner.py` 中一個獨立問題：`_run_dynesty()` 會在 ellipsoid/bound 出錯時從 `bound="live"` 退回 `bound="multi"`，但 `run()` 記錄的 `metadata["sampler_kwargs"]` 卻仍寫著原本請求的設定——等於隱藏了 fallback。修正後會回傳並記錄 `sampler_kwargs`（實際使用值）、`requested_sampler_kwargs`（原始請求值）、`bound_fallback_occurred`/`bound_fallback_from`/`bound_fallback_to`。

**驗證**：新增 `tests/test_gw_preprocess.py` 涵蓋 off-source 選取邏輯、一個高音量 on-source burst 會污染 on-source PSD 但不污染 off-source PSD（該污染比例的具體數字 ~7×10⁶ 倍是在下一輪（`b34c813`）重新確認時記錄的，見下）、視窗不足時 fail-closed、無事件位置時明確標記 on-source 路徑、短 mock 片段的自動縮小。`tests/test_inference.py` 新增 monkeypatch 過的 bilby `run_sampler` 案例（live→multi fallback / 無 fallback）。全套測試：105 passed, 1 skipped（修正前 98 passed, 1 skipped）。

---

### 第二輪 — notch filter 遺失 IIR 分母，導致尺度失真 ~8000 倍

- **Commit**: [`b34c813`](https://github.com/dragonheart8787/whitehole/commit/b34c81325accc3bd66b62cb555f03ab88dabc7f4) `fix: notch_filter dropped the IIR denominator, distorting strain scale ~8000x`
- **日期**: 2026-07-04 19:55:20 +0800
- **異動檔案**: `src/whitesearch/preprocess/gw_preprocess.py`、`tests/test_gw_preprocess.py`

**根因**：`GWPreprocessor.notch_filter()` 呼叫 `scipy.signal.iirnotch(f0, Q, fs)` 取得 IIR 係數 `(b, a)`，但實際只用分子 `b` 透過 `np.convolve(result, b, mode="same")` 做卷積，把分母 `a` 完全丟棄。把一個 IIR 的分子單獨當 FIR kernel 使用，等於套用了一個完全不同（而且極度錯誤）的轉移函數：32s 的白雜訊，RMS 從 ~1×10⁻²¹ 被放大到 ~8×10⁻¹⁸——約 **8000 倍**的尺度暴衝，污染了 notch 之後所有下游計算（PSD 估計、白化、noise-weighted inner product）。這是與第一輪 PSD 自我污染**獨立的**另一個尺度污染來源；第一輪修正的 off-source PSD 邏輯本身未被觸動。

**修法**：改為對每一條 notch 線正確套用 `tf2sos(b, a)` + `sosfiltfilt`，與 `bandpass_filter()` 已經使用的風格一致，維持零相位濾波。逐線序列迴圈、`f0 >= Nyquist` 的保護、函式簽名皆未變動，`prepare_raw()` 不需修改。

**驗證**：修正後測得（32s 白雜訊，預設 notch 線組）：輸出/輸入 RMS 比 **0.95**；notch 線頻率上的 PSD 被壓低 **3–7 個數量級**，鄰近頻率（+20Hz）維持在幾個百分點以內的變化。新增 `TestNotchFilter`：白雜訊 RMS 須落在輸入的 0.5×–2× 之間、60/120/500Hz 的功率需降到輸入的 1% 以下、+20Hz 鄰近頻率需維持在 50% 以上（確保濾波器既沒有暴衝也不是空操作）。同時移除了兩個 off-source PSD 測試中原本用來繞開本 bug 的 `notch_lines=[]` workaround，改用預設 notch 鏈路重跑；確認 100Hz burst 的 on/off-source 污染比例（第一輪測試的斷言 >1000×）維持在 **~7×10⁶ 倍**不變（因為 100Hz 不是 notch 線，不受本輪修正影響）。全套測試：107 passed, 1 skipped, 1 deselected（修正前 105/1/1）。

---

### 第三輪 — 頻帶遮罩、LL_MIN 重新校準、crc32 seed、sampler provenance 持久化

- **Commit**: [`84fdba1`](https://github.com/dragonheart8787/whitehole/commit/84fdba143571253b1c3b60c602de7f5c67a2951b) `fix: band-mask GW inner products, rescale LL_MIN, crc32 seed, persist sampler provenance`
- **日期**: 2026-07-04 22:36:44 +0800
- **異動檔案**: `src/whitesearch/cli.py`、`src/whitesearch/likelihoods/gw_likelihood.py`、`tests/test_cli_provenance.py`（新增）、`tests/test_likelihoods.py`

**根因（四個獨立問題，皆由前兩輪修正後的真實資料驗證觸發）**：對 GW150914 H1（`nlive=500`, `likelihood-mode full`）跑驗證時，兩個模型都收斂到 likelihood floor 平原上，回報的 ln BF（~5.7×10¹²）明顯是 artifact。追查後發現：

1. **Inner product 沒有頻帶遮罩**：`_full_inner_product_loglike`、`_mf_snr_loglike`、`_null_loglike` 都對整個 rfft 頻率網格積分。真實資料上 **99.93% 的 `⟨d|d⟩ = 1.147×10¹³`** 來自 1700–2048Hz（超出 bandpass 截止頻率，是濾波器 rolloff 造成的殘餘噪聲，不是物理訊號），另外還有 9.4×10⁸ 來自 0–10Hz。
2. **LL_MIN 尺度過舊**：測得真實資料 null lnL = −5.78×10⁷；2000 次 prior draw 的真實 template lnL 落在 [−1.94×10⁹, −5.77×10⁷] 之間，**100% 低於**舊的 −1×10⁶ floor，導致被拒絕的樣板反而形成全域最大值平原，dynesty 在平原上終止。
3. **per-model seed 不是確定性的**：`compare`/`rank` 用 `seed + hash(model_name) % 1000` 產生每個模型的 seed，但 Python 的 `str.__hash__()` 受 `PYTHONHASHSEED` 隨機化影響，同一個 `--seed 42` 在不同進程間會餵給 dynesty 不同的 seed（觀測到同一組輸入在某次執行得到 660）。
4. **sampler fallback 未被持久化到 artifact**：即使第一輪已修正 `bilby_runner.py` 內部的 fallback 記錄，`cli.py` 的 `_run_single_fit` 仍未把這些欄位寫進儲存的 metadata JSON，只存在於物件記憶體/log 中。

**修法**：新增共用的 `_band_mask()` helper，把三條 inner product 路徑都限制在 `[low_freq_cutoff, min(high_freq_cutoff, 0.95*nyquist)]` 內（`strain_f`/`template_f`/`psd` 在呼叫前先切片，`math_utils.noise_weighted_inner_product` 與 `gw_units` 本身未被更動），遮罩後為空則 fail-closed 回傳 `ll_min`。遮罩後真實資料的 `⟨d|d⟩` 從 1.147×10¹³ 降為 **1.156×10⁸**。`LL_MIN` 從 −1×10⁶ 下修到 **−1×10¹²**（約在觀測到最差真實似然值之下 ~500 倍，同時仍遠高於 −1×10³⁰ 的 non-finite 保護值，不影響 mock 尺度的似然）。`cli.py` 改用 `zlib.crc32(name) % 1000` 取代 `hash()` 產生 per-model seed offset，跨進程穩定。`_run_single_fit` 補齊把 `sampler_kwargs`、`requested_sampler_kwargs`、`bound_fallback_occurred`/`from`/`to` 從 `InferenceResult.metadata` 複製進儲存的 `*_metadata.json`。

**驗證**：新增 `TestGWBandMask`（遮罩邊界；未遮罩時 out-of-band 雜訊會讓 `⟨d|d⟩` 暴衝，遮罩後 masked null lnL 落在統計上合理的 ~−N_band 量級；band 內功率不受影響，band 外音調對 lnL 的影響 <1%）、`test_model_seed_offset_stable_across_hash_randomization`（在 `PYTHONHASHSEED` 為 0/1/12345 的子進程間結果一致）、`test_run_single_fit_persists_sampler_provenance`。全套測試：112 passed, 1 skipped, 1 deselected（修正前 107/1/1）。真實資料驗證（GW150914 H1, full, nlive=500）：seed 42/43 給出 ln Z(bh_ringdown) = −57327008.03±0.12 / −57327008.40±0.59（差異 0.37, 0.6σ），ln Z(null) = −57824267.26，**ln BF = +497259 ± 0.6**；無 plateau 警告、無 bound fallback，每 seed ~27 分鐘。**本輪記錄但未處理的已知殘留問題**：band 內的 data/PSD 前處理階段仍不一致，讓 lnL 量級仍然膨脹（~−5.8×10⁷ 量級；ln BF ~5×10⁵ 仍然明顯不合理地大）；`gw_diagnostics.py` 的 `frac_finite` 判斷閾值（硬編碼 −1×10⁵）與未遮罩的 `max_mf_snr` 都跟不上真實資料的尺度。

---

### 第四輪 — 分子分母前處理階段對齊、mock 雜訊正規化、frac_finite scale-free 化

- **Commit**: [`7cb1308`](https://github.com/dragonheart8787/whitehole/commit/7cb13085d6c711d0f5c4300c4dcdceed2f862a16) `fix: align likelihood numerator with PSD stage, mock noise norm, scale-free frac_finite`
- **日期**: 2026-07-04 23:20:45 +0800
- **異動檔案**: `src/whitesearch/dataio/gw_observation.py`、`src/whitesearch/preprocess/gw_preprocess.py`、`src/whitesearch/simulators/grav_wave.py`、`src/whitesearch/validation/gw_diagnostics.py`、`tests/test_gw_diagnostics.py`（新增）、`tests/test_gw_preprocess.py`、`tests/test_simulators.py`

**根因（直接對應第三輪記錄的殘留問題）**：追蹤資料流後發現：`prepare_raw()` 內部依序算出 raw → bandpass（`strain_bp`）→ notch（`strain_notched`），off-source PSD 是從 `strain_notched` 估計的，但函式只回傳了 `strain_bandpass`；`prepare_gw_observation()` 因此把「只做過 bandpass、沒做 notch」的 strain 匯出成 `out["strain"]`，而這正是 `GWLikelihood._parse_data` 餵給 noise-weighted inner product 的分子。分子（僅 bandpass）與分母（bandpass+notch 的 PSD）在 notch 線頻率上因此互相矛盾：未經 notch 抑制的線譜功率，被拿去除以一個在該頻率被 notch 壓低的 PSD，把那幾個 bin 的貢獻異常放大。（白化後的 `strain_whitened` 其實從未被用在 likelihood 路徑上——白化是透過 1/PSD 權重隱式完成的，這部分本身沒問題，`strain_whitened` 一直只是 QA 用途。）

**修法**：`prepare_raw()` 現在也回傳 `strain_notched`；`prepare_gw_observation()` 改用它作為分析用 strain（並記錄 `analysis_stage="bandpass_notch"`、`strain_rms_notched`），`strain_bandpass` 仍保留供檢查用途。同時修正兩個獨立問題：mock 模擬器 `simulators/grav_wave.py` 的 `gaussian_noise_from_psd()`，其 `sigma = sqrt(Sn/(2·df))` 是針對 dt-scaled 係數的公式，但在 `gw_units.py` 的慣例 `h̃(f) = rfft(h)·dt` 下，這讓生成的雜訊功率比宣稱的 PSD 低了約 `2·dt²` 倍；修正為 `sigma = sqrt(Sn/(4·df))/dt`。以及 `validation/gw_diagnostics.py` 的 `frac_finite` 判斷：舊版硬編碼 `lnL > -1e5` 的閾值是針對 mock 尺度調的，會把所有真實資料的 draw（lnL ~ −1×10⁶ 以下）都誤判為 non-finite；改為 scale-free 規則——只要 lnL 是 finite 且不等於 likelihood 自己的拒絕哨兵值（`ll_min`）就算 finite，並保留可選的 `finite_threshold` 參數以便需要絕對閾值時使用。

**驗證**：真實 GW150914 H1 資料上測得的 per-bin inner-product 貢獻（理想值 ~2）：50–200Hz median **18.40 → 15.41**，mean **3319.5 → 24.6**；600–1700Hz median **2.17 → 1.83**，mean **2711.6 → 47.7**；遮罩後的 `⟨d|d⟩` 從 **1.157×10⁸ → 1.956×10⁶**（理想值 ~1.08×10⁵）。新增 `TestAnalysisStageAlignment`（合成 60Hz 線訊號：只用 bandpass 的分子會讓該線 bin 膨脹 >100 倍，寬頻貢獻維持正常）、GW 模擬器雜訊正規化測試（per-bin 貢獻 ~2，10 次實現測得 1.9996）、`tests/test_gw_diagnostics.py`（mock 資料上被接受的 draw 數量；規則在 lnL 遠低於 −1×10⁵ 時仍成立；明確指定閾值時仍可用）。全套測試：117 passed, 1 skipped, 1 deselected（修正前 112/1/1）。真實資料驗證（GW150914 H1, GWOSC, full, nlive=500）：seed 42/43 給出 ln Z(bh_ringdown) = −978150.60±0.16 / −978150.48±0.16（差異 0.12, 0.5σ），ln Z(null) = −978157.55，**ln BF = +6.95 / +7.07**（差異 0.12, 0.5σ），frac_finite = 0.88，無 bound fallback、無 plateau 警告，每 seed 取樣時間 ~10 分鐘（修正前 ~27 分鐘）。**本輪記錄但未處理的已知殘留問題**：遮罩後的 `⟨d|d⟩` 仍是統計理想尺度的 ~18 倍（band 內 median per-bin ~2.4，50–200Hz median ~15），代表即使 ln BF 已經回到合理量級，絕對的 ln Z 量級依然膨脹。

---

### 第五輪 — FFT taper 一致性修正

- **Commit**: [`c7dc6e0`](https://github.com/dragonheart8787/whitehole/commit/c7dc6e0331d1b732f569d3bc597bed3349d1a1dc) `fix: taper strain/template FFTs to match Welch PSD spectral-leakage profile`
- **日期**: 2026-07-18 18:02:36 +0800
- **異動檔案**: `src/whitesearch/likelihoods/gw_likelihood.py`、`src/whitesearch/likelihoods/gw_units.py`、`src/whitesearch/validation/gw_diagnostics.py`、`tests/test_gw_units.py`

**根因（直接對應第四輪記錄的殘留問題）**：`time_to_freq()` 對整段 32s strain 做 rfft 時，隱含使用的是矩形窗（沒有任何 taper），而 `estimate_psd()`（Welch 法，4s 分段，Hann 窗）有非常不同、也小得多的頻譜洩漏（spectral leakage）足跡。這個視窗函數不一致，讓沒有被 taper 過的邊緣/不連續點洩漏，滲入到遮罩後的 noise-weighted inner product 裡，而且滲入的尺度是 PSD 完全沒有描述過的。

**修法**：`gw_units.py` 的 `time_to_freq()` 新增可選參數 `taper_alpha=0.0`（預設值保留舊行為，所有沒有明確傳入這個參數的呼叫方——mock、其他通道、`freq_to_time` 的使用者——都不受影響）；當 `taper_alpha > 0` 時，在 rfft 前套用 Tukey 窗，且不做任何功率補償（因為窗函數在外側 `taper_alpha/2` 邊緣之外增益是 1，且 strain 與 template 套用完全相同的 taper，兩者相減得到殘差時會抵消，只要訊號不是剛好落在 taper 的邊緣區域）。`gw_likelihood.py` 新增 `TAPER_ALPHA = 0.1` 常數（數值選定與 `simulators/grav_wave.py` 中 mock 訊號注入已經使用的 Tukey taper alpha 一致），套用在 `loglike()` 裡**每一次** `time_to_freq()` 呼叫上（strain **和** template 都要 taper，否則殘差 `d − h` 會看到不對稱的邊緣洩漏），以及 `_null_loglike()`。`gw_diagnostics.py` 的 matched-filter SNR 診斷也改用同樣的 `TAPER_ALPHA`，讓它反映的是 likelihood 實際看到的東西。

**驗證**：
- 合成資料（白雜訊 + 已知平坦 PSD，32s @ 4096Hz）：置中的 ringdown 注入（`t_merger` = 片段中點，對齊真實 GW150914 的 `t_merger=16s/32s` 慣例）：SNR 在 alpha=0.1 時變化 +0.0012%，alpha=0.2 時變化 +0.0042%（相對 alpha=0.0）；模板能量 `⟨h|h⟩` 在不同 alpha 下逐位元相同，因為 burst 在 ~10ms 內衰減完畢，完全落在 taper 的單位增益區域內。
- 純雜訊 per-bin `⟨d|d⟩`（獨立實現的 Welch PSD 套用在整段 FFT 上）：alpha=0.0 mean=**2.117**，alpha=0.1 mean=**1.986**，alpha=0.2 mean=**1.854**（理論值 2.0）。alpha=0.1 最接近理論值，alpha=0.2 則明顯過度修正。
- 真實 GW150914 H1 資料（32s，off-source Welch PSD）：遮罩後 per-bin `⟨d|d⟩`，50–200Hz 從 **mean=24.55/median=15.41**（未 taper）變為 **mean=2.06/median=1.43**（alpha=0.1）；600–1700Hz 從 **mean=47.68/median=1.84** 變為 **mean=2.08/median=1.38**；全分析頻帶的 `sum(⟨d|d⟩)` 從 ~2.03×10¹¹ 降到 ~1.28×10⁵。

`tests/test_gw_units.py` 新增：`taper_alpha` 預設不 taper（向下相容）；合成的「Welch PSD vs 全片段 FFT」重現測試，證明加了 taper 之後遮罩後 per-bin `⟨d|d⟩` mean 更接近理論值 2.0；置中 ringdown 注入的回歸測試鎖定 taper 不會顯著改變 matched-filter SNR（設計上 <1%，實測 <0.01%）。

---

## 關鍵教訓

給未來維護這條 pipeline 的人（不是自我表揚，每一條都對應到上面實際發生過的 bug）：

1. **PSD 估計絕對不能用 on-source（含事件訊號）的資料。**
   第一輪（`c4d07b1`）修正前，PSD 直接用含事件的 strain 估計，訊號功率會反過來污染自己的雜訊估計，讓 `⟨h|h⟩`、`⟨d|h⟩` 系統性偏誤。修法是強制用時間平移的 off-source 背景視窗，且視窗數不足時要 **fail-closed**（拋錯），不能靜默退回 on-source——因為靜默退回會讓使用者以為自己拿到的是乾淨的 PSD。

2. **IIR 濾波器的分子分母必須一起用；只用分子當 FIR kernel 是完全不同的系統。**
   第二輪（`b34c813`）的 8000 倍尺度暴衝，根因只是 `iirnotch()` 回傳的 `(b, a)` 只用了 `b` 去做 `np.convolve`。這種 bug 在單元測試只測「有沒有跑起來」而不測「數值尺度合不合理」時完全不會被抓到——`TestNotchFilter` 後來新增的 RMS 上下界斷言（0.5×–2×）就是專門守住這類問題的不變量，任何新的濾波器相關改動都應該有類似的尺度守門測試。

3. **Noise-weighted inner product 的分子（strain）與分母（PSD）必須來自前處理管線的同一個階段，且要有測試守住這個不變量。**
   第四輪（`7cb1308`）的 bug 是分子只做了 bandpass、分母的 PSD 卻是從 bandpass+notch 之後的資料估計的，兩者在 notch 線頻率上互相矛盾。這類「兩條資料路徑各自往前走了幾步，但走的步數不一樣」的 bug 很容易在重構時無聲地重新引入，需要專門的一致性測試（例如 `TestAnalysisStageAlignment`）長期守著，不能只靠一次性修正。

4. **Likelihood floor（`LL_MIN`）不能寫死一個假設的常數，必須用實際資料的似然尺度重新校準，而且要留可觀測的安全邊界。**
   舊的 `-1e6` floor 是在 mock 尺度下選的；套用到真實資料上，因為真實似然值普遍在 `-1e7` 到 `-1e9` 量級，這個 floor 反而變成「全域最大值」，讓所有真實模板都被錯誤地判定為比 floor 差、dynesty 因此收斂到一個毫無意義的 floor 平原上。教訓是：任何硬編碼的數值 sentinel，只要資料尺度可能跨數量級變化（mock vs 真實資料），就必須有機制重新校準，且要有測試觀察「這個 floor 是否仍然低於所有觀測到的真實似然值」。

5. **頻域計算裡，任何一步的窗函數（window function）都必須跟下游用到的統計量的窗函數假設一致。**
   第三輪先修了頻帶遮罩，第四輪修了前處理階段對齊，兩者都修完之後，第五輪才發現最後一層問題：`strain` 的全段 rfft 隱含用矩形窗，但 PSD 是用 Hann 窗的 Welch 法估計的，兩者的頻譜洩漏特性不同，導致遮罩後的 `⟨d|d⟩` 仍然比理論值大一個數量級以上（第四輪修完後仍是 mean~24.6，理論值是 2）。這類問題不會在任何單一函式的單元測試裡現形，只有把「strain 怎麼變成頻域」和「PSD 怎麼估計」兩條路徑放在一起、對真實資料跑端到端驗證，才會被看見——這也是為什麼這五輪修正每一輪都是被「真實資料驗證不穩定」觸發，而不是被 mock 測試觸發。

---

## 最終跨事件驗證結果

修正五輪後，對兩個獨立真實事件（GW150914、GW170814，皆為 H1 單站，`bh_ringdown` vs `null`，`--likelihood-mode full --nlive 500`，各兩個 seed）執行 compare 的完整輸出：

| 指標 | GW150914 seed42 | GW150914 seed43 | GW170814 seed42 | GW170814 seed43 |
|---|---|---|---|---|
| ln Z (candidate, bh_ringdown) | −55713.8655 ± 0.1080 | −55713.9593 ± 0.1122 | −55403.7684 ± 0.0643 | −55403.7286 ± 0.0640 |
| ln Z (null，解析解) | −55713.3375 ± 0.0 | −55713.3375 ± 0.0 | −55402.5551 ± 0.0 | −55402.5551 ± 0.0 |
| ln BF vs null | −0.5280 ± 0.1080 | −0.6218 ± 0.1122 | −1.2133 ± 0.0643 | −1.1735 ± 0.0640 |
| 判定 | not worth mentioning | not worth mentioning | not worth mentioning | not worth mentioning |
| internal gate | FAIL | FAIL | FAIL | FAIL |
| frac_finite (candidate / null) | 0.815 / 1.0 | 0.815 / 1.0 | 0.815 / 1.0 | 0.815 / 1.0 |
| per-bin ⟨d\|d⟩ 50–200Hz median/mean | 1.428 / 2.059 | 同左 | 1.362 / 2.040 | 同一份 strain 資料 |
| per-bin ⟨d\|d⟩ 600–1700Hz median/mean | 1.384 / 2.075 | 同左 | 1.385 / 2.073 | 同一份 strain 資料 |
| 理論期望（χ²₂ per-bin） | median ≈ ln4 ≈ 1.386，mean = 2 | — | — | — |
| bound_fallback_occurred | false | false | false | false |
| psd_source / n_off_source_windows | off_source / 15 | off_source / 15 | off_source / 15 | off_source / 15 |

**跨 seed 穩定性**：兩個事件的 ln Z(candidate) 在 seed42/seed43 間的差異都遠小於合併誤差（GW150914: 差 0.094 nats，合併誤差 ~0.156；GW170814: 差 0.040 nats，合併誤差 ~0.091），皆在 <1σ 範圍內。

**跨事件一致性**：兩個獨立真實事件在五輪修正後都收斂到同一種模式——ln BF 為 O(1) 量級的負值、gate 判定一致為 FAIL/"not worth mentioning"、frac_finite 完全相同（0.815）、per-bin `⟨d|d⟩` 在兩個頻帶都緊貼理論值（mean≈2.0，median≈1.39）、皆無 bound fallback、PSD 皆成功使用 off-source 估計（15 個背景視窗）。這支持「pipeline 現在對不同真實 BBH 事件普遍給出不誇大的合理量級結果」這個工程層面的觀察，而不是 GW150914 一筆資料湊巧收斂——但這仍然只是 pipeline 數值行為的一致性驗證，不是天文物理結論。

---

## 已知限制／未解決事項

誠實列出，不美化：

- **目前的跨事件驗證只涵蓋 H1 單一偵測器，且使用簡化的 ringdown template（非完整 CBC 波形）。** 這代表這裡的 ln BF/ln Z 量級**本來就不能**跟正式 LIGO/Virgo 分析（多偵測器、完整 IMRPhenom/SEOBNR 等波形族）的 ln BF 直接比較，兩者是不同複雜度的模型在做不同範疇的比較。
- **只有 `gw` 通道經過這五輪同等程度的數值驗證。** `radio`、`xray`、`image` 通道的 likelihood/前處理鏈尚未經過對應的真實資料尺度驗證，不應假設它們沒有類似的尺度問題。
- **`validation/gw_diagnostics.py` 裡的 `max_mf_snr` 診斷值目前沒有套用頻帶遮罩。** 第三輪引入的 `_band_mask()` 只套用在 `likelihoods/gw_likelihood.py` 的 likelihood 計算路徑上；`gw_diagnostics.py` 的 `run_gw_diagnostics()`（第 85 行 `matched_filter_snr(hf, strain_f, psd, df)`）呼叫 `matched_filter_snr` 時，傳入的 `strain_f`/`psd`/`hf` 是未經 `_band_mask()` 切片的全頻段資料，`math_utils.matched_filter_snr()` 本身也不做任何遮罩。第五輪只讓這個診斷路徑的 taper 跟 likelihood 路徑一致，並未處理遮罩缺失的問題。這是本次追蹤過程中透過閱讀程式碼發現的既有殘留項，如實記錄，其對 `max_mf_snr` 數值的實際影響幅度尚未測量，本文件不做臆測。

---

## 操作性附註（非科學內容）

`.git/hooks/post-commit`（對應的可追蹤原始檔為 `scripts/hooks/post-commit`）會在**每次 commit 完成後自動 `git push` 到 origin**（失敗時不會擋下 commit，只會印出 `[whitesearch] auto-push skipped ...`）。這個 hook 從 repo 最初的 Initial commit（`eca27df`，2026-05-20）就存在，**不是這五輪修正的一部分**，是在這次追蹤過程中意外重新確認到的既有行為。在此記錄供未來維護者知悉：在這個 repo 裡執行 `git commit` 之後，除非事先停用這個 hook 或設定 `WHITESEARCH_REMOTE` 指向別處，否則變更會立即出現在 `origin/main`，不要以為「commit 了但還沒 push」。
