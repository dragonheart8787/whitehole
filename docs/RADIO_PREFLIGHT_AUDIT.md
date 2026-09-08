# radio 通道 forward-model 一致性前置稽核

## 範圍聲明

**這份文件記錄的是 radio 通道（FRB 類動態頻譜）在執行任何 SBC/coverage campaign 之前的
forward-model 一致性稽核結果。不構成任何白洞訊號偵測或未偵測的科學宣稱，也不包含任何
天文物理結論。**

WhiteSearch 是 candidate ranking engine（候選訊號排序引擎），不是白洞證明器。本輪
**沒有執行任何 SBC、campaign、dynesty 或取樣**，只做逐參數擾動的數值比對與結構性檢查，
全部使用 mock 模擬資料，未使用任何真實 CHIME/FRB 觀測資料。

**依 fail-closed 原則，本輪只回報現況與量測數字，沒有修改任何 production 程式碼。**
發現的問題該不該修、怎麼修，等決策。

方法比照 `docs/BOUNCE_PREFLIGHT_AUDIT.md` 對 `bounce` 所做的 B.1（取樣維度結構）、
B.2（逐參數活性擾動）、B.5（先驗支撐 vs 可表示範圍）三項檢查。

---

## R.0 鏈路與稽核設定

| 層 | 檔案 | 本輪確認的內容 |
|---|---|---|
| 資料來源 | `dataio/chime_frb.py` | CHIME/FRB 介面；無真實資料時回傳 `source: "SYNTHETIC"` 的合成紀錄 |
| 前處理 | `preprocess/radio_preprocess.py` | 動態頻譜前處理 |
| 模擬器 | `simulators/em_burst.py::EMBurstSimulator` | `channel = "radio"`，產生 (n_freq, n_time) 動態頻譜 |
| likelihood | `likelihoods/em_likelihood.py::RadioBurstLikelihood` | 逐像素高斯 |
| 模型 | `models/pbh_tunneling.py`、`models/alternatives.py` | **三個模型宣告 `channel = "radio"`**：`pbh_tunneling`、`magnetar`、`grb_frb` |

走這條通道的白洞模型是 **`pbh_tunneling`**（`PBHTunnelingWhiteHole`，`channel = "radio"`）；
`magnetar`（`MagnetarFlare`）與 `grb_frb`（`GRBAfterglowFRB`）是同通道的對照模型。
`cli.py` 的 radio 預設模型是 `magnetar`。

**稽核用的 context**（`em_burst.py` 的宣告預設值）：

```
freq 400-800 MHz, n_freq_chans = 64, t_start_s = 0.1, t_end_s = 0.5,
n_time_bins = 2048, tsys_jy = 1000.0, t_samp_ms = 0.1
```

由此導出的儀器尺度（實測）：

| 量 | 值 |
|---|---|
| 時間格寬 `dt` | **0.29311 ms** |
| 分析視窗長度 | **600.00 ms** |
| `sigma_noise` | **40.0000 Jy** |
| 像素數 | 131072 |
| DM 掃頻（400 vs 800 MHz） | **0.019448 ms per pc/cm³** |
| 一個時間格對應的 DM | **15.0719 pc/cm³** |
| `np.roll` 繞回的 DM | 30852.2 pc/cm³ |

### 一個與 GW 通道不同的結構特性（先講清楚，因為它決定了哪些問題可能存在）

`RadioBurstLikelihood.loglike()` 的樣板是**呼叫同一個 `EMBurstSimulator` 產生的**
（`sim.simulate(theta, context, rng=default_rng(0))`，再減掉 `noise_realisation` 得到
無雜訊訊號）。因此 **GW 通道 B.4 那種「模擬器與 likelihood 用了不同公式」的不一致，
在 radio 通道結構上不可能發生**——注入與擬合是同一段程式碼。

代價是另一面：**任何在模擬器裡死掉的參數，會同時在兩邊死掉**。這種情況下 SBC 的 rank
會是均勻的（posterior = prior），**校準檢查會「通過」，但通過的是一個空的模型成分**——
正是 `bounce` B.3 的失效模式。所以本輪的重點放在活性與先驗支撐，而不是公式比對。

---

## R.1 結構性檢查（B.1 對應項）

| 模型 | model 宣告 | likelihood 宣告 | likelihood 要但 model 沒有 | model 有但 likelihood 忽略 | `effective_parameter_names()` |
|---|---|---|---|---|---|
| `pbh_tunneling` | 10 | 9 | 無 | `log10_eta_gamma` | 9（正常） |
| `magnetar` | 7 | 5 | 無 | `log10_rm`, `linear_pol_frac` | 5（正常） |
| `grb_frb` | 5 | 5 | **`log10_W_ms`, `log10_tau_sc_ms`, `spectral_index`** | `log10_T90_s`, `z`, `spectral_index_radio` | **拋出 `ValueError`** |

`pbh_tunneling` 與 `magnetar` 的交集機制運作正常：`log10_eta_gamma`（gamma 通道效率）、
`log10_rm` / `linear_pol_frac`（偏振量，動態頻譜模擬器不產生 Stokes 參數）被正確排除在
取樣維度之外。

`grb_frb` 見 §R.5。

---

## R.2 逐參數活性檢查（B.2 對應項）

方法：抽先驗樣本 θ，用固定雜訊種子模擬 base 資料；對每個宣告參數擾動先驗跨度的 30%
（`normal` 先驗擾動 0.3σ），用**同一個雜訊種子**重新模擬，量測 `max|Δdata|`；
並在**同一份 base 資料**上比較 `lnL(θ)` 與 `lnL(θ′)`。三次抽樣取中位數。

### `pbh_tunneling`（白洞模型）

| 參數 | 在 likelihood 清單 | `max|Δdata|` [Jy] | 相對變化 | `|ΔlnL|` | 資料會動 | lnL 會動 |
|---|---|---|---|---|---|---|
| `log10_M_g` | 是 | **0** | **0** | **0** | **否** | **否** |
| `log10_f_pbh` | 是 | **0** | **0** | **0** | **否** | **否** |
| `log10_k_tunnel` | 是 | **0** | **0** | **0** | **否** | **否** |
| `log10_eta_r` | 是 | **0** | **0** | **0** | **否** | **否** |
| `log10_eta_gamma` | 否 | 0 | 0 | 0 | 否 | 否 |
| `z` | 是 | 3.66971×10⁻⁵ | 2.14066×10⁻⁷ | 5.98817×10⁻⁶ | 是 | 是 |
| `DM_host` | 是 | 4.61631×10⁻⁵ | 2.69284×10⁻⁷ | 1.25781×10⁻⁵ | 是 | 是 |
| `log10_W_int_ms` | 是 | 2.8437×10⁻³ | 1.65882×10⁻⁵ | 1.08188×10⁻³ | 是 | 是 |
| `log10_tau_sc_ms` | 是 | 3.41983×10⁻² | 1.9949×10⁻⁴ | 1.28087×10⁻² | 是 | 是 |
| `spectral_index` | 是 | 1.96665×10⁻⁵ | 1.14721×10⁻⁷ | 1.3284×10⁻⁵ | 是 | 是 |

**9 個取樣參數中有 4 個完全死掉**（`max|Δdata|` 與 `|ΔlnL|` 都是精確的 0，不是小值）。
成因見 §R.3。

### `magnetar`（對照模型）

| 參數 | 在 likelihood 清單 | `max|Δdata|` [Jy] | 相對變化 | `|ΔlnL|` | 資料會動 |
|---|---|---|---|---|---|
| `log10_fluence_jy_ms` | 是 | 72.5246 | 0.42306 | 1033.55 | 是 |
| `log10_W_ms` | 是 | **0** | **0** | **0** | **否** |
| `DM` | 是 | **0** | **0** | **0** | **否** |
| `log10_tau_sc_ms` | 是 | 1.48904 | 8.68607×10⁻³ | 2.61969 | 是 |
| `spectral_index` | 是 | 6.33647×10⁻² | 3.69627×10⁻⁴ | 6.17369×10⁻³ | 是 |
| `log10_rm` | 否 | 0 | 0 | 0 | 否 |
| `linear_pol_frac` | 否 | 0 | 0 | 0 | 否 |

**5 個取樣參數中有 2 個完全死掉。** 成因見 §R.4。

### `grb_frb`（對照模型）

| 參數 | 在 likelihood 清單 | `max|Δdata|` [Jy] | 相對變化 | `|ΔlnL|` | 資料會動 |
|---|---|---|---|---|---|
| `log10_fluence_jy_ms` | 是 | 68.2893 | 0.398354 | 2805.78 | 是 |
| `log10_T90_s` | 否 | 0 | 0 | 0 | 否 |
| `z` | **否** | **3.10539×10⁻²** | 1.81148×10⁻⁴ | **3.08512×10⁻⁴** | **是** |
| `spectral_index_radio` | 否 | **0** | **0** | **0** | **否** |
| `DM` | 是 | **0** | **0** | **0** | **否** |

注意 `z`：它**確實會改變資料**（透過 `_get_dm()` 的 `z * 855`），卻**不在 likelihood 宣告的
清單裡**，所以會被固定住不取樣。likelihood 對 `grb_frb` 同時**多要**了三個不存在的參數、
**漏掉**了一個真的有作用的參數。見 §R.5。

原始數表：`docs/calibration/radio_preflight_parameter_activity.csv`。

---

## R.3 阻斷性問題 (I)：`pbh_tunneling` 的 fluence 路徑從未接上

`EMBurstSimulator._get_fluence()` 只認兩個鍵：

```python
if "fluence_jy_ms" in params:
    return float(params["fluence_jy_ms"])
return float(10.0 ** params.get("log10_fluence_jy_ms", 0.0))
```

**`pbh_tunneling` 這兩個都沒有宣告**，所以每一次抽樣都落到預設值 `10**0.0 = 1.0 Jy·ms`。
200 次先驗抽樣實測：

| 量 | 最小 | 中位數 | 最大 |
|---|---|---|---|
| 模擬器實際用的 `fluence_jy_ms` | **1** | **1** | **1** |

而模型自己**有**一個 `PBHTunnelingWhiteHole.burst_fluence_jy_ms()`，從 `log10_M_g`、
`log10_eta_r`、`z` 與寬度推導 fluence——**模擬器從來沒有呼叫它**。逐筆對照：

| 抽樣 | `log10_M_g` | `log10_eta_r` | `z` | 模擬器實際用的 | `model.burst_fluence_jy_ms` |
|---|---|---|---|---|---|
| 0 | +14.058 | −7.198 | 0.01863 | 1 | 7.7011×10⁻¹⁰ |
| 1 | +13.224 | −4.387 | 0.3815 | 1 | 1.14436×10⁻¹⁰ |
| 2 | +13.987 | −7.084 | 0.0005221 | 1 | 1.1132×10⁻⁶ |
| 3 | +15.913 | −6.812 | 0.3809 | 1 | 2.10529×10⁻¹⁰ |
| 4 | +15.571 | −1.488 | 0.00166 | 1 | 1.66149 |

**這就是 `log10_M_g` 與 `log10_eta_r` 精確死掉的原因。**

`log10_f_pbh`（PBH 佔暗物質比例）與 `log10_k_tunnel`（穿隧係數）則是另一回事：
它們是**事件率／壽命**參數，本來就不進入單一爆發的波形，所以即使 fluence 接上了
它們仍然不會影響動態頻譜。**這是模型設計層面的問題，不只是接線問題**——目前的
單事件 likelihood 沒有任何管道能約束它們。

`log10_M_g` / `log10_eta_r` 是「該接沒接」，`log10_f_pbh` / `log10_k_tunnel` 是
「這個 likelihood 結構上約束不了」。兩者要分開處理。

---

## R.4 阻斷性問題 (II)：參數命名不一致，讓 `magnetar` 兩個參數靜默失效

模擬器讀的鍵與模型宣告的名字對不上，而 `params.get(key, default)` 會**安靜地**用預設值：

| 模擬器讀的鍵 | 預設值 | `magnetar` 宣告的名字 | 後果 |
|---|---|---|---|
| `log10_W_int_ms` | 1.0 → **W_int = 10 ms** | `log10_W_ms` | 寬度被釘死在 10 ms |
| `DM_total`，否則 `100 + z*855 + DM_host` | **150** | `DM` | DM 被釘死在 150 |

直接驗證（同一組 magnetar 抽樣，同雜訊種子）：

| 動作 | `max|Δdata|` [Jy] |
|---|---|
| 加 1 到 `log10_W_ms`（模型宣告的名字） | **0** |
| 加 1 到 `log10_W_int_ms`（模擬器讀的名字） | 0.0602424 |
| 加 1 到 `DM`（模型宣告的名字） | **0** |
| 加 1 到 `DM_total`（模擬器讀的名字） | 3.48711×10⁻³ |
| 加 1 到 `spectral_index_radio`（`grb_frb` 宣告的名字） | **0** |
| 加 1 到 `spectral_index`（模擬器讀的名字） | 5.17515×10⁻³ |

同一組抽樣的實際數值：模型宣告 `log10_W_ms` = 2.8019（即 **633.658 ms**）、
`DM` = **22.7563**，而模擬器 metadata 記錄的是 `W_int_ms` = **10**、`dm` = **150**。

**兩邊差了兩個數量級，而且完全沒有任何警告。**

---

## R.5 阻斷性問題 (III)：`grb_frb` 目前無法在這條通道上執行

`RadioBurstLikelihood.parameter_names` 是一個依 `model_name` 手寫的分支，
只處理 `"null"` 與 `"pbh_tunneling"`，其餘一律落到註解寫著 `# magnetar_flare` 的
那個 5 參數清單。`grb_frb` 因此拿到的是 **magnetar 的參數清單**。

結果：

```
ValueError: Likelihood RadioBurstLikelihood requires parameter(s)
['log10_W_ms', 'log10_tau_sc_ms', 'spectral_index'] that model
'GRBAfterglowFRB' does not declare
(model provides ['log10_fluence_jy_ms', 'log10_T90_s', 'z',
 'spectral_index_radio', 'DM']).
```

B.1 的交集機制**正確地 fail-closed 擋下來了**（這是好事，不是新 bug），
但代價是 **`grb_frb` 目前完全不能當作 radio 通道的對照假說使用**。

同時它還漏掉了 `z`——一個實測會改變資料（`max|Δdata|` = 3.10539×10⁻²，
`|ΔlnL|` = 3.08512×10⁻⁴）的參數。**清單同時多要與少要。**

---

## R.6 阻斷性問題 (IV)：`pbh_tunneling` 在自己的先驗下訊號強度趨近於零

因為 fluence 被釘死在 1 Jy·ms，而 `sigma_noise` = 40 Jy，注入訊號的
matched-filter SNR（`sqrt(Σs²)/σ`，200 次先驗抽樣）：

| 模型 | fluence 中位數 [Jy·ms] | 峰值中位數 [Jy] | SNR 中位數 | SNR p10 | SNR p90 | SNR < 1 的比例 | SNR < 8 的比例 |
|---|---|---|---|---|---|---|---|
| **`pbh_tunneling`** | **1**（恆定） | 0.012335 | **0.0220607** | 6.61688×10⁻⁴ | 0.193888 | **1.0000** | **1.0000** |
| `magnetar` | 60.5609 | 3.31739 | 3.36758 | 5.54222×10⁻³ | 1248.52 | 0.4100 | 0.5950 |
| `grb_frb` | 21.0139 | 1.32343 | 1.43062 | 1.76008×10⁻³ | 910.712 | 0.4650 | 0.5950 |

**`pbh_tunneling` 的先驗中沒有任何一筆抽樣的 SNR 到 1。** 也就是說，即使是那 5 個
「活著」的參數，在這個設定下也幾乎沒有資訊可用——這正好對應 §R.2 表中那些
10⁻⁷ 等級的相對變化。

**而且把 `burst_fluence_jy_ms()` 接上去並不會解決這件事。** 20000 次先驗抽樣的
模型推導 fluence（log10，Jy·ms）分位數：

| 0% | 1% | 10% | 50% | 90% | 99% | 100% |
|---|---|---|---|---|---|---|
| −18.125 | −15.771 | −12.430 | **−6.768** | −1.259 | +1.976 | +4.350 |

按 SNR 對 fluence 線性外推：中位數 SNR 會變成 **3.763×10⁻⁹**，
只有 **1.355%** 的抽樣 SNR ≥ 1、**0.465%** 的抽樣 SNR ≥ 8。
（此儀器設定下 SNR = 8 需要 fluence ≈ **362.6 Jy·ms**。）

**所以這不是單純「把函式接起來」就能收尾的問題**：接上之後，先驗的絕大部分區域
仍然是無訊號區。要讓這條通道有意義，先驗範圍、儀器設定（`tsys_jy`、頻寬、取樣）
與 fluence 公式三者需要一起檢視。**本輪不提出修法，只回報這個量化事實。**

---

## R.7 先驗支撐 vs 可表示範圍（B.5 對應項）

20000 次 `pbh_tunneling` 先驗抽樣：

| 檢查 | 比例 |
|---|---|
| `W_int` 小於一個時間格（無法解析） | **0.1188** |
| `W_int` 比整個視窗還寬 | **0.0563** |
| `tau_sc` 在 400 MHz 比視窗還寬 | **0.2094** |
| `tau_sc` 在 400 MHz 低於 `> 0.01 ms` 那道 guard | 0.0000 |
| DM 掃頻超過視窗（`np.roll` 會繞回） | 0.0000 |
| DM 掃頻小於一個時間格（不可見） | 0.0000 |
| `W_obs` 比視窗還寬 | 0.0563 |

各量在先驗上的實際跨度：

| 量 | 先驗跨度 | 儀器尺度 |
|---|---|---|
| `W_int` | 0.1 – 999 ms | 格寬 0.2931 ms、視窗 600 ms |
| `tau_sc`（400 MHz） | 0.3906 – 3906 ms | 視窗 600 ms |
| `DM_total` | 101.1 – 5329 pc/cm³ | 掃頻 1.966 – 103.6 ms |

觀察：

1. **`log10_W_int_ms` 的先驗 `uniform(-1, 3)` 有 11.88% 在時間解析度之下、
   5.63% 超過分析視窗**。這與 GW 通道 B.5（`M` 先驗支撐超出可分析頻帶）同類型，
   但**程度輕得多**，而且沒有硬性拒絕機制——超出範圍的抽樣不會被 reject，
   只是變成一條無法解析或填滿整個視窗的脈衝。
2. **`log10_tau_sc_ms` 的先驗有 20.94% 讓 400 MHz 頻道的散射時間超過視窗**。
   `scatter_broaden()` 的 kernel 長度是 `min(10*int(tau/dt)+1, len(times))`，
   被上限截斷後再正規化成 sum = 1，所以極大的 τ 會變成近似均勻抹平，
   而不是報錯——同樣是安靜的行為改變。
3. **`> 0.01 ms` 那道 guard 在整個先驗範圍內從未觸發（比例 0.0000）**，是無作用的程式碼。
4. **DM 是被離散化的**：`apply_dm_dispersion()` 用 `int(round(delay_s/dt))` 做整數格位移，
   一個時間格對應 **15.0719 pc/cm³**，所以 DM 只能在約 15 pc/cm³ 的網格上被表示。
   `pbh_tunneling` 的 `DM_total` 最小 101.1（約 6.7 格），沒有落到不可見區；
   但 `magnetar` / `grb_frb` 的 `DM` 先驗下界是 **10 pc/cm³**，低於一個格——
   若 §R.4 的命名不一致被修好，那一段先驗會變成完全不可見。
5. `apply_dm_dispersion()` 用的是 **`np.roll`（環狀位移）**，不是截斷。本輪三個模型的
   DM 先驗都不會繞回（繞回門檻 DM = 30852.2），但這是先驗範圍決定的，不是程式邏輯保證。

---

## R.8 非阻斷、但應記錄的事項

1. **`t_samp_ms` 與實際時間格寬不一致。** `sigma_noise = tsys / sqrt(Δν · t_samp)` 用的是
   context 的 `t_samp_ms = 0.1 ms`，但資料格點是
   `linspace(-t_start, t_end, n_time)` 給出的 **0.29311 ms**，比值 **2.9311**。
   因為注入與樣板用同一個模擬器，這不會造成兩邊不一致的偏差，但代表 `sigma_noise`
   與資料網格不自洽（差 √2.9311 ≈ 1.71 倍）。
2. **沒有到達時間參數。** 脈衝在模擬器裡永遠置中在 `t = 0`
   （`pulse = exp(-0.5 (times_s / sigma_t_s)**2)`）。所以 radio 通道不存在 GW 通道
   `log10_dt_bounce_s` 那種連續到達時間自由度，也就不會有對應的窄峰／簡併問題。
   但這也表示爆發時刻被當成已知，是一項建模簡化。
3. **`log10_eta_gamma`、`log10_rm`、`linear_pol_frac` 被交集機制正確排除**，
   不是死參數問題——它們屬於這個 likelihood 不涵蓋的通道（gamma、偏振）。
4. **`preprocess/radio_preprocess.py` 不在這條 likelihood 路徑上。**
   `RadioPreprocessor` 除了 `preprocess/__init__.py` 的匯出之外，
   在 `src/` 與 `tests/` 中**沒有任何呼叫端**。對照組是 GW 通道：那裡
   `gw_preprocess` 透過 `dataio/gw_observation.py` 實際參與 likelihood 前的處理。
5. **`chime` 資料來源目前接不到這個 likelihood。** `dataio/loader.py` 的 `chime`
   分支回傳的是 `CHIMEFRBLoader.load_catalog()` 的**目錄表（DataFrame）**，
   不是 (n_freq, n_time) 動態頻譜。實測合成目錄的形狀是 (500, 10)、
   欄位是 `name, dm, sn, fluence_jy_ms, width_observed, scattering_time,
   is_repeater, ra, dec, source`；把它餵給 `RadioBurstLikelihood.loglike()`
   會拋出 `KeyError: 'data'`。**也就是說 radio 通道目前只有 mock 路徑，
   沒有真實資料路徑**——這與 GW 通道（有 GWOSC 路徑並已在 GW150914/GW170814 驗證）
   的成熟度不同。順帶一提，該目錄表**有** `fluence_jy_ms` 這個欄位，
   正是 §R.3 中 `_get_fluence()` 在找的鍵。

---

## R.9 稽核總結

| # | 問題 | 影響的模型 | 類型 | 阻斷性 |
|---|---|---|---|---|
| R.3 | fluence 路徑從未接上，`fluence_jy_ms` 恆為 1.0 | `pbh_tunneling` | 死參數（`log10_M_g`, `log10_eta_r`） | **是** |
| R.3 | `log10_f_pbh` / `log10_k_tunnel` 是率／壽命參數，單事件 likelihood 結構上約束不了 | `pbh_tunneling` | 模型設計 | **是** |
| R.4 | 參數命名不一致，`params.get(key, default)` 安靜吃預設值 | `magnetar`（`log10_W_ms`, `DM`）、`grb_frb`（`spectral_index_radio`） | 死參數 | **是** |
| R.5 | likelihood 的參數清單是手寫分支，對 `grb_frb` 同時多要與少要 | `grb_frb` | 取樣維度 | **是**（`ValueError`，無法執行） |
| R.6 | 自身先驗下 SNR 中位數 0.0221，100% 的抽樣 SNR < 1；接上推導 fluence 後仍只有 1.355% ≥ 1 | `pbh_tunneling` | 先驗／儀器設定 | **是** |
| R.7 | `W_int` 11.88% 不可解析、5.63% 超過視窗；`tau_sc` 20.94% 超過視窗；DM 離散化為 15.07 pc/cm³ 網格 | 全部 | 先驗支撐 | 需決策 |
| R.8 | `t_samp_ms` 與實際格寬差 2.9311 倍 | 全部 | 自洽性 | 否 |
| R.8.4 | `RadioPreprocessor` 沒有任何呼叫端，不在 likelihood 路徑上 | 全部 | 未接線 | 否（但與 GW 通道不對稱） |
| R.8.5 | `chime` 分支回傳目錄表而非動態頻譜，餵進 likelihood 會 `KeyError: 'data'` | 全部 | 無真實資料路徑 | 需決策 |

**確認一致、沒有問題的部分：**

- **沒有 GW B.4 那種 forward-model 公式不一致**，而且結構上不可能有：
  likelihood 的樣板就是呼叫同一個 `EMBurstSimulator`。
- **`effective_parameter_names()` 的交集機制在 radio 通道正確生效**：
  `pbh_tunneling`（10 → 9）與 `magnetar`（7 → 5）都正確排除了 likelihood 不使用的參數，
  對 `grb_frb` 的不一致也正確地 fail-closed 拋出例外而不是安靜降維。
- `z`、`DM_host`、`log10_W_int_ms`、`log10_tau_sc_ms`、`spectral_index` 五個參數在
  `pbh_tunneling` 上**確實同時影響模擬資料與 lnL**，方向一致，沒有單邊生效的情形。
- DM 掃頻在三個模型的先驗下都不會觸發 `np.roll` 的環狀繞回。
- mock 路徑本身（模擬 → likelihood）在資料形狀、`sigma_noise` 傳遞、
  無雜訊訊號的取得（減 `noise_realisation`，確定性種子，減法是精確的）上都自洽。

### 為什麼現在不跑 campaign

在 R.3–R.6 處理之前跑 SBC 會得到**具誤導性的「通過」**：4 個死參數的 rank 會是均勻的
（posterior = prior），校準檢查會通過，但通過的是一個空的模型成分——這正是
`bounce` B.3 的失效模式，而且這次的訊號強度（SNR 中位數 0.0221）讓連活著的參數
也幾乎沒有資訊。

**本輪未修改任何 production 程式碼，也不提出修法選擇。等決策。**

## R.10 產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `docs/calibration/radio_preflight_parameter_activity.csv` | 三個模型全部宣告參數的擾動前後資料變化、`ΔlnL`、活性旗標 | 是 |
| `docs/calibration/radio_preflight_signal_strength.csv` | 三個模型在自身先驗下的 fluence、峰值、matched-filter SNR 分位數 | 是 |
| `scratchpad/radio_audit.py`, `scratchpad/radio_audit2.py` | 稽核腳本（只讀 production 模組） | 否 |
