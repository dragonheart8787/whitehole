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

---

# 修正輪：R.4 / R.3（機械部分）/ R.5 對齊，與 R.6 尺度診斷（2026-09-09）

> 本節記錄一次 **production 修正**（`simulators/em_burst.py`、`models/alternatives.py`、
> `likelihoods/em_likelihood.py`）與一次**純診斷**（R.6，未修正任何東西）。
> 沒有跑任何 SBC/campaign。本節不含任何天文物理宣稱。

## R.11 修正內容

### R.11.1 R.4：統一參數名稱，並讓模擬器 fail-closed

**命名方向是逐項按「改動面積最小」決定的，不是一律偏向某一邊：**

| 不一致 | 統一成 | 改在哪一邊 | 理由 |
|---|---|---|---|
| `log10_W_ms`（magnetar） vs `log10_W_int_ms`（模擬器 + pbh） | **`log10_W_int_ms`** | 模型 | 模擬器與 `pbh_tunneling` 已經用這個名字；改模擬器要連帶動 pbh 模型、config、conftest、likelihood 清單 |
| `spectral_index_radio`（grb） vs `spectral_index`（模擬器 + pbh + magnetar） | **`spectral_index`** | 模型 | 三者中已有兩個模型用這個名字 |
| `DM`（magnetar + grb） vs `DM_total`（模擬器） | **`DM`** | **模擬器** | `DM_total` **沒有任何模型宣告過**，只出現在 summary-stat 的輸出鍵；改模擬器一行即可 |

模擬器端 `params.get(key, default)` 改為明確要求：

- 新增 `_require(params, key)`，缺鍵時 `KeyError` 並列出實際收到的參數。
- `log10_W_int_ms`、`log10_tau_sc_ms`、`spectral_index` 改用 `_require`。
- `_get_dm()`：接受 `DM`（總量）或 `z` + `DM_host`（分解式，pbh 用），兩者皆無則 raise。
  **保留兩種慣例是刻意的**，因為模型本身就分成這兩類。
- `_get_fluence()`：`fluence_jy_ms` → `log10_fluence_jy_ms` → pbh 推導路徑 → raise。

修正後的活性實測（同 §R.2 方法）：

| 模型 | 參數 | 修正前 `max|Δdata|` | 修正後 |
|---|---|---|---|
| `magnetar` | `log10_W_int_ms`（原 `log10_W_ms`） | **0** | **0.362274** |
| `magnetar` | `DM` | **0** | **1.37282×10⁻³** |
| `magnetar` | `spectral_index` | 6.33647×10⁻² | 3.81739×10⁻³ |
| `pbh_tunneling` | `log10_M_g` | **0** | 3.09296×10⁻⁸ |
| `pbh_tunneling` | `log10_eta_r` | **0** | 5.03544×10⁻⁷ |

`magnetar` 五個取樣參數現在全部同時影響資料與 lnL。`pbh_tunneling` 的兩個
fluence 參數從「精確為 0」變成「有作用但極小」——極小的原因是 R.6，不是接線。

### R.11.2 R.3（機械部分）：接上 `burst_fluence_jy_ms()`

`EMBurstSimulator._get_fluence()` 現在會在模型提供
`log10_M_g` / `log10_eta_r` / `z` / `log10_W_int_ms` / `log10_tau_sc_ms` 全部五個鍵時，
呼叫 `PBHTunnelingWhiteHole().burst_fluence_jy_ms(params)`，取代原本的
`10**0.0 = 1.0 Jy·ms`。派發方式沿用該模組既有的「依參數是否存在」慣例
（`_get_dm()` 本來就是這樣寫的），不引進新的模型判斷機制。

同時把 `log10_f_pbh` 與 `log10_k_tunnel` 從
`RadioBurstLikelihood("pbh_tunneling").parameter_names` 移除——它們是**事件率／壽命**
參數，決定這種事件多久發生一次，不決定單一爆發長什麼樣，單一動態頻譜對它們沒有資訊。
做法比照 `BOUNCE_PREFLIGHT_AUDIT.md` B3-3：**模型的完整宣告不變**，只是這條通道的
取樣維度對齊 likelihood 真正用得到的參數。取樣維度因此從 9 降為 **7**。

### R.11.3 R.5：`RadioBurstLikelihood.parameter_names` 改為逐模型明確分支

原本只處理 `null` 與 `pbh_tunneling`，其餘一律回傳 magnetar 清單。現在每個模型都有
自己的分支，**未涵蓋的模型 raise `ValueError`** 而不是繼承別人的參數。

`grb_frb` 的清單是「模型宣告 ∩ 模擬器實際使用」= `log10_fluence_jy_ms`,
`spectral_index`, `DM`。`effective_parameter_names()` 不再拋 `ValueError`。

**兩點與任務指示不同，必須說明：**

1. **`z` 沒有被放進 `grb_frb` 的取樣清單。** 稽核當時 `z` 會影響資料，是因為
   `grb_frb` 宣告的 `DM` 名字對不上，`_get_dm()` 落到分解式分支才用到 `z`。
   命名統一之後 `_get_dm()` 直接取 `DM` 總量，**`z` 在這個模擬器裡已經沒有任何管道**
   影響動態頻譜（模擬器只在 DM 這一處用到 z）。把一個已量測為死的參數放進取樣維度，
   正是 B3-3／R.3 要避免的事，所以沒有放。**若希望 `z` 有作用，需要決定它應該
   怎麼進入前向模型（例如宇宙學紅移對頻譜或 fluence 的影響），那是建模決策，本輪不做。**
2. **`log10_W_ms` / `log10_tau_sc_ms` 沒有被放進 `grb_frb` 的清單**，因為
   `GRBAfterglowFRB` **根本沒有宣告寬度或散射參數**（它只有 `log10_T90_s`）。
   放進去會讓 `effective_parameter_names()` 再次拋 `ValueError`。見 §R.12。

### R.11.4 測試

新增 `tests/test_radio_forward_model.py`（33 項）：名稱共享、缺鍵 fail-closed、
每個取樣參數擾動後資料與 lnL 都會動、pbh fluence 等於模型推導值（`rel=1e-12`）、
三個模型的 `effective_parameter_names()` 結果、未知模型 raise、grb 仍不可模擬的
現況鎖定。

`tests/test_cli_provenance.py` 有一項需要調整，見 §R.12.3。

**pytest：247 passed, 1 skipped, 1 deselected**（基準 `919e1d3` 是 213 passed，
差額 +34 = 33 項新測試 + 1 項新的跨通道測試；**沒有非預期 regression**）。

---

## R.12 修正過程中浮現的新問題（如實回報，未處理）

### R.12.1 `GRBAfterglowFRB` 沒有宣告任何寬度或散射參數

模擬器需要 `log10_W_int_ms` 與 `log10_tau_sc_ms`，`grb_frb` 兩個都沒有——它只有
`log10_T90_s`（先驗 `uniform(-3, 3)`，即 1 ms 到 1000 s）。fail-closed 之後：

```
KeyError: EMBurstSimulator requires parameter 'log10_W_int_ms', which the model
did not provide. Got parameters: ['DM', 'log10_T90_s', 'log10_fluence_jy_ms',
'spectral_index', 'z']
```

**`grb_frb` 仍然不能在這條通道上被模擬**，但失敗點從「likelihood 要了三個不存在的
參數」變成「模型缺少前向模型需要的寬度參數」，而且錯誤訊息直接指名缺什麼。

把 `log10_T90_s`（秒）當成脈衝寬度是一個看似自然的對應，但它的先驗上界 1000 s
對一個電波爆發寬度而言不合理，**要不要這樣對應、先驗要不要跟著改，是建模決策，
本輪不做也不建議。**

### R.12.2 `_load_mock()` 不檢查模型宣告的通道與模擬器是否相符

`dataio/loader.py::_load_mock()` 會把**任何**模型丟給**任何**通道的模擬器。
修正前這被預設值遮蔽（把 GW 模型 `bh_ringdown` 注入 radio 通道，會產生一個
W=10 ms、DM=150、α=−1.5、fluence=1 Jy·ms 的爆發）；fail-closed 之後會 raise。

**要不要在 `_load_mock()` 加通道相容性檢查，是設計決策，本輪不做。**

### R.12.3 因此調整的既有測試

`tests/test_cli_provenance.py::test_inject_model_defaults_to_fit_model_via_loader`
原本就是把 `bh_ringdown` 注入 radio 通道，只斷言 provenance 記錄了
`inject_model`。它之所以能通過，**完全依賴 R.4 那個被移除的靜默預設行為**。

處理方式：改名為 `test_inject_model_recorded_in_provenance_via_loader` 並改用
同通道的 `magnetar`（斷言意圖不變），另外新增
`test_cross_channel_injection_now_fails_closed` 把新行為鎖住。
**這是唯一一項既有測試的修改**，理由與新行為都寫在測試的 docstring 裡。

### R.12.4 次像素寬度的脈衝會整個消失，不只是「無法解析」

§R.7 記錄了 `log10_W_int_ms` 有 11.88% 的抽樣低於一個時間格。撰寫測試時實測到
一個更強的後果：時間軸 `linspace(-t_start, t_end, n_time)` **不一定包含 t = 0**，
而脈衝恆定置中在 t = 0。當 σ_t 遠小於格寬時，最近的取樣點落在
`exp(-0.5 (Δt/σ_t)²)` 的極遠尾巴上——實測一組 `W_obs = 0.1525 ms`、格寬 2.353 ms
的設定，訊號在浮點下**精確為 0**（`data − noise_realisation` 全 0）。
也就是說那 11.88% 不只是「解析不出來」，而是可能**完全沒有訊號被注入**。
本輪未處理，僅記錄為 §R.7 的加強版。

---

## R.13 R.6 診斷：`burst_fluence_jy_ms()` 的獨立尺度檢查（未修正）

**這是純診斷。沒有修改這個函式，也沒有調整任何先驗。**

### R.13.1 逐項單位核對

以 `M = 10¹⁵ g`、`η_r = 10⁻³`、`z = 0.1` 逐步重算，與函式輸出比對：

| 步驟 | 值 |
|---|---|
| `M_g × 1e-3` | 1.000000×10¹² kg |
| `E_tot = M c²` | 8.987552×10²⁸ J |
| `η_r · E_tot` | 8.987552×10²⁵ J |
| `D_L(z = 0.1)` | 477.522094 Mpc = 1.473479×10²⁵ m |
| `4π D_L²` | 2.728336×10⁵¹ m² |
| `η_r E / (4π D_L² Δν)` | 3.294151×10⁻³⁵ J m⁻² Hz⁻¹ |
| `/ (JY × 1e-3)` | 3.294151×10⁻⁶ Jy·ms |
| **函式回傳** | **3.294151×10⁻⁶ Jy·ms** |

逐項檢查結果：

- `M_g`（克）→ kg 的 `1e-3` **正確**。
- `E = Mc²` 用的 `C = 2.997925×10⁸ m/s` **正確**。
- `1 Jy·ms = 10⁻²⁶ W m⁻² Hz⁻¹ × 10⁻³ s = 10⁻²⁹ J m⁻² Hz⁻¹`，程式除以
  `JY × 1e-3 = 1.000000×10⁻²⁹` **正確**。
- `_dl_mpc()` 用 `H0 = 67.4`、`Ω_m = 0.315`、`Ω_Λ = 0.685` 的平坦 ΛCDM 梯形積分，
  z = 0.1 給 477.52 Mpc，與標準 Planck-類宇宙學的 ~475 Mpc 差 <1%，**正確**。

**在我檢查的範圍內，沒有找到任何單位換算錯誤。**

### R.13.2 但有兩項建模假設值得標注（都不是數量級來源）

1. **`W_obs_ms` 在函式內被計算，但完全沒有出現在回傳式裡。** 這是死程式碼。
   對「fluence」（時間積分量）而言與寬度無關本來就是對的物理，所以這不影響數值，
   但計算了卻不用是誤導性的。
2. **`Δν` 被寫死成 1 GHz**，等於假設整個靜止質量能量平攤在 1 GHz 頻寬上；
   而且式子裡**沒有任何 (1+z) 因子**。這兩項合起來的影響是「幾倍」等級，
   不是「幾個數量級」等級。

### R.13.3 樂觀角落：先驗其實包含可偵測的組合

把 `M_g`、`η_r` 取先驗上界、`z` 取先驗下界（最有利組合）：

| 量 | 值 |
|---|---|
| 最樂觀角落 | `log10_M_g = 16`, `log10_eta_r = 0`, `z = 1e-4` |
| 該處 fluence | **3.796142×10⁴ Jy·ms** |
| vs CHIME 下限 0.4 Jy·ms | 比值 9.49×10⁴（**+4.98 dex**） |
| vs CHIME 中位 4 Jy·ms | 比值 9490（**+3.98 dex**） |
| vs CHIME 上限 7 Jy·ms | 比值 5423（**+3.73 dex**） |

**先驗的樂觀端比 CHIME 門檻高 3.7–5.0 個數量級**，不是低。所以「比門檻低 7 個數量級」
描述的是**中位數**，不是先驗的可達範圍。

### R.13.4 哪個先驗主導這 7 個數量級

`fluence ∝ η_r · M / D_L(z)²`，取對數後是
`log10_eta_r + log10_M_g − 2 log10 D_L(z) + 常數`。20000 次抽樣：

| 項 | 先驗跨度 | 與 log10 fluence 的 Pearson r | log 變異數貢獻 |
|---|---|---|---|
| `log10_eta_r` | **9.999 dex** | **+0.6915** | **8.2685** |
| `log10_z` | 4.698 dex | −0.6955 | 1.8571 |
| `log10_M_g` | 3.000 dex | +0.2044 | 0.7568 |

log10 fluence：中位數 **−6.670**，p1 −15.776、p99 **+1.971**，全距 −17.866 到 +4.479。

**主導者是 `log10_eta_r` 的 `uniform(-10, 0)`——10 個數量級的無線電輻射效率先驗。**
它的變異數貢獻是第二名（z）的 4.5 倍、第三名（M_g）的 11 倍。

### R.13.5 SNR ≥ 8 那個尾端的參數組合合不合理

20000 次抽樣中 90 筆（0.450%）達到 SNR ≥ 8：

| 參數 | 尾端中位數 | 先驗中位數 | 尾端範圍 |
|---|---|---|---|
| `log10_M_g` | +15.61 | +14.51 | [+14.59, +15.99] |
| `log10_eta_r` | **−0.4471** | −4.973 | [−1.713, −0.005398] |
| `z` | **1.773×10⁻⁴** | 2.182×10⁻² | [1.01×10⁻⁴, 6.753×10⁻⁴] |

尾端的 fluence 是 362.7 – 3.014×10⁴ Jy·ms。

**這個尾端在物理上是極端的**：`η_r` 中位數 −0.447 表示把約 **36% 的 PBH 靜止質量能量**
轉成同調無線電輻射；`z ≈ 1.8×10⁻⁴` 對應 D_L ≈ 0.79 Mpc，即本星系群之內。

換個角度，固定在先驗最大質量與**最大可能效率 η_r = 1**，可偵測距離上限是：

| 門檻 | 最大 z | 對應 D_L |
|---|---|---|
| 0.4 Jy·ms | 0.030123 | 137.0 Mpc |
| 4 Jy·ms | 0.0096714 | 43.33 Mpc |
| 7 Jy·ms | 0.0073238 | 32.76 Mpc |

把效率降到比較不極端的 `η_r = 10⁻³`：0.4 Jy·ms 只到 z = 9.74×10⁻⁴（4.33 Mpc）、
4 Jy·ms 只到 z = 3.08×10⁻⁴（1.37 Mpc）。

**「若要讓模型在 CHIME 靈敏度內產生可偵測訊號，需要什麼範圍」的量化答案**
（只是資訊，不代表應該調整先驗）：在 `η_r ≤ 1`、`M ≤ 10¹⁶ g` 的物理上界下，
`z` 需落在 **≲ 0.01–0.03**（現行 `log_uniform(1e-4, 5)` 先驗中約
42% 的質量在 z < 0.0097 以下）；若同時要求 `η_r` 落在較保守的 10⁻³ 附近，
則 `z` 需 **≲ 10⁻³**（約 21% 的先驗質量）。
從參考點（M = 10¹⁴·⁵ g、η_r = 10⁻⁵、z = 0.1，fluence 1.042×10⁻⁸ Jy·ms）
到 1 Jy·ms 需要 **+7.98 dex**。

### R.13.6 一句話結論

**在本輪檢查範圍內沒有找到任何具體的單位換算錯誤——四項換算（g→kg、`Mc²`、
Jy·ms→J m⁻² Hz⁻¹、平坦 ΛCDM 的 `D_L`）逐項重算都與函式輸出一致到有效位數，
而先驗的最樂觀角落給出 3.796×10⁴ Jy·ms、比 CHIME 門檻高 3.7–5.0 個數量級——
所以這次檢查**傾向支持「這反映的是先驗範圍本身的選擇」而不是公式／單位問題**，
主導者是 `log10_eta_r` 的 `uniform(-10, 0)`（log 變異數貢獻 8.27，是第二名的 4.5 倍）。**

需要一併說明的保留：函式裡 `W_obs_ms` 計算後未使用、`Δν` 寫死 1 GHz、
沒有 (1+z) 因子——這三項都是建模假設而非算術錯誤，影響量級是「幾倍」，
無法解釋 7 個數量級的落差，但它們是否符合意圖仍需要作者確認。
**本節沒有修改這個函式，也沒有調整任何先驗。**

## R.14 本輪產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `tests/test_radio_forward_model.py` | 33 項 radio forward-model 對齊 regression 測試 | 是 |
| `scratchpad/fluence_scale_diag.py` | R.6 尺度診斷腳本（只讀 production 模組） | 否 |

---

# R.15 R.13 三項保留意見的處理，與 `_load_mock()` 跨通道檢查（2026-09-09）

> 本節記錄一次 **production 修正**（`models/pbh_tunneling.py`、`models/__init__.py`、
> `dataio/loader.py`、`cli.py`、`simulators/em_burst.py`）。沒有跑任何 SBC/campaign。
> 本節不含任何天文物理宣稱。

## R.15.1 `W_obs_ms` 死程式碼：查到原始意圖，但**該意圖本身是錯的**——刪除

`git log -S` 顯示這段從初始 commit（`eca27df`）起就沒改過。原始 docstring 寫的是：

```
F ≈ η_r * E_tot / (4π D_L² * Δν * W)
where Δν ≈ 1 GHz (typical bandwidth) and W is the observed width.
```

**所以設計意圖是明確存在的：本來就想除以 W。但那個公式與函式的宣告單位不符。**
量綱檢查：

| 式子 | 量綱 | 是什麼 |
|---|---|---|
| `E /(4π D_L² Δν)` | J m⁻² Hz⁻¹ | **fluence**（時間已積分） |
| `E /(4π D_L² Δν W)` | J m⁻² Hz⁻¹ s⁻¹ = W m⁻² Hz⁻¹ | **flux density**（Jy） |

函式名是 `burst_fluence_jy_ms`、回傳單位宣告是 Jy·ms，而 docstring 的公式其實
是一個 Jy。更關鍵的是消費端：

```python
# simulators/em_burst.py
F_peak_jy = fluence_jy_ms / max(W_obs_ms, 1e-6)
```

**模擬器自己已經除了一次寬度。** 若照 docstring「接上」W，寬度會被除兩次。

**處置：刪除死變數，並改寫 docstring 說明為什麼 fluence 不含寬度。**
這是本輪唯一一項「找到意圖但判定不該接線」的項目——接上去會製造 bug，不是修好 bug。
連帶把 `EMBurstSimulator._PBH_FLUENCE_KEYS` 從 5 個鍵縮成 3 個
（`log10_M_g`, `log10_eta_r`, `z`），因為推導不再需要寬度參數。

## R.15.2 `Δν` 寫死 1 GHz → 改用 CHIME 實際頻寬

`configs/instruments/chime.yaml` 早就有 `observing.freq_low_mhz: 400.0` /
`freq_high_mhz: 800.0`，即 **Δν = 400 MHz**。原本寫死的 1 GHz 不是本專案分析的
任何一個頻帶。

新增 `CHIME_BAND_LOW_MHZ` / `CHIME_BAND_HIGH_MHZ` / `RADIO_BANDWIDTH_HZ = 4.0e8`，
並讓 `burst_fluence_jy_ms(params, delta_nu_hz=RADIO_BANDWIDTH_HZ)` 可覆寫。

**為什麼是模組常數而不是在模型裡讀 YAML**：沿用 GW 側既有慣例
（`alternatives.py` 的 `BAND_LOW_HZ` / `BAND_HIGH_HZ` 也是常數 + 註解指向
`ligo.yaml`），避免物理模型做 import-time 檔案 IO。**但這次多加了一項
GW 側沒有的保障**：`tests/test_radio_fluence_physics.py` 會實際讀
`chime.yaml` 斷言常數與它一致，所以 YAML 仍是 single source of truth，
漂移會被測試抓到。

數值影響：**×1e9/4e8 = 2.5 倍**，與所有其他參數無關。

## R.15.3 `(1+z)`：`D_L` 已經正確，但**頻寬轉換確實少了一個獨立因子**——補上

先確認 `_dl_mpc()`：

```python
d_c_mpc = (c_km_s / H0) * chi_integral
return float((1.0 + z) * d_c_mpc)
```

**距離的紅移修正已經是對的**，這就是標準平坦宇宙光度距離 `D_L = (1+z) D_C`。
所以「報告提到 (1+z) 就加一個」是錯的做法。

但獨立推導顯示還缺一個來自**時間與頻寬轉換**的因子：

1. 靜止系總能量 `E`，`L(t_e)` 滿足 `E = ∫L dt_e`。
2. 觀測流量 `F(t_o) = L/(4π D_L²)`，而 `dt_o = (1+z) dt_e`。
3. 因此**波段積分後的能量 fluence** `∫F dt_o = (1+z) E/(4π D_L²)`
   ——`D_L` 的定義針對的是**光度**（每單位時間），對已對時間積分的 fluence
   而言那個時間膨脹因子要還回來。
4. 再除以**觀測**頻寬 `Δν_obs` 得到每單位頻率的 fluence。

即 `F_ν = (1+z) η_r E / (4π D_L² Δν_obs)`。這與 FRB 能量學常用的
`E = 4π D_L² F_ν Δν / (1+z)` 反解一致（測試 `test_matches_the_standard_frb_energetics_relation`
就是把這個關係反算回 `E` 驗證，`rel=1e-12`）。

**處置：補上這一個 `(1+z)`，並在 docstring 明確寫清楚它不是距離修正。**
數值影響：`×(1+z)`，先驗內從 1.0001（z=1e-4）到 6（z=5）。

## R.15.4 `_load_mock()` 跨通道檢查

`cli.py::_get_likelihood` 本來就有一張相容性表，但它只擋**擬合**side；
**注入** side（`dataio/loader.py::_load_mock`）完全沒有檢查（§R.12.2）。

處置：把那張表抽到 `models/__init__.py` 成為
`CHANNEL_COMPATIBILITY` + `check_model_channel()`，**注入與擬合兩側共用同一張表**，
`cli.py` 改為呼叫它（行為不變），`_load_mock()` 在模擬之前先檢查。

`radio` 被列入 `xray` 的相容集合是既有設計（`PBHTunnelingWhiteHole` 帶有
X 光模擬器要用的 gamma 效率），照抄未改。未知的資料通道**接受空集合**
（fail-closed），所以新增通道而忘了決定相容性會報錯，不會安靜放行。

`tests/test_cli_provenance.py::test_cross_channel_injection_now_fails_closed`
的預期例外因此從 `KeyError`（模擬器缺鍵）改為 `ValueError`（通道不符），
現在根本到不了模擬器。

## R.15.5 數值影響：修正前後對照

單點比較（`ratio = 2.5 × (1+z)`，逐點吻合到 1e-12）：

| 參數組 | 修正前 [Jy·ms] | 修正後 [Jy·ms] | 倍率 |
|---|---|---|---|
| 樂觀角落 M=10¹⁶ g, η_r=1, z=1e-4 | 3.796142×10⁴ | **9.491305×10⁴** | **2.50025** |
| 中位數附近 M=10¹⁴·⁵, η_r=1e-5, z=0.022 | 2.399975×10⁻⁷ | 6.131937×10⁻⁷ | 2.55500 |
| §R.13.1 的工作案例 M=10¹⁵, η_r=1e-3, z=0.1 | 3.294151×10⁻⁶ | 9.058915×10⁻⁶ | 2.75000 |
| 高 z 端 M=10¹⁵, η_r=1e-3, z=5 | 3.304671×10⁻¹⁰ | 4.957007×10⁻⁹ | 15.00000 |

全先驗（20000 抽樣）：

| 量 | 修正前 | 修正後 | 變化 |
|---|---|---|---|
| log10 fluence 中位數 | −6.670 | **−6.198** | **+0.472 dex** |
| log10 fluence p1 / p99 | −15.776 / +1.971 | −14.890 / +2.369 | +0.886 / +0.398 |
| 全距 | −17.866 – +4.479 | −16.702 – +4.877 | — |
| 隱含 SNR ≥ 1 比例 | 0.01355 | 0.02210 | — |
| 隱含 SNR ≥ 8 比例 | 0.00465 | 0.00785 | — |
| 樂觀角落 vs CHIME 0.4/4/7 | +4.98 / +3.98 / +3.73 dex | **+5.38 / +4.38 / +4.13 dex** | — |
| η_r=1, M=10¹⁶ 時 4 Jy·ms 的最遠距離 | z=0.00967（43.3 Mpc） | z=0.01534（69.0 Mpc） | — |

## R.15.6 R.6 結論是否需要重新檢視：**不需要**

以 CHIME 中位門檻 4 Jy·ms 為基準：修正前中位數落差 **7.27 dex**，
修正後 **6.80 dex**。**這三項修正合計解釋了約 0.47 dex，佔整個落差的 6.5%。**

先驗變異數的主導者不變：

| 項 | r（與 log10 fluence） | log 變異數 |
|---|---|---|
| `log10_eta_r` | +0.7084（原 +0.6915） | 8.2685 |
| `log10_z` | −0.6770（原 −0.6955） | 1.8571 |
| `log10_M_g` | +0.2097（原 +0.2044） | 0.7568 |

**§R.13.6 的結論維持成立：落差主要反映的是先驗範圍（尤其
`log10_eta_r` 的 `uniform(-10, 0)`）的選擇，不是公式或單位問題。**
差別在於原本列為「未確認」的三項保留意見，現在兩項已修正、一項
（`W_obs_ms`）確認為應刪除而非應接線。

新增測試 `tests/test_radio_fluence_physics.py`（11 項）中的
`TestCorrectionsStayFactorLevel` 直接把這件事鎖住：全先驗的修正倍率
下界 ≥ 2.5、上界 < 2 dex、中位數 < 10。

**pytest：265 passed, 1 skipped, 1 deselected**（基準 `824392e` 是 247 passed，
差額 +18 = 11 項 fluence 物理測試 + 7 項通道相容性測試；**沒有非預期 regression**）。

## R.15.7 本節產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `tests/test_radio_fluence_physics.py` | 11 項 fluence 公式物理契約測試（含 chime.yaml 一致性斷言） | 是 |
