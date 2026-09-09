# xray / image 通道 forward-model 一致性前置稽核

## 範圍聲明

**這份文件記錄的是 xray（X 光光變曲線）與 image/VLBI（EHT 視相位／可見度）兩條通道在
執行任何 SBC/coverage campaign 之前的 forward-model 一致性稽核結果。不構成任何白洞訊號
偵測或未偵測的科學宣稱，也不包含任何天文物理結論。**

WhiteSearch 是 candidate ranking engine（候選訊號排序引擎），不是白洞證明器。本輪
**沒有執行任何 SBC、campaign、dynesty 或取樣**，只做逐參數擾動的數值比對與結構性檢查，
全部使用 mock 模擬資料，未使用任何真實 Chandra／XMM／EHT 觀測資料。

**依 fail-closed 原則，本輪只回報現況與量測數字，沒有修改任何 production 程式碼。**

方法比照 `docs/BOUNCE_PREFLIGHT_AUDIT.md` 的 B.1／B.2／B.5 與
`docs/RADIO_PREFLIGHT_AUDIT.md` 的 R.1／R.2／R.7。

## 為什麼兩條通道合併成一份文件

**先更正一個可能的預期**：xray 與 image **並沒有共用 likelihood 或模擬器**——

| | xray | image |
|---|---|---|
| 模擬器 | `simulators/em_burst.py::XRayLightCurveSimulator` | `simulators/image_shadow.py::ImageShadowSimulator` |
| likelihood | `likelihoods/em_likelihood.py::XRayBurstLikelihood`（Poisson） | `likelihoods/visibility.py::VisibilityLikelihood`（Gaussian + von Mises） |
| 共同祖先 | 僅 `likelihoods/base.py::BaseLikelihood` | 同左 |

合併的理由是**它們共用的是外圍管線**（`dataio/loader.py`、
`models.CHANNEL_COMPATIBILITY`、`BilbyRunner.effective_parameter_names()`、
preprocess 的接線狀況、真實資料路徑），而本輪最主要的發現正好落在那些共用部分；
分成兩份會把同一段分析寫兩次。**逐參數擾動則是兩條通道各自的模型分開做的。**

---

## X.0 鏈路

| 層 | xray | image |
|---|---|---|
| dataio | `heasarc.py`（`chandra.py` / `xmm.py` 亦存在） | `eht.py` |
| loader 分支 | `source in ("heasarc", "eht")` → 未實作，fail-closed | 同左 |
| preprocess | `xray_preprocess.py::XRayPreprocessor` | **無** |
| 模擬器 | `XRayLightCurveSimulator`（`channel = "xray"`） | `ImageShadowSimulator`（`channel = "image"`） |
| likelihood | `XRayBurstLikelihood` | `VisibilityLikelihood` |

**宣告 `channel` 的模型：**

| 通道 | 原生模型（`channel` 等於該通道） | `CHANNEL_COMPATIBILITY` 允許的模型 |
|---|---|---|
| xray | **無** | `pbh_tunneling`, `null`, `magnetar`, `grb_frb`（經 `radio`/`generic` 相容） |
| image | `gr_eternal`, `bh_accretion` | 上述兩個 + `null` |

**xray 通道沒有任何原生模型**——`cli.py` 的 `_default_alt_model` 把 xray 對到
`pbh_tunneling`（一個 `channel = "radio"` 的模型），走的是
`CHANNEL_COMPATIBILITY["xray"]` 允許 `radio` 的既有設計。

稽核用 context（各模擬器的宣告預設值）：

```
xray : e_low_kev 0.5, e_high_kev 10.0, area_cm2 1000, bg_rate_cps 0.5,
       duration_s 100.0, dt_s 1.0     -> 100 個 bin，爆發峰值固定在 t = 20 s
image: thermal_noise_jy 0.05, freq_ghz 230.0, fov_muas 200.0, n_pixels 128
       -> 像素 3.1250 μas，16 條基線，5 個 closure triangle
```

---

## X.1 阻斷性問題 (I)：**xray 通道沒有任何模型能建立先驗**

`XRayBurstLikelihood.parameter_names` 是**不看 `model_name` 的固定清單**
（與 `RadioBurstLikelihood` / `GWLikelihood` 的逐模型分支不同）：

```python
return ["log10_fluence_erg_cm2", "log10_T90_s", "log10_eta_gamma"]
```

對四個被允許的模型逐一檢查 `effective_parameter_names()`：

| 模型 | 模型宣告數 | likelihood 要但模型沒有 | 結果 |
|---|---|---|---|
| `pbh_tunneling` | 10 | `log10_fluence_erg_cm2`, `log10_T90_s` | **`ValueError`** |
| `grb_frb` | 5 | `log10_fluence_erg_cm2`, `log10_eta_gamma` | **`ValueError`** |
| `magnetar` | 7 | 三個全缺 | **`ValueError`** |
| `null` | 0 | 三個全缺 | **`ValueError`** |

**registry 裡沒有任何模型宣告 `log10_fluence_erg_cm2`。** 所以 xray 通道目前
**四個被允許的模型全部無法建立先驗**，這條通道整體不可執行。交集機制正確地
fail-closed（與 `grb_frb` 在 radio 上的 R.5 同一類），但代價是通道不可用。

附帶：`log10_eta_gamma` 在 likelihood 清單裡，但 `XRayLightCurveSimulator.simulate()`
**從未讀取它**（原始碼字串檢查：`log10_fluence_erg_cm2` True、`log10_T90_s` True、
`log10_eta_gamma` **False**）。即使先驗能建起來，它也會是死參數。

---

## X.2 xray 逐參數活性檢查（B.2 / R.2 對應項）

方法：抽先驗樣本，固定雜訊種子模擬 base 資料；擾動先驗跨度的 30%，同種子重模擬，
量 `max|Δdata|`；在同一份 base 資料上比 lnL。三次抽樣取中位數。
（先驗雖然建不起來，`loglike()` 本身仍可直接呼叫，所以活性可以量。）

### `pbh_tunneling` on xray

| 參數 | 在 likelihood 清單 | `max|Δdata|` | `|ΔlnL|` |
|---|---|---|---|
| 全部 10 個參數 | 僅 `log10_eta_gamma` 是 | **全部 0** | **全部 0** |

**十個參數全部死亡。** 成因很直接：模擬器只讀 `log10_fluence_erg_cm2` 與
`log10_T90_s`（都用 `params.get(key, default)`），`pbh_tunneling` 兩個都沒宣告，
所以每次抽樣都用預設值 −7.0 與 0.0，產生一條與參數完全無關的固定光變曲線。
**這是 R.4 那個「安靜吃預設值」模式在 xray 上的版本**，而且是全滿版。

### `grb_frb` on xray

| 參數 | 在 likelihood 清單 | `max|Δdata|`（counts） | `|ΔlnL|` |
|---|---|---|---|
| `log10_fluence_jy_ms` | 否 | **0** | **0** |
| `log10_T90_s` | 是 | **599** | **3273.59** |
| `z` | 否 | 0 | 0 |
| `spectral_index` | 否 | 0 | 0 |
| `DM` | 否 | 0 | 0 |

只有 `log10_T90_s` 是活的。`log10_fluence_jy_ms` 死掉不是命名筆誤——
Jy·ms（電波 fluence）與 erg/cm²（X 光 fluence）**是不同的物理量**，
需要一個轉換而不是改名。這與 R.4 的純命名不一致要分開看。

---

## X.3 xray 先驗支撐（B.5 / R.7 對應項）

以 `grb_frb` 唯一活著的 `log10_T90_s`（`uniform(-3, 3)` → 1 ms – 999.6 s）對照
100 s／1 s bin 的光變曲線（20000 抽樣）：

| 檢查 | 比例 |
|---|---|
| `T90` 小於一個時間格（1 s） | **0.4996** |
| `T90` 比整段光變曲線還長（>100 s） | **0.1630** |
| `sigma_t` 小於一個時間格 | **0.6132** |
| 爆發被 t = 0 截掉（`t_peak − 3σ < 0`，峰值固定在 20 s） | **0.2460** |

**約一半的先驗抽樣其爆發短於一個時間格**，另有 16.3% 長於整段資料。
與 radio R.7 同類型（時間尺度先驗跨度遠大於儀器可表示範圍），但比例更極端。
爆發位置固定在片段 20%，沒有到達時間參數。

---

## X.4 image 結構性檢查（B.1 / R.1 對應項）

`VisibilityLikelihood` **不接受 `model_name` 參數**（`__init__` 只有
`use_closure_phases` / `closure_kappa` / `robust_data`），`parameter_names`
是固定的 7 個。

| 模型 | 模型宣告 | likelihood 要但模型沒有 | 模型有但 likelihood 忽略 | `effective_parameter_names()` |
|---|---|---|---|---|
| `gr_eternal` | 9 | 無 | `log10_ne`, `log10_B` | **7（正常）** |
| `bh_accretion` | 6 | `position_angle`, `ring_width_frac`, `log10_brightness` | `log10_mdot_edd`, `jet_power_frac` | **`ValueError`** |
| `null` | 0 | 七個全缺 | 無 | **`ValueError`** |

**`gr_eternal`（走這條通道的白洞模型）結構完全正確**：交集 7 個，
`log10_ne` / `log10_B` 被正確排除（幾何環模型不含電漿物理）。

`bh_accretion` 與 `null` 無法建立先驗——同 R.5 家族。
`bh_accretion` 缺的三個參數，模擬器都會用 `params.get(..., default)` 補
（`position_angle` → 0、`ring_width_frac` → 0.1、`log10_brightness` → 0），
所以若繞過交集檢查直接模擬，它會跑出一個由預設值決定的結果——**R.4 模式**。

**`null` 在 xray 與 image 兩條通道都無法建立先驗**：`GWLikelihood` 與
`RadioBurstLikelihood` 都有 `if model_name == "null": return []` 分支，
`XRayBurstLikelihood` 與 `VisibilityLikelihood` 都沒有（後者根本收不到 model_name）。

---

## X.5 image 逐參數活性檢查

### 第一次量測（無條件抽樣）：**全部參數死亡**

| 模型 | 結果 |
|---|---|
| `gr_eternal` | 9 個參數 `max|Δdata|` 與 `|ΔlnL|` **全部為 0** |
| `bh_accretion` | 6 個參數**全部為 0** |

這個結果**不是接線問題**，而是先驗支撐問題造成的，見 §X.6。直接證據：

| 抽樣 | `M` | `D_L` | 環半徑 | `max|image|` | `max|vis_signal|` |
|---|---|---|---|---|---|
| 0 | 2.423×10⁶ | 44.6 | 0.00276 μas | **0** | **0** |
| 1 | 2.253×10⁹ | 590.8 | 0.195 μas | **0** | **0** |
| 2 | 3.63×10⁸ | 63.93 | 0.289 μas | 1.63×10⁻²⁹⁹ | 3.18×10⁻²⁹⁸ |
| 3 | 1.636×10⁸ | 2.735 | 3.06 μas | 2.62×10⁻⁵ | 5.21×10⁻⁴ |

像素是 3.1250 μas；環半徑遠小於一個像素時，高斯環整個落在取樣點之間，
影像在浮點下**精確為 0**（或下溢到 10⁻²⁹⁹），可見度就只剩雜訊。

### 第二次量測（限定環半徑落在 [10, 150] μas 的可表示抽樣）

| 模型 | 參數 | 在 likelihood 清單 | `max|Δdata|` [Jy] | `|ΔlnL|` |
|---|---|---|---|---|
| `gr_eternal` | `M` | 是 | 0.328802 | 346.847 |
| | `a_star` | 是 | 0.0110777 | 1.35957 |
| | `D_L` | 是 | 0.319646 | 330.519 |
| | `i` | 是 | 0.458839 | 597.415 |
| | `position_angle` | 是 | 0.0479528 | 3.82494 |
| | `ring_width_frac` | 是 | 1.0687 | 3474.11 |
| | `log10_brightness` | 是 | 20.4172 | 1.33007×10⁶ |
| | `log10_ne` | 否 | **0** | **0** |
| | `log10_B` | 否 | **0** | **0** |
| `bh_accretion` | `M` / `a_star` / `D_L` / `i` | 是 | 632.798 / 2.85336 / 632.725 / 709.682 | 1.28×10⁹ / 2.65×10⁴ / 1.28×10⁹ / 1.61×10⁹ |
| | `log10_mdot_edd`, `jet_power_frac` | 否 | **0** | **0** |

**在環可被表示的區域，`gr_eternal` 的 7 個取樣參數全部同時影響資料與 lnL，
方向一致，沒有單邊生效的情形。** 唯二死掉的 `log10_ne` / `log10_B` 已被
likelihood 正確排除，不是問題。

---

## X.6 阻斷性問題 (II)：image 的先驗支撐——**80.98% 的抽樣環半徑小於一個像素**

環角半徑 `_shadow_radius_muas(M, a_star, D_L)`。`gr_eternal` 與 `bh_accretion`
的 `M`（`log_uniform(1e6, 1e10)` M⊙）與 `D_L`（`log_uniform(1, 2000)` Mpc）
先驗相同，20000 抽樣：

| 量 | 值 |
|---|---|
| 環半徑範圍 | 2.759×10⁻⁵ – 464.4 μas |
| p1 / 中位數 / p99 | 8.275×10⁻⁵ / **0.1214** / 165.8 μas |
| **小於一個像素（3.1250 μas）** | **0.8098** |
| 大於 FoV 半寬（200 μas） | 0.0073 |
| **落在 [1 像素, FoV] 之間（可被表示）** | **0.1829** |

**中位數環半徑 0.1214 μas 是像素尺度的 1/26。** 這與 GW 通道 B.5
（`M` 先驗支撐超出可分析頻帶）、radio R.7（`W_int` 次格寬）是同一類型，
但**程度是三者中最極端的**：只有 18.29% 的先驗質量落在儀器可表示的範圍內，
而且落在外面的部分不會被拒絕，只是安靜地產生全零影像。

（`bh_accretion` 的比例完全相同，因為它的 `M` / `D_L` 先驗與 `gr_eternal` 一致。）

---

## X.7 真實資料路徑

| 檢查 | xray | image |
|---|---|---|
| `loader.py` 分支 | `heasarc` | `eht` |
| 未加 `allow_mock_fallback` 時 | **`DataLoadError`（fail-closed，正確）** | 同左 |
| 加了之後 | 走 `_load_mock`，provenance 記為 `MOCK_FALLBACK` | 同左 |
| 資料格式是否與 likelihood 相容 | 未接線，無從比對 | **相容** |

**xray**：`chandra.py::ChandraLoader` 與 `xmm.py::XMMLoader` 存在且由
`dataio/__init__.py` 匯出，但 **`load_observation_data()` 完全沒有它們的分支**
（只有 `mock` / `gwosc` / `chime` / `heasarc` / `eht`），`cli.py` 的
`DATA_CHOICES` 也不含它們。`HEASARCLoader` 提供的是
`query_region` / `query_time_interval` / `load_event_list` / `estimate_background`，
與 `XRayBurstLikelihood` 需要的 counts 陣列之間沒有轉接層。
**xray 目前只有 mock 路徑。**

**image**：與 radio 的 CHIME 情況**不同，這裡格式是相容的**。實測把
`EHTLoader._mock_eht_data('M87', 2017, 'LO')` 的 dict 直接餵給
`VisibilityLikelihood.loglike()` 會回傳有限值（−3959.39），因為該 dict 帶有
`visibilities` / `sigma` / `uv_coverage`，正是 likelihood 讀的鍵。
**但 image 仍然只有 mock 路徑**（`eht` 分支未實作），且有兩點要記錄，見 §X.8。

**兩條通道的 preprocess 都不在 likelihood 路徑上**：`XRayPreprocessor` 除了
`preprocess/__init__.py` 的匯出之外**沒有任何呼叫端**（與 `RadioPreprocessor`
相同的狀況）；image 通道**根本沒有 preprocess 模組**。

---

## X.8 非阻斷、但應記錄的事項（image）

1. **`VisibilityLikelihood` 沒有把資料自己的 `uv_coverage` 轉給模擬器。**
   它呼叫 `sim.simulate(theta, context, ...)`，而 `ImageShadowSimulator` 讀的是
   `context.get("uv_coverage", _default_eht_uv())`——資料 metadata 裡的
   `uv_coverage` **不會**被使用。目前**沒有實際造成偏差**，因為
   `_mock_eht_data()` 用的就是同一組預設基線（實測兩者 shape 都是 (16, 2)
   且 `np.allclose` 為真，帶不帶 `uv_coverage` 進 context 算出的 lnL
   完全相同：−3169.5309826139687）。**但這是資料層面的巧合**：任何真實
   uvfits 載入若有不同基線，模型與資料就會在不同的 (u,v) 點上被逐項相比。
   這是 GW B.4 那類 forward-model 不一致的**潛在**版本。
2. **closure phase 在缺鍵時安靜失效。** likelihood 宣告
   `log L = log L_amp + log L_phase`，但 `meta.get("closure_phases", None)`
   為 `None` 時該項直接是 0 且無任何記錄。EHT loader 的 record **沒有**
   `closure_phases` 鍵（實測鍵為 `freq_ghz, sigma, source, stations, u,
   uv_coverage, v, visibilities`），所以真實資料路徑一旦接上，會安靜地
   退化成只有振幅。mock SimData 路徑則有（n = 5），且確實有貢獻：
   amp+phase = −9.906，amp only = +28.868。

---

## X.9 稽核總結

| # | 問題 | 通道／模型 | 類型 | 阻斷性 |
|---|---|---|---|---|
| X.1 | `XRayBurstLikelihood.parameter_names` 是不看模型的固定清單，**四個被允許的模型全部拋 `ValueError`**；沒有任何模型宣告 `log10_fluence_erg_cm2` | xray 全部 | 取樣維度 | **是（通道不可執行）** |
| X.1 | `log10_eta_gamma` 在清單裡但模擬器從未讀取 | xray | 死參數 | 是 |
| X.2 | 模擬器 `params.get(key, default)` 讓 `pbh_tunneling` 的 **10 個參數全部**產生零變化 | xray / `pbh_tunneling` | 死參數 | **是** |
| X.2 | `log10_fluence_jy_ms`（Jy·ms）與 `log10_fluence_erg_cm2`（erg/cm²）是不同物理量，缺轉換 | xray / `grb_frb` | 建模 | 需決策 |
| X.3 | `T90` 先驗 49.96% 短於一個時間格、16.30% 長於整段資料、24.60% 被 t=0 截斷 | xray | 先驗支撐 | 需決策 |
| X.4 | `bh_accretion` 與 `null` 無法建立先驗 | image | 取樣維度 | **是** |
| X.6 | **80.98% 的先驗抽樣環半徑小於一個像素**，安靜產生全零影像；只有 18.29% 可被表示 | image 全部 | 先驗支撐 | **是** |
| X.7 | `chandra.py` / `xmm.py` 存在但 `loader.py` 沒有分支；xray 只有 mock 路徑 | xray | 無真實資料路徑 | 需決策 |
| X.8.1 | likelihood 不轉發資料的 `uv_coverage`，真實載入會讓模型與資料落在不同基線上 | image | 潛在 forward-model 不一致 | 否（目前未觸發） |
| X.8.2 | closure phase 缺鍵時安靜退化成只有振幅 | image | 安靜降級 | 否 |
| X.7 | 兩條通道的 preprocess 都不在 likelihood 路徑上（xray 無呼叫端、image 無模組） | 兩者 | 未接線 | 否 |

**確認一致、沒有問題的部分：**

- **`gr_eternal`（image 通道的白洞模型）的 forward-model 對齊是乾淨的**：
  在環可被表示的區域，7 個取樣參數全部同時影響模擬資料與 lnL，
  沒有任何單邊生效；`log10_ne` / `log10_B` 被 likelihood 正確排除。
  這是本輪四個模型-通道組合裡唯一結構完整的一個。
- **`effective_parameter_names()` 的交集機制在兩條通道都正確 fail-closed**：
  該擋的（`bh_accretion`、`null`、四個 xray 模型）都擋下來了，
  該通過的（`gr_eternal` 10 → 7，`log10_ne`/`log10_B` 正確排除）也通過了。
- **真實資料來源的 fail-closed 是正確的**：`heasarc` 與 `eht` 在未明確允許
  fallback 時拋 `DataLoadError`，允許之後 provenance 記為 `MOCK_FALLBACK`。
- **image 的資料格式在 loader record 與 likelihood 之間是相容的**
  （與 radio/CHIME 的目錄表不相容形成對比）。
- image 的 mock 路徑本身（模擬 → likelihood）在可見度陣列形狀、`sigma` 傳遞、
  closure phase 計算（n = 5 個 triangle）上都自洽。

### 為什麼現在不跑 campaign

- **xray**：沒有任何模型能建立先驗，通道不可執行；即使繞過，`pbh_tunneling`
  的 10 個參數全部是死的，SBC 會得到全均勻的 rank——「通過」一個完全空的模型。
- **image**：`gr_eternal` 的接線是乾淨的，但 80.98% 的先驗抽樣不會產生任何訊號。
  在那個狀態下跑 SBC，多數注入的 posterior 會等於 prior，rank 均勻，
  校準檢查同樣會給出誤導性的「通過」。

**本輪未修改任何 production 程式碼，也不提出修法選擇。等決策。**

## X.10 產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `docs/calibration/xray_image_preflight_parameter_activity.csv` | 四個模型-通道組合全部宣告參數的擾動前後資料變化、`ΔlnL`、活性旗標（無條件抽樣版） | 是 |
| `scratchpad/xi_audit.py` | 稽核腳本（只讀 production 模組） | 否 |
