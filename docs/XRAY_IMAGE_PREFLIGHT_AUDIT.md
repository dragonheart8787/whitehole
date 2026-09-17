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

---

# X.11 image 通道修正輪（2026-09-09）

> 本節記錄一次 **production 修正**（`likelihoods/visibility.py`、
> `models/gr_eternal.py`、`cli.py`）。**範圍限定 image 通道，沒有動 xray 的任何程式碼。**
> 沒有跑任何 SBC/campaign。本節不含任何天文物理宣稱。

## X.11.1 X.6 環半徑先驗：**先驗未收窄，回報 tension**

### 可表示範圍的推導（已進程式碼）

新增 `IMAGE_FOV_MUAS` / `IMAGE_N_PIXELS` / `IMAGE_PIXEL_MUAS`（引用
`configs/instruments/eht.yaml` 的 `imaging.{fov_muas, n_pixels}`，並由
`tests/test_image_forward_model.py` 斷言與該 YAML 一致），以及
`GREternalWhiteHole.ring_radius_representable_range_muas()`
（比照 bounce 的 `_m_prior_bounds_for_band()`）。

**下界不是「一個像素」那麼簡單。** `_gaussian_ring_image()` 產生的是
`exp(-0.5 ((r_grid − r_ring)/(r_ring · ring_width_frac))²)`，所以薄環
（`ring_width_frac = 0.01`）只有在某個網格點剛好落在 `r_ring` 附近時才會被取到。
掃描 `ring_width_frac ∈ [0.01, 0.5]` 的最壞情況（實測，brightness = 1）：

| `r_ring` | frac=0.01 | 0.03 | 0.1 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| 1 μas | 0 | 0 | 2×10⁻³³ | 2.33×10⁻⁴ | 0.0492 |
| 3.125 μas（1 像素） | 5.4×10⁻¹⁸⁰ | 1.21×10⁻²⁰ | 0.0161 | 0.632 | 0.848 |
| 6 μas | 1×10⁻²⁸ | 7.74×10⁻⁴ | 0.525 | 0.931 | 0.975 |
| 12 μas | 0.998 | 1 | 1 | 1 | 1 |
| 25 μas | 0.923 | 0.991 | 0.999 | 1 | 1 |

**而且它不是單調的**：最壞情況在 3 像素是 0.1217、4 像素掉到 0.0003、5 像素回到
0.9749——取決於 `r_ring` 是否與像素網格可公度。這是**取樣假影**，不是平滑的解析度極限。
最壞情況首次穩定超過半個峰值是在 **8.245 像素 = 25.77 μas**，這就是採用的下界；
上界是 FoV 半寬 200 μas。**可表示視窗只有 0.89 dex。**

### 為什麼沒有收窄先驗

影像約束的是環半徑，而環半徑只透過**比值**依賴 `M` 與 `D_L`：

```
r[μas] = 5.130245e-08 * M[M_sun] / D_L[Mpc]
```

獨立的 log-uniform 先驗下，`r` 的跨度是 `dex(M) + dex(D_L)` = 4.000 + 3.301 =
**7.301 dex**，對上 0.89–1.20 dex 的視窗。要達到 ~100% 可表示率，需要
`dex(M) + dex(D_L) ≤ 1.204`，例如**質量與距離各只能跨一個 4 倍區間**。

| 若 `M` 跨 | 則 `D_L` 只能跨 |
|---|---|
| 0.00 dex（×1） | 1.20 dex（×16） |
| 0.30 dex（×2） | 0.90 dex（×8.02） |
| 0.60 dex（×3.98） | 0.60 dex（×4.02） |
| 0.90 dex（×7.94） | 0.30 dex（×2.01） |

**這比這條通道存在的理由本身還窄。** 用程式碼自己的公式算兩個實際 VLBI 目標：

| 目標 | `M` | `D_L` | 環半徑 |
|---|---|---|---|
| M87* | 6.5×10⁹ M⊙ | 16.8 Mpc | **19.667 μas** |
| Sgr A* | 4.15×10⁶ M⊙ | 0.008178 Mpc | **25.795 μas** |

兩者質量差 **3.19 dex**、距離差 **3.31 dex**，環半徑卻都在 20–26 μas——
因為兩者共變。**收窄邊際先驗無法表達這個相關性。**

**額外發現（比 tension 更尖銳）**：推導出的下界 25.77 μas **高於 M87\* 的
19.67 μas**。也就是說，目前出貨的成像網格連這條通道最主要的目標都無法忠實表示
（對薄環而言）；Sgr A* 的 25.80 μas 也只是剛好擦過下界。

**依指示不硬縮先驗。** 結構性的修法是把參數化改到資料真正約束的比值上
（比照 `BOUNCE_PREFLIGHT_AUDIT.md` B3-2 的爆發時序重新參數化），
那是設計決策，本輪不做。推導與這段理由都寫進
`ring_radius_representable_range_muas()` 的 docstring，測試把視窗、
與 YAML 的一致性、以及**目前僅 18.29% 可表示**這個現況都鎖住——
後者標明是 KNOWN DEFICIENCY，若日後重新參數化落地，該測試**應該**失敗並被主動更新。

## X.11.2 X.8.1 修正：`uv_coverage` 改為從資料取得，缺少則 fail-closed

`VisibilityLikelihood.loglike()` 現在用
`_require_uv_coverage(meta, context)` 取得基線：優先讀資料 metadata，
其次讀 context，兩者皆無則 `KeyError`。取得後**複製一份 context** 並塞入
`uv_coverage` 再交給 `ImageShadowSimulator`，所以模型可見度一定建在資料自己的
(u, v) 點上。（複製而非就地修改，測試 `test_context_is_not_mutated` 鎖住。）

原本的 `context.get("uv_coverage", _default_eht_uv())` 會在資料基線與預設不同時，
把模型與資料在不同 (u, v) 點上逐項相減——GW B.4 那類 forward-model 不一致的
image 版本。

## X.11.3 X.8.2 修正：closure phase 不再安靜退化

- `use_closure_phases=True`（預設）現在代表**呼叫端聲明資料帶有 closure phase**，
  缺鍵時 `KeyError` 並提示改用 `use_closure_phases=False`。
- `use_closure_phases=False` 是**明確聲明的 amplitude-only 分析**。
- 兩種情況都寫進 `self.last_closure_config`：
  `used_closure_phase` / `reason` / `n_closure_phases`（比照 GW 的
  `last_taper_config`）。

## X.11.4 X.4 修正：`null` 已接上；`bh_accretion` 回報 trade-off，未實作

`VisibilityLikelihood` 現在接受 `model_name`（預設 `"gr_eternal"`），
並加上 `if self.model_name == "null": return []` 分支——
`GWLikelihood`、`RadioBurstLikelihood`、`XRayBurstLikelihood` 之外的每一條
通道都有這個分支，而整個專案的輸出是 ln BF vs null，**所以 null 必須能在
每條通道上被擬合，設計意圖明確**。`cli.py` 改為 `VisibilityLikelihood(model)`。
`gr_eternal` 的 7 維取樣維度完全不變。

**`bh_accretion` 沒有實作，因為它不是接線問題。** 設計意圖是明確的
（它宣告 `channel = "image"`，且 `cli.py::_default_alt_model` 把 image 對到它），
但它宣告的是 `log10_mdot_edd` / `jet_power_frac`，而**沒有**
`position_angle` / `ring_width_frac` / `log10_brightness`。三個選項的 trade-off：

| 選項 | 做法 | 代價 |
|---|---|---|
| **A** 交集分支 | `["M", "a_star", "D_L", "i"]`，比照 R.5 對 grb_frb 的做法 | 缺的三個幾何參數會落到模擬器預設值（0 / 0.1 / 1.0）——**正是 R.4 的靜默預設模式**；而且 `log10_mdot_edd` / `jet_power_frac` 仍然是死的，`bh_accretion` 變成「凍結三個幾何參數的 `gr_eternal`」，不是一個物理上不同的對立假說 |
| **B** 讓模型補宣告三個幾何參數 | 一樣機械 | `bh_accretion` 在 likelihood 眼中會與 `gr_eternal` **完全相同**，ln BF(WH/accretion) 恆等於 0，對立假說失去意義 |
| **C** 擴充 `ImageShadowSimulator` | 讓 `log10_mdot_edd` / `jet_power_frac` 真的驅動一個吸積盤亮度分布 | 唯一在物理上有意義的路徑，但是實質的建模工作，不是機械修正 |

**依指示回報現況與選項，未自行選定。** 測試
`test_bh_accretion_still_cannot_build_priors` 把現況鎖住並在 docstring 說明原因。

## X.11.5 產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `tests/test_image_forward_model.py` | 16 項 image 通道對齊與 fail-closed 測試 | 是 |

---

# X.12 落地實作：`D_L` context 化（I-2）與 `bh_accretion` 物理實作（I-1）

> 這一節是 X.6 / X.11.1（環半徑先驗）與 X.11.4（`bh_accretion`）兩項設計決定的
> 落地實作紀錄。**不含任何科學結論**；只記錄實作內容、量測到的數字，
> 以及實作過程中發現、但這一輪**沒有**修的問題。
> 沒有跑任何 SBC/campaign/取樣。

## X.12.1（I-2）`D_L` 成為逐目標已知常數

### 機制：比照 GW 通道，不另創一套

GW 通道對「逐事件已知量」的處理在 `GWLikelihood._parse_data`：

```python
t_merger = float(meta.get("t_merger", context.get("t_merger", 0.5)))
low_freq = float(meta.get("low_freq_cutoff", context.get("low_freq_cutoff", 20.0)))
```

亦即 **observation metadata 優先、analysis context 次之**。image 通道沿用同一個
順序，只有一處刻意不同：**沒有第三層預設值**。`utils/targets.py::require_target()`
在兩層都沒有 `target` 時 raise，理由與 `VisibilityLikelihood._require_uv_coverage`
（X.11.2）相同——猜一個目標等於猜一個距離，而兩個目標的距離差 3.313 dex。

### `target` 欄位（而不是重用 `source`）

`EHTLoader` 既有的 record 已經有一個 `source` 鍵，裡面放的是**目標名稱**
（`"M87"` / `"SgrA"`），與 GW / radio 路徑上 `source` 代表**資料來源證跡**
（`GWOSC` / `MOCK_EXPLICIT` / `MOCK_SIMULATOR`）的意義相衝突。
這次**新增獨立的 `target` 鍵**，不動 `source` 的既有語意，並在 `eht.py`
的 docstring 裡把這個碰撞寫清楚。`EHTLoader` 的三條路徑
（`_load_ehtim` / `_load_cache` 以外的 `_mock_eht_data` / `_parse_uvfits_minimal`）
都明確寫入 `target`；`load_from_file()` 現在**要求**呼叫端傳入 `target`
（原本寫死 `source="custom"`），因為檔案本身不會說它是哪個源。

### 距離與質量常數（含出處）

`utils/targets.py::EHT_TARGETS`：

| 目標 | `distance_mpc` | 出處 | 質量先驗 `[low, high]` M_sun | 出處 |
|---|---|---|---|---|
| `M87*` | 16.8 | EHT Collaboration 2019, ApJL 875, L6 | `[3.0e9, 1.0e10]` | EHT 2019 L6 給 (6.5 ± 0.2|stat ± 0.7|sys)e9；先驗放寬到涵蓋該文比較的兩個獨立動力學測量：恆星動力學 6.2e9（Gebhardt+2011, ApJ 729, 119）與氣體動力學 3.5e9（Walsh+2013, ApJ 770, 86），兩者差約 1.8 倍 |
| `SgrA*` | 0.008178 | GRAVITY Collaboration 2019, A&A 625, L10（R0 = 8178 ± 13|stat ± 22|sys pc） | `[3.5e6, 5.0e6]` | GRAVITY 2019 給 (4.154 ± 0.014|stat ± 0.014|sys)e6；先驗放寬到涵蓋 Keck 獨立軌道擬合 3.975e6 ± 0.058e6（Do+2019, Science 365, 664） |

專案原本就在程式與測試裡用 16.8 Mpc 與 0.008178 Mpc，這次把出處補進常數旁。
**先驗寬度是照文獻上的獨立測量訂的，不是照網格可表示範圍反推的**——這一點
很重要，見 X.12.2。

### 取樣維度的變化

| 模型 | 之前 | 之後 |
|---|---|---|
| `gr_eternal` | `M, a_star, D_L, i, position_angle, ring_width_frac, log10_brightness`（7） | `M, a_star, i, position_angle, ring_width_frac, log10_brightness`（6） |
| `bh_accretion` | 無法建立先驗 | `M, a_star, i, position_angle, log10_mdot_edd, jet_power_frac`（6） |

`VisibilityLikelihood.MODEL_PARAMETERS` 改成逐模型查表，未列名的模型名稱 raise
（而不是安靜地繼承環的參數向量）。模型建構改由 `models.model_for_context(name, context)`
統一供應 `target`；`get_model(name)` 不帶 target 仍可建構（`check_model_channel`
需要），但任何需要距離的方法都會 raise。

## X.12.2（I-2）重新推導 M 的可表示先驗——**先驗側問題解決，網格側沒有**

### 先驗側：7.301 dex → 0.523 / 0.155 dex

X.6 記錄的機制是：環半徑只透過比值 `r ∝ M / D_L` 進入影像，所以獨立的
log-uniform 先驗讓環半徑跨 `dex(M) + dex(D_L) = 4.000 + 3.301 = 7.301 dex`，
對上 0.89 dex 的可表示視窗，只有 18.29% 落在裡面、80.98% 產生精確為零的影像。

固定 `D_L` 之後，環半徑的先驗展寬**只剩質量先驗本身**：

| 目標 | 質量先驗寬度 | 環半徑範圍（a\* = 0.998） |
|---|---|---|
| `M87*` | 0.523 dex | 9.07 – 30.24 μas |
| `SgrA*` | 0.155 dex | 21.74 – 31.06 μas |

**先驗側的問題完全消失**：在一個能表示這個尺度的網格上，先驗抽樣不再產生任何
一張精確為零的影像（`test_no_prior_draw_produces_an_empty_image_on_a_working_grid`
逐張檢查 50 筆）。

### 網格側：**shipped 網格仍然表示不了大部分的先驗**

但在 **shipped 的 200 μas / 128 px 網格**上，可表示下界是
`8.245 px × 3.125 μas/px = 25.77 μas`，而兩個目標的環都在這個下界附近或以下：

| 目標 | 質量先驗可表示比例（a\* = 0.998） | 同上（a\* = 0） |
|---|---|---|
| `M87*` | **13.29%** | 14.11% |
| `SgrA*` | **52.39%** | 55.14% |

**如實回報：這個數字沒有接近 100%，而且原因不在先驗。**
`M87*` 在測到的質量 6.5e9 下環半徑 19.66 μas，本來就低於 25.77 μas 的下界；
要讓整段質量先驗都可表示，需要的是**網格設定**：

| 槓桿 | `M87*` | `SgrA*` |
|---|---|---|
| 固定 FoV = 200 μas，需要的 `n_pixels` | **≥ 364** | **≥ 152** |
| 固定 `n_pixels` = 128，可用的 FoV 半寬區間 | **[30.24, 70.42] μas** | **[31.06, 168.76] μas** |

其中 FoV 是比較便宜的槓桿：shipped 的 200 μas **半寬**（400 μas 全寬）
對一個 20–31 μas 的環大了約一個數量級。**兩個目標在 FoV = 50 μas / 128 px
下都是 100% 可表示**，而且先驗中最大的環（31.06 μas）仍然安穩落在視野內。

**這一輪沒有改 `configs/instruments/eht.yaml` 的 `imaging.fov_muas`，也沒有改
`cli.py` 的預設 context。** 理由：這是分析設定的決定，不是接線修正；而且
依既有規則，發現阻塞性問題要回報而不是順手改掉。推導已經進程式碼
（`GREternalWhiteHole.mass_representable_range_msun` /
`mass_prior_representable_fraction` / `required_n_pixels_for_prior` /
`required_fov_muas_for_prior`），上面每一個數字都由測試鎖住。
**列為新的待決策 I-3。**

> 另記：`cli.py::_default_context("image")` 用的是 `n_pixels: 64`，
> 與 `configs/instruments/eht.yaml` 的 `128` 不一致。在 64 px 下可表示下界是
> 51.53 μas，**兩個目標的可表示比例都是 0%**。這個不一致是既有的，不是這次引入的。

## X.12.3（I-1）`bh_accretion` 的物理實作

選了 X.11.4 表格裡的 **C**（擴充 `ImageShadowSimulator`），作法比照 GW 通道
`bh_ringdown` 的 `log10_A`：宣告一條唯象標度律，不做第一原理 GRMHD、不引入
新的物理套件或輻射轉移。

### 兩個參數各自驅動什麼

```
log10 I0        = LOG10_I0_EDD        + BRIGHTNESS_MDOT_INDEX * log10_mdot_edd
log10 (w / r)   = LOG10_W_FRAC_EDD    + W_FRAC_MDOT_INDEX     * log10_mdot_edd
asym_amp        = JET_CONTRAST_MAX    * jet_power_frac
```

| 常數 | 值 | 理由 |
|---|---|---|
| `LOG10_I0_EDD` | 2.0 | Eddington 率下峰值面亮度 10² Jy/μas²；在先驗 `[-5, 0]` 上跨 10⁻³–10² Jy/μas²，乘上 ~250 μas² 的環面積後涵蓋 M87\* 230 GHz 實際的 ~0.5–1 Jy 緊緻流量 |
| `BRIGHTNESS_MDOT_INDEX` | 1.0 | 亮度隨吸積率上升，取 log-log 線性、指數 1 |
| `LOG10_W_FRAC_EDD` | log10(0.05) | 接近 Eddington 的薄盤 H/R ~ 0.05 |
| `W_FRAC_MDOT_INDEX` | −0.18 | 厚度**隨吸積率下降**：M87\* 與 Sgr A\* 是輻射低效吸積流，低吸積率下幾何厚（H/R ~ 0.4），−0.18 把先驗兩端接到 0.397 與 0.05 |
| `JET_CONTRAST_MAX` | 3.0 | `jet_power_frac` = 1 時足點亮度 4 倍 |
| `JET_FOOTPOINT_SIGMA_RAD` | 0.6 | 足點方位角半寬 ~34° |

**`log10_mdot_edd` 同時決定亮度與厚度，這正是與 `gr_eternal` 的實質差異**：
`gr_eternal` 把這兩者當成兩個獨立自由參數，所以它可以造出「亮而薄」或「暗而厚」
的環，`bh_accretion` 不行。厚度再 clip 到 `[0.01, 0.5]`，與 `gr_eternal`
`ring_width_frac` 先驗同一個物理帶。

### 方位角不對稱

`_gaussian_ring_image()` 加三個參數 `asym_amp / asym_azimuth_rad / asym_sigma_rad`，
在既有的環上乘一個因子：

```
image *= 1 + asym_amp * exp(-0.5 (Δφ / asym_sigma_rad)^2)
```

`Δφ` 是在**環自己的座標系**（已經被 `pos_angle_rad` 旋轉過）裡量的方位角差，
所以噴流足點跟著投影自轉軸走。`asym_amp = 0` 時影像與原本**完全相同**
（`gr_eternal` 路徑一個位元都沒變）。副作用是 `position_angle` 在正對
（`i = 0`）時也不再是 no-op——`gr_eternal` 在 `i = 0` 時旋轉一個圓環，
實測影像變化 2.1e-16。

### simulator 與 likelihood 的公式一致性

兩條標度律只寫在 `ring_emission_from_params()` 一個地方。
`ImageShadowSimulator.simulate()` 呼叫它，`VisibilityLikelihood` 透過呼叫
simulator 取得模型影像，而 `VisibilityLikelihood.predictive_summary_stats()`
與 `BHAccretion.summary_stats()` 也都改成呼叫它（原本各自複寫了一份
`theta_d` / 亮度的算式）。測試 `TestSimulatorAndLikelihoodShareTheFormula`
逐項比對 predictive stats 與 simulator metadata。

模型辨識沿用 GW 模擬器的既有慣例（`if "log10_A" in params:`）：以哪一個亮度參數
在場來分支。同時帶兩個、或一個都沒帶，都 raise（不是優先序規則）。
`M / a_star / i / position_angle` 四個幾何參數改成**明確要求**，不再走
`params.get(key, default)`——即 R.4 的靜默預設模式。

### 逐參數活性（perturbation）

在 FoV = 50 μas / 128 px（可表示網格）、`thermal_noise_jy = 0.05` 下，
從 `M = 6.5e9, a* = 0.5, i = 0.9, PA = 0.7, log10_mdot = −2.5, f_jet = 0.3` 擾動：

| 參數 | 擾動後 | 影像最大相對變化 | `max|ΔV| / max|V|` | `ΔlnL` |
|---|---|---|---|---|
| `M` | ×1.05 | > 1e−3 | 0.1025 | −1.5e6 |
| `a_star` | 0.9 | > 1e−3 | 6.65e−3 | −6.53e3 |
| `i` | 1.3 | > 1e−3 | 0.5427 | −4.22e7 |
| `log10_mdot_edd` | −2.2 | > 1e−3 | 0.7619 | −8.31e7 |
| `jet_power_frac` | 0.9 | > 1e−3 | 0.4198 | −2.52e7 |
| **`position_angle`** | 1.9 | **~1.0** | **4.14e−4** | **+3.2** |

**五個參數同時改變 simulator 輸出與 lnL。`position_angle` 沒有**——
它把影像改了約 100%，可見度卻幾乎不動。原因不在 `bh_accretion`，見 X.12.5。

### 兩個假說確實不同

在**相同真值**（同環半徑、同厚度 0.1409、同峰值亮度 0.3162 Jy/μas²，
唯一差別是噴流足點）下：

| 量 | 數值 |
|---|---|
| 影像最大相對差 | **89.8%** |
| 影像對 180° 旋轉的不對稱度 | `bh_accretion` **0.473** / `gr_eternal` **0.0**（精確） |
| 同一筆 `bh_accretion` 資料上的 lnL 差 | **6.32e6** |
| 參數向量交集 | `{M, a_star, i, position_angle}`；各自獨有 `{ring_width_frac, log10_brightness}` 與 `{log10_mdot_edd, jet_power_frac}` |

## X.12.4 這一輪的產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `src/whitesearch/utils/targets.py` | `ImageTarget` / `EHT_TARGETS` / `resolve_target_name` / `require_target` / `target_distance_mpc` | 是 |
| `tests/test_image_target_and_accretion.py` | 58 項：目標解析與 fail-closed、逐目標可表示比例、吸積標度律、逐參數活性、兩假說差異 | 是 |

## X.12.5 阻斷性問題 (III)：**`_compute_visibilities()` 的 Gλ → rad⁻¹ 換算錯誤**

這是實作 I-1 的逐參數活性測試時發現的，**不在這次的 scope 內，沒有修**。

### 症狀

`position_angle` 把影像旋轉掉約 100%，模型可見度只變 4.14e−4。

### 根因

`simulators/image_shadow.py::_compute_visibilities()`：

```python
wavelength_m = C / (freq_ghz * 1e9)
uv_rad = uv_coverage * 1e9 * wavelength_m  # Gλ → rad^{-1}
```

以 Gλ 為單位的基線**本來就是**「每弧度幾個週期」，也就是已經是 rad⁻¹；
乘上波長（公尺）得到的是**基線的物理長度（公尺）**，不是空間頻率。
在 230 GHz 下這個乘數是 1.3037e−3，於是每一條 EHT 基線都被縮小了約 767 倍。

實測（FoV = 50 μas / 128 px，影像 FFT 的 uv 格點間距 2.06e9 rad⁻¹）：

| | 值 |
|---|---|
| 現行換算後的 uv 範圍 | −3.91e6 – 8.47e6 rad⁻¹ |
| 正確值（`uv * 1e9`） | −3.00e9 – 6.50e9 rad⁻¹ |
| FFT 格點間距 | 2.06e9 rad⁻¹ |

也就是 16 條基線**全部落在 uv 原點所在的那一個格子裡**（距原點 < 0.4% 格），
雙線性內插回傳的就是零間距流量。

### 後果（已量測）

| 量 | 現行 | 正確取樣（直接 DFT） |
|---|---|---|
| `\|V\|` 在 16 條基線上 | 全部 = 總流量，165.68 vs 165.70 Jy（相對散布 1.3e−3） | 39.7 – 163.7 Jy |
| `position_angle` 0.7 → 1.9 的 `max\|ΔV\|/max\|V\|` | 4.14e−4 | **0.571**（`max\|Δ\|V\|\|` 0.355） |
| 同一擾動的逐基線 n-σ（σ = 0.05 Jy） | < 0.1 | 33 – 2385 |

**image 通道的可見度目前只帶影像的總流量，不帶任何結構資訊。**
其他五個參數之所以看起來還「活著」，是因為它們都會改變總流量
（質量與傾角改環面積、吸積率改亮度與厚度、噴流改足點加亮）——
不是因為通道量到了環的形狀。

### 影響範圍

同樣的錯誤換算也出現在 `dataio/eht.py::_mock_eht_data()`
（`uv_rad = uv * 1e9 * wavelength_m`，用來產生 mock 的 sinc 可見度），
所以「修一行」不準確：至少兩處，且修正會改變 image 通道**每一個**既有數字。

### 分類

**阻斷性，依規則回報後停下，未修。** 列為新的待決策 I-4。
在此之前，image 通道的 SBC 不會是有意義的校準。
測試 `test_position_angle_moves_the_image_but_not_the_visibilities` 把現況、
正確取樣下的對照值、以及「若這個測試開始失敗代表換算已被修正」都鎖在 docstring 裡。

---

# X.13 image 通道 uv 基線尺度修正與網格／先驗一致性補強

> 這一節記錄 I-4（阻斷性：uv 基線尺度換算）的修正，以及順手處理的
> I-3（網格與先驗不匹配）、`n_pixels` 設定不一致、光子環半徑公式不一致。
> **不含任何科學結論。** 沒有跑任何 SBC/campaign/取樣。

## X.13.1（I-4）`_compute_visibilities()` 的兩個錯誤

### 錯誤一：基線單位

```python
wavelength_m = C / (freq_ghz * 1e9)
uv_rad = uv_coverage * 1e9 * wavelength_m   # Gλ → rad^{-1}
```

以 Gλ 為單位的基線**本身就是角頻率**——每弧度幾個條紋週期。乘上波長得到的是
**基線的物理長度（公尺）**。230 GHz 下這個乘數是 1.3037e−3，每條 EHT 基線
被縮小 767 倍。

| | 值 |
|---|---|
| 影像 FFT 的 uv 格點間距（FoV 50 μas） | 2.06e9 rad⁻¹ |
| 錯誤換算後的基線範圍 | −3.91e6 – 8.47e6 rad⁻¹ |
| 正確值（`uv * 1e9`） | −3.00e9 – 6.50e9 rad⁻¹ |

16 條基線全部落在 uv 原點所在的那一個格子裡，雙線性內插回傳零間距流量。

### 錯誤二：取樣方式（只修錯誤一還是錯的）

原本的做法是把影像 FFT 到規則 uv 網格再內插。該網格的間距是 `1 / (2·FoV)`，
所以**影像的視野會悄悄決定 uv 平面的精度**。這與 I-3 直接衝突：I-3 要把 FoV
縮到 50 μas，而那正是 FFT 內插最差的地方。

以「高斯環的解析 Hankel 變換」`V(u) = F·J₀(2π r₀ u)·exp(−2π²w²u²)` 為外部基準
（r₀ = 21 μas、w = 2.1 μas、F = 0.6 Jy，即 M87\* 的 42 μas 環）實測最大偏差：

| 取樣方式 | FoV / n_pixels | 對解析解的最大偏差（佔峰值） |
|---|---|---|
| FFT + 內插 | 50 μas / 128 | **8.96%** |
| FFT + 內插 | 200 μas / 128 | 0.66% |
| **直接 DFT** | 50 μas / 128 | **0.76%** |
| 直接 DFT | 200 μas / 512 | 0.76% |

因此改成**在要求的 (u,v) 點上直接做 DFT**。變換是可分離的，128×128 影像、
16 條基線約 1.2 ms，成本可接受；殘留的 0.76% 是影像本身的像素化，不是取樣器。
修正後 FoV 與 uv 精度**完全解耦**（FoV 50 與 200 給出相同的可見度）。

### 物理合理性檢查（外部基準，非內部前後比較）

| 檢查 | 結果 |
|---|---|
| 點源偏移 x₀ 的條紋相位 = −2π u x₀（u 以 λ 計） | 相位吻合到 1e−9 rad；若重新乘回波長會差 767 倍 |
| V(0,0) = 總流量 | 吻合到 1e−6 相對誤差 |
| 42 μas 環的第一極小位置，理論 2.405/(2π r₀) = **3.76 Gλ** | 模型給 **3.74 Gλ**；EHT 2019 ApJL 875 L4 觀測到的極小在 ~3.4 Gλ |
| 0.6 Jy 零間距流量 → 最長基線的相關流量 | 8 Gλ 上 0.068 Jy（幾十 mJy 量級，與 EHT 實測相符） |

> **關於前一輪稽核引用的「39.7–163.7 Jy」**：那個區間是從一張總流量 165 Jy 的
> 測試影像算出來的，**絕對尺度是測試設定的假象**，不是 M87\* 的真實量級。
> 這一輪改用無因次的 Bessel 結構與物理歸一化後的 mJy 級相關流量做基準。

### 修正前後：`position_angle`（驗證診斷是否正確）

同一組真值、同一個擾動（`position_angle` 0.7 → 1.9 rad），M87\*、
`log10_mdot_edd = −2.5`、`jet_power_frac = 0.3`：

| 量 | 修正前 | 修正後 | 倍數 |
|---|---|---|---|
| `max\|ΔV\| / max\|V\|` | 9.59e−4 | **0.570** | ×595 |
| `max\|Δ\|V\|\| / max\|V\|` | 4.11e−4 | 0.355 | ×863 |
| `max\|Δphase\|` | 8.67e−4 rad | **2.80 rad** | ×3225 |
| `\|V\|` 的散布（max/min） | **1.001** | **3.373** | — |

I-4 的診斷曾預測修正後應為 0.571；實測 0.570。**診斷成立。**

修正後的逐參數活性（`bh_accretion`，六個宣告參數全部）：

| 參數 | 影像最大相對變化 | `max\|ΔV\|/max\|V\|` | ΔlnL |
|---|---|---|---|
| `M` | 0.237 | 0.101 | −72.5 |
| `a_star` | 0.155 | 0.061 | −26.9 |
| `i` | 0.921 | 0.535 | −2150 |
| `position_angle` | 1.000 | **0.392** | **−408** |
| `log10_mdot_edd` | 0.995 | 0.760 | −4579 |
| `jet_power_frac` | 0.947 | 0.421 | −1750 |

**六個參數全部同時改變 simulator 輸出與 lnL。** 上一輪 I-1 未達成的第 4 項要求
至此完成。

## X.13.2（I-4 第二處）`dataio/eht.py::_mock_eht_data()`

同一個錯誤換算也在 mock 資料產生器裡。**兩邊用同一個錯誤換算，互相比較時
完全自洽**——這正是任何 forward-model 一致性檢查抓不到它的原因。

一併修正的還有形狀：原本用兩個 sinc 的乘積（那是矩形的變換，不是環的），
改成環真正的變換 `V(u) = F·J₀(2π r u)`。這個形狀在修正前**完全不影響結果**，
因為所有基線都在 uv 原點、任何變換在那裡都等於零間距流量；修正後才第一次
起作用。環直徑改用實測值：M87\* 42 μas（EHT 2019 ApJL 875 L1）、
Sgr A\* 51.8 μas（EHT 2022 ApJL 930 L12）。

修正後 mock 的 `|V|`：M87\* 在 0.54 Gλ 上 ~0.98 Jy，在 3.24 Gλ（第一極小附近）
掉到 0.15–0.19 Jy。

## X.13.3（順手三）光子環半徑：兩個公式都是錯的

**不假設哪一個對，實際推導。** 用 Bardeen (1973) 的球面光子軌道參數化算出
精確的臨界曲線，再取面積等效半徑 `√(A/π)`：

```
xi(r)  = [ (r² − a²) − r(r² − 2r + a²) ] / [ a(r − 1) ]
eta(r) = r³ [ 4a² − r(r − 3)² ] / [ a²(r − 1)² ]
alpha  = −xi / sin i ,   beta² = eta + a² cos²i − xi² cot²i
```

**外部交叉驗證**：a\* = 0 給 5.1962 rg = 3√3 ✓；a\* = 0.998 正對給 **4.830 rg**，
與文獻上 a\* = 1 正對陰影半徑 ~4.83 rg 相符。

| a\* | 精確（i=0） | 精確（i=90°） | 模擬器舊式 | 模型舊式 |
|---|---|---|---|---|
| 0.0 | 5.1962 | 5.1962 | 5.1962 | 5.1962 |
| 0.5 | 5.1205 | 5.1574 | 5.1658 (+0.9%) | 5.6458 (+10.3%) |
| 0.9 | 4.9161 | 5.0332 | 5.1485 (+4.7%) | 4.5875 (−6.7%) |
| 0.998 | 4.8304 | 4.9380 | 5.1453 (+6.5%) | 4.0159 (−16.9%) |

- **模擬器的** `3√3(1 − 0.0136a + 0.0038a²)`：方向對（陰影隨自旋縮小），
  但自旋依賴約弱 6 倍，最大誤差 **+6.5%**。
- **模型的** `rg(3 + √(9 − 8a²))`：**三重錯誤**——在 a\* = 0 不連續
  （6 rg vs 正確的 5.196 rg）、在 a\* ≲ 0.4 讓陰影**隨自旋變大**（真實是單調縮小）、
  誤差 −16.9% 到 +13.7%。
- 兩者在 a\* = 0.998 彼此相差 **22.0%**。

**結論：兩個都不對，模擬器那個比較接近。** 改為共用
`utils.math_utils.kerr_shadow_radius_rg(spin, inclination_rad)`——對精確臨界曲線
在 a\* ∈ [0, 0.998] × i ∈ [0°, 90°] 上的最小平方擬合，**最大誤差 0.363%**，
且 Schwarzschild 精確。真實陰影從 a\* = 0 到 0.998 單調縮小 7.0%（正對）。

`VisibilityLikelihood.predictive_summary_stats()` 原本沒有把 `i` 傳下去；
新增的測試抓到了這個新引入的不一致，已修。

## X.13.4（I-4 要求 4）closure phase 路徑是否受影響

**受影響，而且影響很徹底。**

- **修正前**：所有可見度都等於（實數、正的）零間距流量 → 所有相位 ≈ 0 →
  所有三元組組合 ≈ 0。closure phase 恆為常數 ±π（舊的 wrap 把 0 映到 −π），
  **完全不帶任何影像資訊**，von Mises 項是個常數。
- **修正後**：取值有實質分布（例如 3.11、3.12、−2.73、2.47、−0.02 rad），
  且隨影像旋轉改變。

順帶修掉一個 off-by-π：wrap 寫成 `closure % 2π − π`，會把零相位閉合映到 −π，
與 `EHTLoader.compute_closure_phases` 用的 `(cp + π) % 2π − π` 差一個 π。
因為 likelihood 兩邊都走同一個函式，這個偏移在 `κ cos(φ_obs − φ_model)` 裡
互相抵銷，但存進 metadata 的數值是錯的。已改為正確的 wrap。

### 另一項發現：三元組根本不閉合（回報，未修）

`_compute_closure_phases()` 把可見度陣列的連續三個元素當成一個三角形，但
`_default_eht_uv()` 是一串基線而不是台站陣列，**沒有任何一組滿足
u_ij + u_jk = u_ik**：

```
triplet 0: (0.5, 0.2) + (0.5, −0.2) = (1.0, 0.0)   第三個是 (0.2, 0.5)
```

所以這個量是一個相位組合，**不是 closure phase**，沒有 closure phase 存在的
理由——對台站增益的不變性。simulator 與 likelihood 用同一個函式計算它，
所以它是資料的一個自洽統計量、**不會造成推論偏差**，只是名不副實。
要修需要從 `configs/instruments/eht.yaml` 的台站座標推導 uv 覆蓋，讓真正的
三角形存在。**依規則回報，未在本輪修正。**

## X.13.5（順手發現）`bh_accretion` 的亮度歸一化錯了 100 倍

要求 2 的外部物理量級檢查同時抓到上一輪我自己引入的一個錯誤。

`LOG10_I0_EDD = 2.0` 當時的理由寫的是「在 [−5, 0] 先驗上跨 10⁻³–10² Jy/μas²，
乘上 ~250 μas² 的環面積後涵蓋 M87\* 實際的 ~0.5–1 Jy」。**那個算術是錯的**：
高斯環的積分是 `2π r₀ w √(2π)`，在這裡的厚度範圍是 660–1900 μas²，
而且我當時錨定的是峰值面亮度而不是檢查積分值。實際結果：

| `log10_mdot_edd` | 總流量（`LOG10_I0_EDD = 2.0`） |
|---|---|
| −5（先驗最暗端） | 1.87 Jy |
| −3（先驗中央） | 81.6 Jy |
| 0（先驗最亮端） | 2.35e4 Jy |

M87\* 實測 230 GHz 緊緻流量 0.5–1.2 Jy（EHT 2019 ApJL 875 L1/L4）、
Sgr A\* 2.0–2.5 Jy（EHT 2022 ApJL 930 L12）——**真值整個落在先驗之外**
（比先驗最暗端還暗）。

改為 `LOG10_I0_EDD = 0.0`（Eddington 率下峰值 1 Jy/μas²）後先驗跨
**0.019–235 Jy**，兩個目標都落在 `log10_mdot_edd ≈ −3` 附近，先驗內部。
`gr_eternal` 的 `log10_brightness ∈ [−4, 2]` 對應 0.037–3.7e4 Jy，真值在
−2.9 附近，本來就在先驗內，未動。

**這是修正我自己上一輪引入的算術錯誤，不是為了讓結果好看而調先驗。**
明確標記於此。

## X.13.6（I-3 與 `n_pixels` 一致性）成像網格

`configs/instruments/eht.yaml` 的 `imaging.fov_muas` 由 200.0 改為 **50.0**
（`n_pixels` 維持 128）。理由：M87\* 與 Sgr A\* 的環半徑約 19 與 25 μas，
200 μas **半寬**（400 μas 全寬）約大了一個數量級，其 3.125 μas 像素讓可表示
下界落在 25.77 μas——高於 M87\* 的環。

| 網格 | 像素 | 可表示下界 | M87\* 質量先驗可表示比例 | Sgr A\* |
|---|---|---|---|---|
| 200 μas / 64 px（cli.py 舊值） | 6.25 μas | 51.53 μas | **0%** | **0%** |
| 200 μas / 128 px（yaml 舊值） | 3.125 μas | 25.77 μas | 7.92% | 34.23% |
| **50 μas / 128 px（本輪）** | **0.781 μas** | **6.44 μas** | **100%** | **100%** |

（200 μas 那兩列的數字與上一輪的 13.29% / 52.39% 不同，是因為光子環公式
同時被修正了——X.13.3——高自旋下的環比舊式小約 7%。）

先驗**完全沒有動**，仍然由文獻上的獨立質量測量決定。改的是分析設定。
最小可用 `n_pixels`（FoV 50 μas）：M87\* 97、Sgr A\* 41，shipped 的 128 兩者皆過。

`cli.py` 原本另外寫死 `n_pixels: 64`，與 yaml 的 128 不一致。新增
`dataio.eht.eht_imaging_config()` 作為唯一讀取點，`cli.py::_default_context`
改為呼叫它；找不到設定檔時 fail-closed raise，不回退到程式內的數字
（靜默回退正是兩份數字漂移開來的原因）。測試同時鎖住 yaml、
`gr_eternal.IMAGE_*` 常數、以及 `_default_context("image")` 三者一致。

## X.13.7 產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `tests/test_image_uv_sampling.py` | 39 項：基線單位、對解析 Hankel 變換的比對、影像結構是否進入可見度、mock 路徑、closure phase、Kerr 陰影半徑（測試內獨立重算精確臨界曲線） | 是 |
| `src/whitesearch/utils/math_utils.py::kerr_shadow_radius_rg` | 共用的 Kerr 陰影半徑擬合 | 是 |
| `src/whitesearch/dataio/eht.py::eht_imaging_config` | 成像網格的唯一讀取點 | 是 |

## X.13.8 image 通道現在是否具備跑一次有意義校準的條件

**具備了，但有兩項必須寫在報告裡的限制。** 以下是量測值，不是判斷：

| 指標 | 之前 | 現在 |
|---|---|---|
| 先驗抽樣的環可被表示 | 18.29% → 13.29% | **100%**（兩個模型、兩個目標） |
| 可見度是否帶影像結構 | 否（每條基線都等於總流量） | **是**（對解析解 0.76%；`\|V\|` 散布 3.4 倍） |
| 宣告參數全部同時影響資料與 lnL | `gr_eternal` 是；`bh_accretion` 6 之中 5 | **兩個模型皆全部** |
| 真值是否落在先驗內（總流量） | `bh_accretion` 否（真值在先驗外） | **是**（兩個目標都在 `log10_mdot_edd ≈ −3`） |
| 先驗抽樣 SNR ≥ 8 的比例 | 未量測 | `gr_eternal` **81.2%**、`bh_accretion` **66.0%** |

（對照：GW 通道修正後 SNR ≥ 8 的比例是 0.785%，N=100 的 campaign 期望
只有約 0.8 筆帶可偵測訊號。image 通道現在的注入效率高兩個數量級。）

**兩項必須一併回報的限制：**

1. **closure phase 名不副實**（X.13.4）。它是自洽的統計量、不會造成推論偏差，
   但不具備台站增益不變性。若要在報告裡宣稱用了 closure phase，這一點必須說清楚；
   或者用 `use_closure_phases=False` 明確聲明 amplitude-only。
2. **從未跑過任何 SBC**。上表說的是「條件已具備」，**不是「已校準」**。
   在跑過 SBC 之前，這條通道沒有任何 coverage 證據。

---

# X.14 image 通道 gr_eternal 首次 SBC/coverage 校準驗證 —— **未能完成，回報阻塞原因**

> 這一節記錄第一次嘗試對 image 通道跑 SBC/coverage 的結果。
> **沒有產生任何 SBC 結論**，因為 campaign 在這個執行環境裡跑不完。
> 這裡記的是量測到的耗時與阻塞原因，以及一個必須誠實說明的取樣偏差風險。
> 不含任何科學結論。

## X.14.1 設定

| 項目 | 值 |
|---|---|
| 模型 / 通道 | `gr_eternal` / image |
| 目標 | M87\*（`SgrA*` 未跑） |
| 取樣維度 | 6（`M, a_star, i, position_angle, ring_width_frac, log10_brightness`） |
| `use_closure_phases` | **False**（amplitude-only，明確聲明；I-5 未修，本輪不宣稱使用 closure phase） |
| sampler | dynesty，`bound='live'`、`sample='rwalk'`、`nact=2`、`dlogz=0.1` |
| `nlive` | 250 |
| L（rank 分母） | 100 |
| 逾時機制 | `BilbyRunner.run_timeout_s` → `_BudgetGuard`（在 likelihood 內部檢查） |

**`nact` 生效已驗證**（比照 Part K.2 的教訓，看行為不看鍵是否存在）：
bilby log 印出 `An average of 4 steps will be accepted up to chain length 5000`
（= 2 × nact），且完成的那一筆 result JSON 裡 `sampler_kwargs` 確實含
`{'bound': 'live', 'sample': 'rwalk', 'nact': 2, 'dlogz': 0.1}`。

**順帶修好 checkpoint/resume**：bilby 的 `check_point_delta_t` 預設 600 s，比
一個 shard 還長，所以**從來沒有寫出過 resume 檔**，每個 shard 都從頭跑。
改為 45 s 之後 resume 確實生效（實測：shard 結束在 1669 it，下一個 shard
印出 `Reading resume file ...` 並從該處續跑到 2155 it）。

## X.14.2 量測到的耗時

`docs/calibration/image_gr_eternal_sbc_cost_probe.csv`。

| idx | 網路 SNR | 總流量 (Jy) | `log10_brightness` | 結果 | 牆鐘 |
|---|---|---|---|---|---|
| 4 | 0.21 | 0.0033 | −3.53 | **收斂** | **34 s** |
| 0 | 158 | 2.27 | −0.66 | **收斂** | **3292 s（≈55 分）** |
| 3 | 1504 | 21.0 | +0.08 | 逾時 | 120 s |
| 1 | 1560 | 24.5 | −1.50 | 逾時（dlogz 仍 2150） | 240 s |
| 2 | 15755 | 283 | +0.69 | 逾時 | 120 s |
| 5 | 318443 | 5542 | +1.21 | 逾時 | 120 s |

唯一一筆走完全程的中等 SNR 注入（idx 0，SNR 158）：
**4574 次 NS 迭代、約 1.3e6 次 likelihood 呼叫、3292 s**，
分 8 個 shard 以 resume 累積完成。收斂過程明顯減速：

| 累積牆鐘 | 迭代 | `nc`（每迭代呼叫數） | 效率 | 剩餘 `dlogz` |
|---|---|---|---|---|
| 100 s | 1669 | 54 | 3.9% | 308 |
| 200 s | 2155 | 238 | 2.5% | 113 |
| 740 s | 3221 | 314 | 1.0% | 11.9 |
| 1280 s | 3658 | 511 | 0.7% | 4.53 |
| 1825 s | 3977 | 631 | 0.5% | 2.10 |
| 2370 s | 4232 | 744 | 0.4% | 0.885 |
| 2915 s | 4574 | 1081 | 0.4% | 0.236 |
| 3292 s | — | — | — | **收斂** |

likelihood 本身是 **1.755 ms/呼叫**（其中 1.126 ms 在
`_compute_visibilities`，主要是 `c_einsum`）。即使把它完全消掉，
單筆仍要約 1900 s——**瓶頸是呼叫次數，不是單次成本**。

## X.14.3 阻塞原因：先驗預測的 SNR 跨度讓 NS 成本爆炸

巢狀取樣的迭代數約為 `nlive × H`，H 是後驗相對先驗的資訊量（nats），
而 H 隨 SNR 對數成長、隨維度線性成長。idx 0 的 H ≈ 4574 / 250 ≈ **18.3 nats**。

`log10_brightness ~ Uniform(−4, 2)` 是一個 **6 dex** 的自由振幅，而熱噪聲固定
在 0.05 Jy，於是先驗預測的 SNR 橫跨 **0.2 到 3.2e5**（上一輪量到的百分位
[5, 50, 95] = [0.4, 581, 5.9e5]）。在 SNR ~ 3e5 的那一端，
H ≈ 6 × ln(3e5) ≈ 76 nats → 迭代數 ≈ 19000，且 `nc` 早已超過 1000，
單筆估計 10 小時以上。

以 `呼叫數 ∝ (ln SNR)²` 外推（迭代數與 `nc` 各約正比於 ln SNR，
與 idx 0 的實測對齊）：

| 先驗 SNR 百分位 | 估計單筆牆鐘 |
|---|---|
| 5%（SNR 0.4） | 秒級 |
| 50%（SNR 581） | ≈ 5200 s（87 分） |
| 95%（SNR 5.9e5） | ≈ 22000 s（6 小時） |

**N=100 的期望總成本約 145 小時，且尾部很重。** 這個執行環境的容器在
turn 之間會暫停，實際只能以 ≤ 600 s 的前景片段推進，所以連 N=20
（估計 29 小時）都不實際。

## X.14.4 為什麼不能「設一個時間上限、用跑完的筆數回報」

前幾輪（Part K / Part L）遇到耗時問題時的做法是設定每筆時間上限、
用已完成的筆數回報。**這一次那個做法會產生無效的結果，必須說清楚。**

上表顯示收斂與否幾乎完全由 SNR 決定：跑完的兩筆是 SNR 0.21（後驗≈先驗，
34 s）與 SNR 158（3292 s），其餘 SNR ≳ 1500 的全部逾時。因此任何時間上限
都會**系統性地只留下最安靜的注入**——而那些注入的後驗幾乎就是先驗，
rank 當然接近均勻。**這會產生一份看起來校準良好、但完全是選擇效應造成的
SBC 報告。** 這正是「不要調整讓結果好看」要防的情況，所以本輪
**不回報任何 SBC rank / KS p 值 / coverage 數字**。

（`SgrA*` 組同理未跑。）

## X.14.5 修正上一輪 readiness 評估裡的一個疏漏

X.13.8 判定 image 通道「具備跑一次有意義校準的條件」，依據之一是
先驗抽樣 SNR ≥ 8 的比例 81.2%，並與 GW 通道的 0.785% 對比。
**那個評估只看了可偵測性，沒有看取樣成本。** 極高的 SNR 對偵測是好事，
對巢狀取樣卻是成本來源——資訊量 H 越大、需要的先驗體積壓縮越多。
GW 通道的 campaign 之所以跑得動，部分正是因為它的注入大多很安靜。

**應該補上的判準**：一條通道「可以跑校準」不只要求真值落在先驗內、
參數全部生效、資料可被表示，還要求**先驗預測的資訊量分布落在取樣器
負擔得起的範圍內**。image 通道目前不滿足最後這一項。

## X.14.6 已產出、可續跑的東西

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `scripts/run_image_sbc.py` | 可續跑的 shard runner：每筆一個 JSON、已存在即跳過；resume 已修好（`check_point_delta_t=45`）；per-run 逾時走 `_BudgetGuard` | 是 |
| `docs/calibration/image_gr_eternal_sbc_cost_probe.csv` | 上表的原始數字 | 是 |
| `artifacts/image_sbc/`、`artifacts/image_probe/` | 逐筆 JSON 與 bilby resume 檔 | 否（artifacts） |

idx 0 的完整後驗已存在（880 samples），可作為日後續跑的第一筆。
**單筆的 rank 與 coverage 不構成校準證據**（N=1 下 rank 本來就均勻分布），
所以這裡不列出。

---

# X.15 `gr_eternal` 亮度先驗合理性評估與取樣成本 pilot（I-6）

> 這一節評估 `log10_brightness` 先驗是否該收窄，並用小規模 pilot 量測收窄後
> 的取樣成本。**先驗本身沒有被修改**——結論是「該收窄，但收窄不足以解決
> 取樣成本」，依規則先回報不自行定案。不含任何科學結論。

## X.15.1 先驗與物理量級核對

`gr_eternal` 的 `log10_brightness` 是**峰值面亮度**（Jy/μas²），先驗
`Uniform(-4, +2)`，6 dex。總流量是導出量：

```
F_total = I0 × G ,  G = G(M, a*, i, ring_width_frac)   [μas²]
```

G 是幾何因子（高斯環的面積分 `2π r0 w √(2π)` 乘上傾角壓縮）。
在現行先驗下（600 筆抽樣，固定 `log10_brightness = -3`）：

| G 百分位 | 1% | 5% | 25% | 50% | 75% | 95% | 99% |
|---|---|---|---|---|---|---|---|
| μas² | 0.65 | 8.13 | 40.5 | **120.5** | 407.5 | 1492 | 2879 |

**G 本身跨 2.264 dex（5–95%）、1.003 dex（25–75%）。**
（全域跨度 18.66 dex 是單一病態抽樣：`i → π/2` 時 `axial_ratio = |cos i|`
被 clip 到 0.01，環退化成沿 x 軸、兩個方向都次像素的針狀物。600 筆中 1 筆。
受控掃描確認一般情況下總流量與解析解 `2π r0 w √(2π) I0` 相差 < 2%，
即使環厚只有 0.25 像素，所以**這不是普遍的取樣問題**。）

### 觀測量級（沿用前一輪已核對過的數字）

| 目標 | 230 GHz 緊緻總流量 | 出處 |
|---|---|---|
| M87\* | **0.5–1.2 Jy** | EHT 2019, ApJL 875 L1 / L4 |
| Sgr A\* | **2.0–2.5 Jy** | EHT 2022, ApJL 930 L12 |

以中位幾何 G = 120.5 μas² 反推：

| 要涵蓋的流量 | 對應 `log10_brightness` | 跨度 |
|---|---|---|
| M87\* 0.5–1.2 Jy | [−2.382, −2.002] | 0.380 dex |
| Sgr A\* 2.0–2.5 Jy | [−1.780, −1.683] | 0.097 dex |
| 兩個目標 0.5–2.5 Jy | [−2.382, −1.683] | 0.699 dex |
| 上列 ± 1 dex 餘裕 | **[−3.38, −0.68]** | **2.70 dex** |

### 現行先驗確實與真實物理脫節

| 先驗 | dex | 總流量 5/50/95% (Jy) | 網路 SNR 5/50/95% | SNR > 10³ 的比例 |
|---|---|---|---|---|
| **現行 (−4, +2)** | 6.00 | 0.011 / **11.32** / 9947 | 0.78 / 748 / 5.99e5 | **47.8%** |
| ±1.5 dex (−3.88, −0.18) | 3.70 | 0.007 / 1.173 / 120 | 0.48 / 73 / 6561 | 18.7% |
| **±1.0 dex (−3.38, −0.68)** | 2.70 | 0.019 / **1.278** / 48.3 | 1.13 / 79.5 / 2834 | 12.8% |
| ±0.65 dex (−3.03, −1.03) | 2.00 | 0.030 / 1.243 / 33.0 | 2.01 / 79.0 / 1836 | 8.2% |
| ±0.3 dex (−2.68, −1.38) | 1.30 | 0.050 / 1.247 / 19.3 | 3.31 / 77.0 / 1145 | 5.8% |

原始數字：`docs/calibration/image_gr_eternal_brightness_prior_options.csv`。

**現行先驗的中位總流量是 11.3 Jy，比 M87\* 實測的 0.5–1.2 Jy 高約一個數量級，
而且 47.8% 的先驗質量落在網路 SNR > 1000**——那不是 EHT 資料所在的區域。
**所以收窄在物理上本身就有理由，與取樣成本無關。**

## X.15.2 但收窄之後跨度仍然偏寬——回報 tension

即使 `log10_brightness` 收成一個 delta function，**幾何仍然貢獻 2.264 dex
(5–95%) 的流量散布**。所以：

| | 貢獻 |
|---|---|
| 亮度先驗（±1 dex 方案） | 2.70 dex |
| 幾何 G（5–95%） | 2.264 dex |
| 結果：總流量 5–95% | 0.019 – 48.3 Jy |
| 結果：SNR 5–95% | 1.13 – 2834 |

**這超過「2–3 dex」的門檻，因此依指示先回報、不自行定案收窄到哪個數字。**

## X.15.3 pilot：實測收窄後的取樣成本

用 ±1 dex 方案（`[-3.38, -0.68]`）以**執行期覆寫**（`scripts/run_image_sbc.py`
的 `--brightness-prior`，不動 shipped 模型）跑 N=4，`nlive=250`、
`use_closure_phases=False`，逐步 resume 累積：

| 先驗 | idx | SNR | 結果 | 累積牆鐘 |
|---|---|---|---|---|
| 6.00 dex | 4 | 0.21 | 收斂 | 34 s |
| 6.00 dex | 0 | 158 | 收斂 | 3292 s |
| 6.00 dex | 3 / 2 / 5 | 1504 / 15755 / 318443 | 逾時 | 120 s |
| **2.70 dex** | 0 | **9.7** | **收斂** | **380 s** |
| **2.70 dex** | 3 | **35.6** | **收斂** | **1006 s** |
| **2.70 dex** | 2 | 173.4 | 逾時 | > 730 s |
| **2.70 dex** | 1 | 273.9 | 逾時 | > 730 s |

原始數字：`docs/calibration/image_gr_eternal_prior_narrowing_pilot.csv`。

四個收斂點 (SNR, 秒) = (0.21, 34)、(9.7, 380)、(35.6, 1006)、(158, 3292)
在 ln(SNR) 上是凸的；上段斜率約 **1044 s / nat**。
內插到收窄後的中位 SNR ≈ 80（ln = 4.38）得 **約 2250 s ≈ 37 分/筆**。

**沒有達到「單筆十分鐘內」的門檻。** N=100 的期望成本約 62 小時，
且 12.8% 的抽樣仍在 SNR > 1000（每筆數小時）。
**因此不開完整規模 campaign。**

## X.15.4 還有哪些因素（不只是先驗寬度）

依指示列出觀察到的其他因素，**未自行選擇任何一項動手改**：

1. **資訊量由幾何主導，不是亮度。** 把 idx 0 的 H ≈ 18.3 nats 用 68% 可信區間
   對先驗寬度粗分解：`position_angle` ≈ 4.5、`M` ≈ 2.6、
   `log10_brightness` ≈ 2.2、`i` ≈ 1.8、`ring_width_frac` ≈ 0.8、
   `a_star` ≈ 0.45。**亮度先驗從 6 dex 收到 2.7 dex 只移除約 0.8 nat**，
   佔總量的 4%。收窄真正的作用是**砍掉高 SNR 的長尾**，不是降低中位成本。

2. **參數化本身**（與 GW 的 B3-2 同類）。`log10_brightness` 是峰值面亮度，
   但物理上知道的量（M87\* ≈ 1 Jy）與資料約束最緊的量（短基線／零間距
   振幅）都是**總流量**。因為 `F = I0 × G`，幾何會漏進流量 2.264 dex，
   而且**「這個源大約 1 Jy」這件事無法寫成 I0 的先驗**。
   改成以 `log10_total_flux_jy` 參數化、把 I0 設為導出量，會讓先驗直接陳述
   實測流量、切斷幾何洩漏。**這是設計決定，未實作。**

3. **取樣器在後驗收緊時退化。** 每次 NS 迭代的 likelihood 呼叫數 `nc`
   從約 50 成長到超過 1000，效率 3.9% → 0.4%（X.14.2）。這是
   `bound='live'` + `sample='rwalk'` 的策略問題，與先驗無關。

4. **likelihood 單次成本 1.755 ms**，其中 1.126 ms 在 `_compute_visibilities`
   的 `c_einsum`（numpy 的 einsum 對這個收縮不走 BLAS）。改成矩陣乘法形式
   估計可省 2–3 倍，結果不變。純效能，未實作。

5. **`axial_ratio = |cos i|` 在接近側視時讓環退化。** 這是 X.15.1 那條
   18.66 dex 尾巴的來源，也讓流量在 `i → π/2` 附近塌掉。這是建模選擇
   （用傾角壓縮一個圓環來代表傾斜的發射環），值得重新檢視。

## X.15.5 對 `bh_accretion` 的影響：無

兩者的亮度相關先驗是**各自獨立定義**的：

| 模型 | 參數 | 定義位置 | 先驗 |
|---|---|---|---|
| `gr_eternal` | `log10_brightness` | `models/gr_eternal.py` 的 ParameterSpec | `Uniform(-4, 2)` |
| `bh_accretion` | `log10_mdot_edd` | `models/alternatives.py` 的 ParameterSpec | `Uniform(-5, 0)` |

兩者只共用 `ring_emission_from_params()` 這個**求值函式**，不共用先驗定義；
交集參數只有 `M, a_star, i, position_angle`。已實測驗證：把 `gr_eternal` 的
`log10_brightness` 重新設界之後，`bh_accretion` 的先驗完全不變。
上一輪修正的 `LOG10_I0_EDD = 0.0`（X.13.5）不受本輪影響。

**本輪沒有修改任何 production 先驗**，所以兩邊都維持原狀。

## X.15.6 結論

1. **`log10_brightness` 先驗確實該收窄**——理由是物理脫節（中位流量比實測
   高一個數量級、47.8% 的先驗質量在 SNR > 1000），不是為了省成本。
   涵蓋兩個目標並留 ±1 dex 餘裕的方案是 **`[-3.38, -0.68]`（2.70 dex）**。
2. **但收窄不足以解決取樣成本**：幾何本身就貢獻 2.264 dex，實測收窄後
   中位仍約 **37 分/筆**，SNR ≳ 170 的注入在 730 s 內仍未收斂。
3. 依指示**先回報、未自行定案**。要讓這條通道的 SBC 可負擔，
   單靠收窄先驗不夠，X.15.4 的第 2 項（改以總流量參數化）看起來是
   對症的方向，但那是設計決定。

---

# X.16 `gr_eternal` 振幅參數化改為總流量（I-6）

> 把 `gr_eternal` 的取樣振幅從 `log10_brightness`（峰值面亮度）改成
> `log10_total_flux_jy`（總流量），並量測取樣成本。不含任何科學結論。

## X.16.1 為什麼換參數

X.15 的結論是：先驗與物理脫節（中位總流量 11.3 Jy vs M87\* 實測 0.5–1.2 Jy、
47.8% 的先驗質量在 SNR > 10³），但**單純收窄 `log10_brightness` 解決不了成本**。
根因是參數化：

```
F = I0 × G(M, a*, i, ring_width_frac)        G 跨 2.264 dex（5–95%）
```

物理上已知的量（M87\* ≈ 1 Jy）與資料約束最緊的量（短基線／零間距振幅）
**都是 F**，但先驗放在 `I0` 上。於是「這個源大約 1 Jy」這件事**無法寫成
`I0` 的先驗**——幾何會漏進流量。

改成取樣 `log10_total_flux_jy`、反算 `I0 = F / G`，先驗就直接建立在被量測、
且被資料約束的量上。與 GW 的爆發時序重新參數化（`BOUNCE_PREFLIGHT_AUDIT.md`
B3-2）是同一個動作。

## X.16.2 實作

`simulators/image_shadow.py` 新增 `build_ring_image(params, context)`，
是**影像唯一的建構點**：simulator、`VisibilityLikelihood.predictive_summary_stats`
都走它，所以三邊不可能漂移。

`RingEmission` 現在帶 `normalisation` 與 `amplitude` 兩個欄位：

| 模型 | `normalisation` | `amplitude` 的意義 |
|---|---|---|
| `gr_eternal` | `"total_flux"` | 積分流量 F [Jy]，I0 由 `F / G` 導出 |
| `bh_accretion` | `"peak_brightness"` | 峰值面亮度 I0 [Jy/μas²]，F 為導出量 |

`G` **以數值方式在影像網格上求值**（先以 I0 = 1 建圖、取積分），不是用解析
環積分，所以歸一化把像素化一併吸收掉——**成像後的總流量與取樣值精確相等**
（實測 1600 筆抽樣，最大相對偏差 4.4e-16）。

同步改動：`models/gr_eternal.py` 的 ParameterSpec 與 `summary_stats`、
`likelihoods/visibility.py` 的 `RING_PARAMETERS` 與 `predictive_summary_stats`、
`tests/conftest.py` 的 `gr_params`。取樣維度仍是 6。

## X.16.3 先驗：逐目標，從實測流量訂

比照 I-2 處理 `D_L` 與質量先驗的風格，流量先驗也放進 `ImageTarget`：

| 目標 | 實測緊緻 230 GHz 流量 | 出處 | 先驗 | 跨度 |
|---|---|---|---|---|
| M87\* | 0.5–1.2 Jy | EHT 2019, ApJL 875 L1 / L4 | **[0.05, 12] Jy** | 2.380 dex |
| Sgr A\* | 2.0–2.5 Jy | EHT 2022, ApJL 930 L12 | **[0.2, 25] Jy** | 2.097 dex |

**餘裕取實測範圍兩端各一個數量級。** 理由寫進 `utils/targets.py`：兩個目標
都會變（Sgr A\* 在小時尺度上變化可達數倍），而且總流量裡有多少屬於被建模的
環、多少屬於延展結構本身就不確定——一個數量級是給這個建模拆分的餘裕，
不是填充。**先驗是逐目標的**，理由與質量先驗相同：兩者流量差約 3 倍，
寫成一個聯集先驗會讓兩個目標都偏離中心。**沒有沿用舊的
`log10_brightness ~ U(-4, 2)` 範圍。**

## X.16.4 `bh_accretion` 是否有同樣的問題（要求 4）

**結構上有，但不該用同樣的方式修，所以本輪沒有動它。**

實測 `bh_accretion` 在 M87\* 下的先驗預測：總流量 5/50/95% =
0.0074 / 0.811 / 81.0 Jy，SNR 5/50/95% = 0.5 / 52.4 / 4967，19.3% 在 SNR > 10³。
中位 0.81 Jy 落在實測 0.5–1.2 Jy 裡（上一輪 X.13.5 修 `LOG10_I0_EDD` 的效果），
但尾巴仍寬，因為幾何同樣會漏進流量。

**但把它改成取樣總流量會刪掉這個假說本身。** `log10_mdot_edd` 同時決定
亮度**與**厚度，這個耦合正是它與 `gr_eternal` 的實質差異（X.12.3）；
若改成歸一化到總流量，亮度那一半的耦合就被歸一化掉，`bh_accretion` 會退化成
「厚度由一個參數決定的 `gr_eternal`」。對它而言對症的做法是重新錨定
`LOG10_I0_EDD`（已於 X.13.5 做過），而不是換參數化。

依指示**回報、未修改**。程式碼裡以 `normalisation="peak_brightness"` 明確
標記這個選擇，並在註解說明理由。

## X.16.5 順帶處理：不可表示的環不能中斷取樣

改用數值歸一化之後，`i → π/2` 且環很薄的角落（`axial_ratio = |cos i|`
把環壓成次像素針狀物）會讓 G 精確等於 0，沒有任何 I0 能把流量放上去。
最初的實作在這裡 `raise`，結果**前四筆 pilot 全部在取樣中途被中斷**。

一個模型表達不出來的參數點，正確行為是**被取樣器拒絕**，不是讓整個 run 崩掉。
因此：`build_ring_image` 丟出專屬的 `UnrepresentableRingError`，
`VisibilityLikelihood.loglike` 接住它並回傳 `-inf`，同時計入
`n_unrepresentable` / `last_unrepresentable`（計數，不是吞掉）。

**這不是新行為**：舊參數化在同樣的幾何下影像流量趨近於零，likelihood 本來
就會給出極差的值。這只是把同一件事寫清楚。注入端若抽到這種幾何，
`scripts/run_image_sbc.py` 記為 `status="unrepresentable"` 並跳過，
讓 campaign 的分母保持誠實。

實測 1600 筆先驗抽樣中，注入端**一次都沒有**觸發（G 的最小值 9.0e-3 μas²，
800 筆中有 5 筆 G < 1 μas²）；取樣器在探索時才會碰到。

## X.16.6 pilot：取樣成本（要求 6）

M87\*、`nlive=250`、`use_closure_phases=False`、逐步 resume 累積，N=6：

| 參數化 | SNR | 結果 | 累積牆鐘 |
|---|---|---|---|
| 峰值亮度，6.00 dex | 0.21 | 收斂 | 34 s |
| 峰值亮度，6.00 dex | 158.3 | 收斂 | **3292 s** |
| 峰值亮度，2.70 dex（收窄） | 9.7 | 收斂 | 380 s |
| 峰值亮度，2.70 dex（收窄） | 35.6 | 收斂 | **1006 s** |
| **總流量，2.38 dex** | 4.8 | 收斂 | **64 s** |
| **總流量，2.38 dex** | 31.3 | 收斂 | **256 s** |
| **總流量，2.38 dex** | 73.5 | 收斂 | **552 s** |
| **總流量，2.38 dex** | 149.2 | 收斂 | **627 s** |
| 總流量，2.38 dex | 201.2 | 逾時 | > 1438 s |
| 總流量，2.38 dex | 334.1 | 逾時 | > 365 s |

**同 SNR 的直接對照：**

| SNR | 峰值亮度參數化 | 總流量參數化 | 加速 |
|---|---|---|---|
| ~33 | 1006 s | 256 s | **3.9×** |
| ~155 | 3292 s | 627 s | **5.3×** |

先驗預測的 SNR 分布也一併改善（M87\*）：

| | 中位 SNR | SNR > 10³ 的比例 |
|---|---|---|
| 峰值亮度，6.00 dex | 748 | **47.8%** |
| 峰值亮度，2.70 dex | ~80 | 12.8% |
| **總流量，2.38 dex** | **48.8** | **0.0%** |

### 加速的來源不是資訊量下降

以 68% 可信區間對先驗寬度做 H 分解，拿 SNR 158（舊）與 SNR 149（新）
這兩筆幾乎同 SNR 的注入比較：

| 參數 | 舊 H (nats) | 新 H (nats) |
|---|---|---|
| `M` | 2.60 | 3.17 |
| `a_star` | 0.45 | 0.63 |
| `i` | 2.49 | 0.78 |
| `position_angle` | 4.48 | 3.14 |
| `ring_width_frac` | 0.77 | 1.05 |
| 振幅 | 2.17 | **5.96** |
| **合計** | **12.97** | **14.73** |

**總資訊量沒有下降，反而略升**，但牆鐘快了 5.3 倍。所以加速**不是**來自
「要壓縮的先驗體積變小」，而是來自**後驗的條件數變好**：舊參數化裡
`I0` 與幾何透過 `F = I0 × G` 強烈相關，後驗是一條細長的斜脊，`rwalk` 在上面
接受率很低；改成直接取樣資料本來就約束得很緊的 F 之後，那條脊消失了。
振幅本身的 H 從 2.17 升到 5.96 正是這件事的表現——舊的 `log10_brightness`
之所以 H 低，是因為它**沒有被約束好**（與幾何簡併），不是因為它不重要。

## X.16.7 完整規模 campaign 的耗時估計（要求 7，未開跑）

以四個收斂點擬合 `t = -236 + 171 × ln(SNR)` 秒，套用新先驗的 SNR 分布：

| | 值 |
|---|---|
| 單筆耗時 5/50/95% | 30 / **427** / 849 s |
| 平均 | 427 s（7.1 分） |
| **N=100** | **約 11.9 小時** |
| N=20 | 約 2.4 小時 |

對照：原始 145 小時 → 收窄亮度先驗 62 小時 → **改參數化 11.9 小時**。
**中位單筆 7.1 分，達到「十分鐘內量級」的目標。**

**這個估計的保留意見（如實列出）：**

1. 擬合只有 **4 個收斂點**，而且觀測到的散布很大：SNR 201.2 那筆已跑
   1438 s 仍未收斂，遠高於擬合預測的約 670 s。**11.9 小時應視為下界。**
2. N=6 的 pilot 裡有 **2 筆未收斂**（SNR 201.2、334.1）。尾部行為沒有量準。
3. 只跑了 M87\*。Sgr A\* 的中位 SNR 是 126.5、7.2% 在 SNR > 10³
   （它本來就比較亮），成本會更高。

## X.16.8 參數化無關的殘留因素（要求 6，回報未處理）

1. **`position_angle` 的資訊量本來就高**（新參數化下仍有 3.14 nats，
   是第二大項）。它由橢圓取向與（`bh_accretion` 的）噴流足點決定，
   在高 SNR 下被約束得很緊。這與振幅參數化無關。
2. **`i → π/2` 的退化長尾**：`axial_ratio = |cos i|` 把環壓成次像素針狀物
   （X.16.5）。現在不會中斷取樣，但那塊先驗區域等於被挖掉，是建模假影。
3. **取樣器在後驗收緊時退化**：`nc` 隨迭代成長（X.14.2）。
   `bound='live'` + `sample='rwalk'` 的策略問題。
4. **likelihood 1.755 ms/呼叫**，其中 1.126 ms 在 `_compute_visibilities` 的
   `c_einsum`（不走 BLAS）。改矩陣乘法估計可省 2–3 倍，結果不變。

**四項都未動手，等你決定。**

## X.16.9 產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `docs/calibration/image_gr_eternal_flux_reparameterisation_pilot.csv` | 上面成本對照表的原始數字 | 是 |
| `tests/test_image_uv_sampling.py::TestFluxParameterisation` | 流量往返精確性、幾何不再漏進流量、先驗由實測值構成、六個參數皆非死參數 | 是 |
| `tests/test_image_uv_sampling.py::TestUnrepresentableRingIsRejectedNotFatal` | 不可表示的環回傳 `-inf` 而非中斷 | 是 |

---

# X.17 `gr_eternal` 新參數化下 N=20 中間規模驗證與耗時模型校準

> N=20 是 N=100 的中間驗證，目的是校準耗時模型、確認收斂率。
> rank/coverage 是**初步方向性觀察**，N=19 不足以做正式判定。
> 不含任何科學結論。

## X.17.1 設定

與 X.16 的 pilot **完全相同**：`gr_eternal`、M87\*、6 維、
`use_closure_phases=False`（amplitude-only，I-5 未修）、
`bound='live'` / `sample='rwalk'` / `nact=2` / `dlogz=0.1`、`nlive=250`、
`log10_total_flux_jy ~ U(-1.301, 1.079)`。
pilot 的 6 筆（idx 0–5）種子與設定完全一致，**直接併入**，沒有重跑。

**L = 100。** 先前 GW 校準遇過「posterior 樣本數不足 L 會讓 rank 分母失真」
的問題；這裡逐筆記錄 `n_posterior` 與 `L_effective`，**20 筆的 posterior
最少 478 個樣本**，全部 ≥ L，所以 `L_effective = 100` 對每一筆都成立，
分母沒有失真。

**單筆上限 1500 s（25 分）**，取自 pilot 觀察到的量級。
這需要改 harness：`_BudgetGuard` 的 `run_timeout_s` 是**每個 shard** 的預算，
resume 之後會重新計時，所以原本無法累計。`scripts/run_image_sbc.py` 現在
把 `cum_wall_s` 與 `final` 存進逐筆記錄，跨 shard 累加，超過 cap 才關閉為
`timeout_capped`。**只改 harness，沒有動 production 程式碼。**

## X.17.2 逐筆結果

`docs/calibration/image_gr_eternal_n20_injections.csv`。

| idx | SNR | 流量 (Jy) | 結果 | 累積牆鐘 | shards | n_post |
|---|---|---|---|---|---|---|
| 11 | 3.3 | 0.062 | ok | 49.9 s | 1 | 478 |
| 17 | 3.4 | 0.061 | ok | 55.7 s | 1 | 702 |
| 18 | 4.0 | 0.056 | ok | 53.9 s | 1 | 644 |
| 4 | 4.8 | 0.077 | ok | 64.0 s | 1 | 722 |
| 19 | 5.3 | 0.098 | ok | 58.9 s | 1 | 799 |
| 6 | 7.2 | 0.096 | ok | 45.2 s | 1 | 690 |
| 14 | 13.2 | 0.189 | ok | 72.0 s | 1 | 734 |
| 9 | 16.3 | 0.230 | ok | 128.3 s | 1 | 1285 |
| 1 | 31.3 | 0.491 | ok | 256.0 s | 1 | 1052 |
| 8 | 40.2 | 0.710 | ok | 123.4 s | 1 | 1179 |
| 7 | 57.3 | 0.770 | ok | 223.7 s | 2 | 794 |
| 0 | 73.5 | 1.052 | ok | 552.0 s | 1 | 881 |
| 12 | 88.1 | 1.415 | ok | 367.4 s | 3 | 1019 |
| 15 | 111.3 | 1.679 | ok | 426.7 s | 3 | 919 |
| 3 | 149.2 | 2.082 | ok | 627.0 s | 1 | 885 |
| **10** | **150.1** | 2.157 | ok | **1466.8 s** | 8 | 1102 |
| **2** | **201.2** | 3.619 | **timeout_capped** | **1500.1 s** | 2 | — |
| 16 | 225.9 | 3.488 | ok | 558.9 s | 3 | 970 |
| 5 | 334.1 | 5.816 | ok | 533.6 s | 2 | 999 |
| 13 | 585.7 | 9.936 | ok | 863.9 s | 5 | 988 |

**收斂率 19/20 = 95%。** 唯一未收斂的是 idx 2（SNR 201.2），用滿 1500 s。
19 筆收斂的牆鐘：最小 45 s、5/50/95% = 49 / 224 / 924 s、最大 1467 s、平均 344 s。
**N=20 實際總計算量 2.23 小時。**

## X.17.3 耗時模型重新擬合

| 擬合 | 資料點 | 模型 |
|---|---|---|
| pilot（X.16） | 4 個收斂點 | `t = -236.0 + 171.0 ln(SNR)` |
| **N=20** | **19 個收斂點** | **`t = -242.8 + 167.1 ln(SNR)`** |

**兩者幾乎相同**（斜率 171.0 → 167.1，截距 -236.0 → -242.8）。

### pilot 對高 SNR 端「失準」的疑慮沒有成立

X.16 曾因為 SNR 201.2 實測 1438 s 遠高於擬合預測的 ~670 s 而懷疑
高 SNR 端有系統性偏差。N=20 顯示**那是散布，不是趨勢**：

| SNR | 實測 | pilot 擬合 | N=20 擬合 |
|---|---|---|---|
| 585.7 | 863.9 s | 853.7 s | 822.1 s |
| 334.1 | 533.6 s | 757.8 s | 728.3 s |
| 225.9 | 558.9 s | 690.8 s | 662.9 s |
| **150.1** | **1466.8 s** | 620.9 s | 594.5 s |
| **149.2** | **627.0 s** | 620.0 s | 593.6 s |
| 111.3 | 426.7 s | 569.8 s | 544.7 s |

最高 SNR 的那一筆（585.7）反而**貼合擬合**。真正的問題是散布：
**R² = 0.584，殘差 rms 231 s、最大 872 s。**

### 散布不是由參數驅動的

idx 3 與 idx 10 的 SNR 幾乎相同（149.2 / 150.1），真值也幾乎相同
（`M` 3.02e9 / 3.00e9、`ring_width_frac` 0.0221 / 0.0222、
`a_star` 0.938 / 0.877、`|cos i|` 0.747 / 0.879），**耗時卻差 2.3 倍**
（627 s vs 1467 s，後者用了 8 個 shard）。

對 19 筆的耗時殘差與每個真值參數做 Spearman 相關，**沒有任何一個顯著**：

| 參數 | ρ | p |
|---|---|---|
| `position_angle` | +0.416 | 0.077 |
| `a_star` | +0.379 | 0.110 |
| `log10_total_flux_jy` | −0.332 | 0.166 |
| `i` | +0.300 | 0.212 |
| `ring_width_frac` | −0.268 | 0.267 |
| `M` | +0.112 | 0.647 |
| `\|cos i\|` | −0.009 | 0.972 |

**結論：殘差主要是取樣器本身的 run-to-run 變異**，與
`BOUNCE_PREFLIGHT_AUDIT.md` 記錄過的「dynesty 在固定種子下仍不可重現」
一致。這表示**用平均值外推 N=100 是可靠的，但逐筆預測不可靠**。

## X.17.4 N=100 耗時重新估計

| 依據 | 單筆平均 | N=100 |
|---|---|---|
| pilot 擬合（X.16） | 429 s | 11.93 h |
| N=20 擬合 | 409 s | 11.35 h |
| **這 20 筆的實測平均（含被 cap 的那筆）** | **401 s** | **11.15 h** |

**跟原本的 11.9 小時沒有明顯差異**（差約 6%，且是往下修）。
N=20 實際花了 2.23 h，線性外推到 N=100 是 11.15 h，三種算法一致。

**Sgr A\*（未測）**：先驗預測中位 SNR 126.5（M87\* 是 48.8）、
7.2% 在 SNR > 10³。把 N=20 的擬合套到 Sgr A\* 的 SNR 分布，
單筆平均約 640 s，**N=100 約 17.8 h**，比 M87\* 貴約 1.6 倍。
這只是外推，沒有實測。

## X.17.5 初步 rank / coverage（N=19，**不是正式判定**）

`docs/calibration/image_gr_eternal_n20_rank_coverage.csv`。
被 cap 的 idx 2 沒有 posterior，因此排除——**這是與 SNR 相關的排除**
（被排除的那筆 SNR 201.2），1/20 的選擇效應雖小但存在，必須記著。

| 參數 | rank 平均 | rank 標準差 | KS p | 68% coverage | 90% coverage |
|---|---|---|---|---|---|
| `M` | 50.5 | 30.1 | 0.974 | 0.632 | 0.789 |
| `a_star` | 56.7 | 29.0 | 0.722 | 0.632 | 0.842 |
| `i` | 57.9 | 36.1 | 0.185 | **0.474** | **0.579** |
| `position_angle` | 45.0 | 29.3 | 0.722 | 0.579 | 0.895 |
| `ring_width_frac` | 41.6 | 28.7 | 0.256 | 0.737 | 0.842 |
| `log10_total_flux_jy` | **66.5** | 29.7 | **0.017** | 0.474 | 0.737 |
| **理論值** | **50.0** | **29.2** | — | **0.673** | **0.891** |

**coverage 的理論值不是 0.68 / 0.90**：用 L = 100 個抽樣的經驗分位數構成的
區間，覆蓋一個新抽樣的機率是 `level × L/(L+1)`，即 0.673 / 0.891。
N=19 的二項標準誤是 ±0.108（68%）與 ±0.071（90%）。

**方向性觀察（都不構成判定）：**

1. **六個參數的 KS p 值全部通過 Bonferroni 門檻**（6 參數、α=0.05 → p > 0.0083）。
2. **`log10_total_flux_jy` 偏差最明顯**：rank 平均 66.5（理論 50），
   rank 明顯偏向高端（19 筆裡有 8 筆 ≥ 86），KS p = 0.017——
   過 Bonferroni 但不過未校正的 0.05。rank 偏高代表真值傾向落在後驗樣本的
   上方，也就是**後驗相對真值偏低**。這是 N=100 時最該盯的一項。
3. **coverage 系統性偏低**：六個參數的 90% coverage 有五個低於理論值 0.891，
   `i` 的 0.579 低了 4.4σ、68% 的 0.474 低了 1.8σ。`i` 的 rank 標準差 36.1
   也明顯大於理論的 29.2（rank 往兩端堆積），與 coverage 偏低一致。
4. 方向一致（偏窄／偏低）但 N=19 無法區分這是真實的 under-coverage 還是抽樣噪聲。

**這 19 筆的 rank 與 CI 都已存進 `artifacts/image_sbc_n20/`，
日後跑到 N=100 可以直接併入，不需要重跑。**

## X.17.6 結論

- **這條路徑可行**：95% 收斂率、N=20 花 2.23 h、耗時模型與 pilot 一致。
- **N=100 約 11.2 小時**（M87\*），Sgr A\* 外推約 17.8 h。
- 建議繼續跑到 N=100 **之前**先看一件事：`log10_total_flux_jy` 的 rank 偏高
  與整體 coverage 偏低是否在更大樣本下持續。這兩項在 N=19 下都還在噪聲範圍內，
  但方向一致。

---

# X.18 `gr_eternal` 新參數化下 N=100 完整校準驗證

> image 通道第一次跑完完整規模的 SBC/coverage。
> **結論：發現統計上顯著的系統性偏差，不是 N=19 時的雜訊。**
> 依指示回報成因候選，未動手修正。不含任何科學結論。

## X.18.1 執行摘要

設定與 X.17 的 N=20 **完全相同**（`gr_eternal`、M87\*、6 維、
`use_closure_phases=False`、`bound='live'`/`sample='rwalk'`/`nact=2`、
`nlive=250`、`log10_total_flux_jy` 參數化、L=100、單筆上限 1500 s）。
`seed = 700000 + idx`，idx 0–99 是同一組不重疊序列；已存檔的 20 筆
**直接併入，沒有重跑**。

| | 值 |
|---|---|
| 起跑 | 100 |
| 收斂 | **94** |
| 被 cap（1500 s 未收斂） | **6** |
| 總計算量 | **10.07 小時**（N=20 的估計是 11.15–11.93 h） |
| 收斂牆鐘 5/50/95% | 51 / 188 / 698 s，平均 290 s |
| 最小 posterior 樣本數 | 478（L=100 分母對每一筆都精確） |

耗時估計準確：實測 10.07 h，估計 11.2 h（實測低 10%）。
**平行化**（3 個 worker、`OMP_NUM_THREADS=1`）把單次 shard 呼叫的計算量
從約 500 s 提升到約 1440 s，是能在這個環境跑完的關鍵。

## X.18.2 分母與選擇效應

**分母用實際收斂的 94 筆。** 被 cap 的 6 筆沒有 posterior，
比照 idx 2 的處理：記錄但不強行湊出一個看似收斂的 posterior。

**這個排除與 SNR 強相關，必須明說：**

| SNR 區間 | 筆數 | 被 cap | 比例 |
|---|---|---|---|
| [0, 50) | 56 | 0 | 0% |
| [50, 150) | 16 | 0 | 0% |
| [150, 400) | 22 | 3 | 14% |
| [400, ∞) | 6 | 3 | **50%** |

被 cap 的 6 筆 SNR 全部 ≥ 201（201, 250, 349, 573, 584, 792）。
**校準結論不涵蓋最亮的那一段先驗。** 不過把收斂的 94 筆依 SNR 中位數
切兩半，coverage 沒有明顯差異（例如 `i` 的 90% coverage 低 SNR 0.660、
高 SNR 0.766），所以這個選擇效應**大概不是**下面那些偏差的來源——
但它確實讓「高 SNR 端是否同樣校準」這個問題沒有被回答。

## X.18.3 完整規模的 rank / coverage（N=94）

`docs/calibration/image_gr_eternal_n100_rank_coverage.csv`。

| 參數 | rank 平均 | rank sd | 極端 rank 比例 | KS p | 68% cov | 90% cov | 90% 偏離 |
|---|---|---|---|---|---|---|---|
| `M` | 56.8 | 26.9 | 0.117 | 0.0478 | 0.702 | 0.883 | −0.3σ |
| `a_star` | 44.4 | 30.1 | 0.191 | 0.2361 | 0.596 | 0.809 | −2.6σ |
| `i` | 47.0 | **34.1** | **0.287** | 0.0194 | 0.489 | **0.713** | **−5.6σ** |
| `position_angle` | 52.4 | **33.3** | **0.234** | 0.0329 | 0.457 | 0.766 | **−3.9σ** |
| `ring_width_frac` | 45.5 | 31.0 | 0.213 | 0.0970 | 0.617 | 0.798 | −2.9σ |
| `log10_total_flux_jy` | **62.2** | 28.0 | 0.170 | **0.0016** | 0.606 | 0.830 | −1.9σ |
| **理論值** | **50.0** | **29.2** | **0.109** | — | **0.673** | **0.891** | — |

理論值說明：rank 分母 L=100 時 rank sd 為 `sqrt(((L+1)²−1)/12)` = 29.15；
極端 rank（≤5 或 ≥95）在均勻分布下佔 11/101 = 0.109；
coverage 的理論值是 `level × L/(L+1)` = 0.673 / 0.891，**不是** 0.68 / 0.90。
二項標準誤：68% ±0.048、90% ±0.032。Bonferroni 門檻（6 參數、α=0.05）p > 0.0083。

## X.18.4 兩項重點訊號：都持續，且都變得更明確

### (1) `log10_total_flux_jy` 的 rank 偏高 —— **確認為顯著**

| | N=19 | **N=94** |
|---|---|---|
| rank 平均 | 66.5 | **62.2**（理論 50） |
| KS p | 0.017（過 Bonferroni） | **0.0016（不過 Bonferroni 0.0083）** |
| 90% coverage | 0.737 | 0.830（−1.9σ） |

**這是六個參數裡唯一 KS 檢定不過 Bonferroni 的。** rank 偏高代表真值傾向
落在後驗樣本的上方，也就是**後驗系統性低估總流量**。

**不是先驗邊界效應**：rank 與「真值在先驗中的相對位置」的
Spearman ρ = +0.182（p = 0.079），而且四個先驗四分位的 rank 平均是
53.7 / 64.4 / 69.1 / 62.5——偏差在先驗中段最強，不是在邊界。

**不是 `I0 = F / G` 反算的數值問題**：成像流量與取樣流量的往返在
X.16 已驗證精確到 4.4e-16。

### (2) `i` 的 coverage 系統性偏低 —— **確認為顯著，而且是全面性的**

| | N=19 | **N=94** |
|---|---|---|
| `i` 90% coverage | 0.579 | **0.713（−5.6σ）** |
| `i` 68% coverage | 0.474 | 0.489（−3.8σ） |
| `i` rank sd | 36.1 | 34.1（理論 29.2） |

而且**不只 `i`**：**六個參數的 90% coverage 全部 ≤ 理論值**，平均 0.800
對上理論 0.891；68% coverage 六個裡五個低於理論，平均 0.578 對上 0.673。

**極端 rank 比例是關鍵診斷**：均勻分布下應該是 0.109，實測
0.117 / 0.191 / **0.287** / **0.234** / 0.213 / 0.170——五個參數是理論值的
1.6–2.6 倍。真值落在後驗兩端尾巴的次數遠超應有，這是
**後驗過窄（under-dispersed）** 的典型特徵，不是偏移。

## X.18.5 成因候選（回報，未動手）

### (a) 已確認的缺陷：likelihood 的模型樣板帶著雜訊

`VisibilityLikelihood.loglike` 目前是：

```python
sim_data = sim.simulate(theta, model_context, rng=np.random.default_rng(0))
model_vis = np.asarray(sim_data.data, dtype=complex)   # <- .data 含熱雜訊
```

`SimData.data` 是**加了熱雜訊之後**的可見度；無雜訊的預測在
`sim_data.metadata["vis_signal"]`。所以模型樣板帶著一組固定
（seed 0）的雜訊實現，σ = 0.05 Jy/baseline。

likelihood 的模型預測應該是 `signal(θ)`，雜訊只該透過 σ 進入。
**這是明確的缺陷。**

**但量級對不上：** 在典型 SNR 下樣板的 `|V|` 只被抬高約 **+0.27%**（中位數），
而後驗在 log10 流量上的 1σ 寬度是 0.0135 dex ≈ **3.17%**。
也就是約 **0.08σ** 的推力——不足以把 rank 平均從 50 推到 62（需要約 0.3σ）。
而且雜訊抬升在暗源上最大（平均 +14%），但 rank 偏差在高低 SNR 上幾乎相同
（低 SNR 61.0、高 SNR 63.5），與這個缺陷的特徵不符。

**結論：是真缺陷、可能是貢獻之一，但不足以單獨解釋。**

### (b) 後驗過窄：`nact = 2` 的鏈長

X.18.4 的極端 rank 比例指向後驗系統性過窄。`nact = 2` 讓 bilby 的
`AcceptanceTrackingRWalk` 平均只接受 **4 步**就結束一條鏈
（bilby log：`An average of 4 steps will be accepted`）。鏈太短會讓
巢狀取樣在每個 shell 內探索不足，後驗因此偏窄——**對每個參數都會發生**，
並在後驗最彎曲的方向最嚴重，與 `i`（0.287）和 `position_angle`（0.234）
最差的觀測一致。

`nact` 正是 `BOUNCE_PREFLIGHT_AUDIT.md` Part K 調查過的那個鏈長旋鈕
（當時比較過 nact=2 與 nact=8）。**這是目前最符合觀測特徵的候選。**

### (c) 在可見度**振幅**上用高斯 likelihood

`|V_obs|` 在 `|V_model|` 附近是 Rician 而非高斯分布，
`E[|V_obs|] > |V_signal|`（雜訊加功率）。amplitude-only 分析下這個
mis-specification 同時影響資料側與模型側，方向不容易單憑推理定下來，
需要實際計算才能說清楚。**列為候選，未量化。**

### (d) 已排除的候選

| 候選 | 證據 |
|---|---|
| 先驗邊界效應 | Spearman ρ = +0.182, p = 0.079；偏差在先驗中段最強 |
| `I0 = F / G` 反算的數值特性 | 流量往返精確到 4.4e-16（X.16） |
| L=100 分母失真 | 94 筆的 posterior 最少 478 個樣本 |
| SNR 選擇效應造成偏差 | 高低 SNR 兩半的 coverage 沒有明顯差異 |

## X.18.6 結論

**image 通道 `gr_eternal` 在 M87\* 上的第一次完整校準驗證已完成，
結果是「未通過」：**

1. `log10_total_flux_jy` 的 rank 分布顯著非均勻（KS p = 0.0016，
   不過 Bonferroni），後驗系統性低估總流量。
2. 六個參數的 90% coverage **全部** ≤ 理論值 0.891（平均 0.800），
   `i` 低 5.6σ、`position_angle` 低 3.9σ；極端 rank 比例是理論值的
   1.6–2.6 倍，指向後驗系統性過窄。
3. N=19 時觀察到的兩個方向性訊號**都不是雜訊**。

**這不是「gr_eternal 已校準」。** 在處理 X.18.5 的成因之前，
這條通道的後驗區間不能當作可信的不確定度。

已確認一個缺陷（模型樣板帶雜訊），但量級不足以單獨解釋；
最符合觀測特徵的候選是 `nact = 2` 的鏈長。**依指示未動手修正。**

94 筆的 rank 與 CI 都存在 `artifacts/image_sbc_n20/`，
修正之後重跑可以直接對照。

---

# X.19 `VisibilityLikelihood` 模型樣板誤用觀測雜訊資料修正

> X.18.5(a) 的修正。只處理這一項——**沒有動 `nact` 或任何取樣器設定
> （那是 (b)），也沒有動 likelihood 的機率模型形式（那是 (c)）**。
> 不含任何科學結論。

## X.19.1 問題：`data` 是觀測，不是預測

`SimData` 的契約（`simulators/base.py`）：

| 欄位 | 內容 |
|---|---|
| `data` | 主要資料陣列 |
| `metadata` | 輔助資訊 |
| `noise_realisation` | 加到訊號上的雜訊 |

在 `ImageShadowSimulator.simulate()` 裡：

```python
vis_signal   = _compute_visibilities(image, ...)      # 無雜訊的模型預測
noise        = (rng.standard_normal(...) + 1j * ...) * thermal_noise_jy
visibilities = vis_signal + noise                     # -> SimData.data
```

所以 **`SimData.data` 是「觀測」（訊號 + 雜訊），`metadata["vis_signal"]`
才是「模型預測」**。likelihood 需要的是後者：`p(d | θ)` 裡的模型項是
`signal(θ)`，雜訊只該透過 `σ` 進入。

`VisibilityLikelihood.loglike()` 原本寫的是：

```python
sim_data = sim.simulate(theta, model_context, rng=np.random.default_rng(0))
model_vis = np.asarray(sim_data.data, dtype=complex)        # <- 觀測,不是預測
```

於是每個模型樣板都帶著一組固定（seed 0）的熱雜訊實現，
σ = 0.05 Jy/baseline。likelihood 變成拿「量到的含雜訊」去比對
「預測的含另一組雜訊」。

**closure phase 那一項有完全相同的缺陷**：
`sim_data.metadata["closure_phases"]` 是從**含雜訊**的可見度算出來的
（觀測側的量），模型側卻也用了它。

## X.19.2 修正

`simulators/image_shadow.py` 現在同時輸出兩個版本，語意寫清楚：

| metadata 鍵 | 來源 | 用途 |
|---|---|---|
| `closure_phases` | 含雜訊的 `visibilities` | 觀測側 |
| `closure_phases_signal` | 無雜訊的 `vis_signal` | 模型樣板 |

`likelihoods/visibility.py` 改為：

```python
model_vis     = sim_data.metadata["vis_signal"]              # 振幅項
model_closure = sim_data.metadata["closure_phases_signal"]   # 相位項
```

`rng` 仍然傳入且仍然固定，所以被丟棄的那次雜訊抽樣不會讓 likelihood
在兩次評估之間變成隨機的。

## X.19.3 驗證：樣板不再被雜訊抬高

以 X.18.5 診斷時完全相同的方法，在**已存檔的 94 筆真值**上量測模型樣板
`|V|` 相對無雜訊預測被抬高的百分比：

| | 修正前（用 `sim_data.data`） | 修正後（用 `vis_signal`） |
|---|---|---|
| 中位數 | **+0.510%** | **0%**（依構造精確為零） |
| 平均 | +10.67% | 0% |
| 95 百分位 | +59.98% | 0% |
| 暗的一半（中位數） | **+13.012%** | 0% |
| 亮的一半（中位數） | +0.111% | 0% |

> **更正 X.18.5(a) 的一個數字。** 當時引用的「+0.27%」是在 40 筆的子集上
> 取的中位數，**掩蓋了很強的 SNR 依賴**：暗源那一半的中位抬高其實是
> **+13.0%**，亮源那一半才是 +0.11%。當時據此說「量級不足以單獨解釋」的
> 推論因此建立在一個被平均掉的數字上，應該打折看待。這一項的實際影響
> 比當時寫的大，但**是否足以解釋 X.18 的偏差要等重跑才知道**——
> rank 偏差在高低 SNR 幾乎相同（61.0 / 63.5），而這個缺陷在暗源上強得多，
> 兩者的特徵仍然對不上。

## X.19.4 對已存檔資料的影響：真值處的 lnL

用已存檔的 94 筆重算**真實參數處**的 lnL（沒有重跑 campaign）：

| | Δ lnL（修正後 − 修正前） |
|---|---|
| 中位數 | **+4.240** |
| 平均 | +4.280 |
| 5 / 95 百分位 | −2.40 / +10.55 |
| 真值處 lnL 變高的比例 | **81%** |
| 暗的一半（中位數） | +2.220 |
| 亮的一半（中位數） | +5.843 |

逐筆抽樣：

| idx | SNR | 修正前 | 修正後 | Δ |
|---|---|---|---|---|
| 11 | 3.3 | 28.673 | 24.225 | −4.447 |
| 4 | 4.8 | 27.027 | 28.931 | +1.904 |
| 1 | 31.3 | 20.818 | 26.378 | +5.560 |
| 8 | 40.2 | 22.752 | 25.252 | +2.501 |
| 0 | 73.5 | 15.557 | 27.403 | **+11.845** |
| 3 | 149.2 | 22.490 | 28.333 | +5.843 |
| 16 | 225.9 | 22.180 | 27.906 | +5.726 |
| 13 | 585.7 | 16.134 | 27.058 | **+10.924** |

**真值處的 lnL 系統性變高**，量級與預期一致：帶雜訊的樣板在 16 條基線上
造成的懲罰約為 `N/2 = 8` nats 的量級。修正前 likelihood 在系統性地
懲罰真實參數，這正是會讓後驗偏移或偏窄的機制。

**這不代表 X.18 的偏差已經解決**——要重跑 campaign 才知道。

## X.19.5 順帶記錄（未處理）

`loglike()` 目前先呼叫 `build_ring_image(theta, model_context)` 做可表示性
檢查，接著 `sim.simulate()` 內部又建了一次同樣的影像。**影像被建了兩次**，
likelihood 的成本因此約為必要值的兩倍。這是 X.16 引入可表示性檢查時帶進來的，
與本輪的缺陷無關，**未修**。

## X.19.6 產出

| 路徑 | 內容 |
|---|---|
| `tests/test_image_uv_sampling.py::TestModelTemplateIsNoiseFree` | 5 項 regression test：simulator 分離預測與觀測、`loglike` 必須等於以 `vis_signal` 為樣板算出的值（且**不等於**以 `.data` 算出的值）、換雜訊種子模型樣板不變、closure phase 同樣鎖住、雜訊確實會抬高 `\|V\|` |
