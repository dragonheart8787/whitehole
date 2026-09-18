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

---

# X.20 模型樣板修正後的校準重新驗證

> 用修正後的 `VisibilityLikelihood`（X.19）對**完全相同的 100 筆注入**重跑
> 推論，與 X.18 並排比較。**只改了 (a)，`nact` 仍是 2**，避免混淆。
> 不含任何科學結論。

## X.20.1 為什麼不需要「只重跑推論、不重跑模擬」

注入完全由 `seed = 700000 + idx` 決定，所以真值與資料**逐位元重現**，
重新產生 100 筆只要 **0.55 s（5.5 ms/筆）**，且與 X.18 存檔的
`theta_true` 比對 **0 個不一致**。模擬不是成本，巢狀取樣才是，
而 likelihood 改了、取樣**必須**重跑。架構上不需要任何改動。

**用了全新的輸出目錄**（`artifacts/image_sbc_fixA/`），否則 bilby 會
resume X.18 用舊 likelihood 算到一半的鏈。

## X.20.2 執行對照

| | X.18（舊樣板） | X.20（修正後） |
|---|---|---|
| 起跑 / 收斂 / 被 cap | 100 / 94 / 6 | 100 / **95** / **5** |
| 總計算量 | 10.07 h | **10.23 h** |
| 收斂牆鐘平均 | 290 s | 309 s |
| 被 cap 的 SNR | 201, 250, 349, 573, 583, 792 | 140, 349, 573, 583, 792 |

**收斂行為基本不變**（+1 筆收斂、計算量 +1.6%）。被 cap 的仍幾乎都是
高 SNR；唯一的變化是 SNR 201 與 250 這兩筆改為收斂，而 SNR 140 那筆
（X.18 花 1467 s 勉強收斂的 idx 10）這次被 cap——與先前判定的
「散布主要是取樣器 run-to-run 變異」一致。

## X.20.3 rank / coverage 並排比較

`docs/calibration/image_gr_eternal_n100_templatefix_comparison.csv`。
理論值：rank 平均 50.0、極端 rank 0.109、68%/90% coverage 0.673/0.891
（`level × L/(L+1)`）；Bonferroni 門檻 p > 0.0083；90% coverage 的
二項標準誤 ±0.032。

| 參數 | rank 平均 前→後 | 極端 rank 前→後 | KS p 前→後 | 90% cov 前→後 |
|---|---|---|---|---|
| `M` | 56.8 → **50.1** | 0.117 → 0.137 | 0.0478 → **0.8354** | 0.883 → 0.863 |
| `a_star` | 44.4 → 43.4 | 0.191 → 0.168 | 0.2361 → 0.1345 | 0.809 → 0.832 |
| `i` | 47.0 → 48.0 | **0.287 → 0.158** | 0.0194 → **0.5875** | **0.713 → 0.842** |
| `position_angle` | 52.4 → 56.5 | 0.234 → 0.200 | 0.0329 → 0.0193 | 0.766 → 0.800 |
| `ring_width_frac` | 45.5 → **51.0** | 0.213 → 0.158 | 0.0970 → **0.7817** | 0.798 → 0.842 |
| `log10_total_flux_jy` | **62.2 → 39.6** | 0.170 → 0.147 | 0.0016 → **0.0014** | 0.830 → 0.853 |

| 彙總 | X.18 | X.20 | 理論 |
|---|---|---|---|
| 六參數 90% coverage 平均 | 0.800 | **0.839** | 0.891 |
| 六參數 68% coverage 平均 | 0.578 | **0.614** | 0.673 |
| 未通過 Bonferroni 的參數數 | 1 | **1** | 0 |

## X.20.4 兩個重點問題的答案

### (1) `log10_total_flux_jy` 的 KS p：**沒有改善，但偏差方向翻轉了**

KS p 0.0016 → 0.0014（實質不變，仍是唯一不過 Bonferroni 的）。
**但 rank 平均從 62.2 翻到 39.6**——從「後驗低估流量」變成「後驗高估流量」。
這不是同一個偏差沒被修好，是**偏差換了一邊**。

### (2) coverage 平均：**明顯往理論值靠近，但沒有到位**

0.800 → **0.839**（理論 0.891）。偏離幅度全面縮小：
`i` 從 −5.6σ 收到 −1.5σ、`position_angle` 從 −3.9σ 收到 −2.9σ。
極端 rank 比例六個裡五個下降，`i` 從 0.287（理論值的 2.6 倍）
降到 0.158（1.4 倍）。

## X.20.5 rank 翻轉指向的是 (c)，不是 (b)

流量 rank 從 62.2 翻到 39.6，這個**符號翻轉**本身就是資訊。

`|V_obs| = |V_signal + n|` 是 Rician 分布，`E|V_obs| > |V_signal|`，
超出量約 `σ² / (2|V|)`。likelihood 用的是以無雜訊 `|V_model|` 為中心的
高斯，所以擬合必須**把流量往上推**才能對上被雜訊墊高的觀測振幅
→ 後驗高估 → rank 偏低。

**量級檢查**：在這 95 筆上，資料側的預期抬高中位數是 **1.386%**，
而後驗在流量上的 1σ 寬度是 **3.08%**，也就是約 **0.45σ** 的推力。
實測的 rank 平均位移是 `(50 − 39.6)/29.15 = 0.36` 個 rank 標準差。
**方向與量級都吻合。**

**修正前後的機制**：X.18 的樣板**也**被自己的雜訊墊高，
**意外地部分抵銷**了資料側的 Rician 抬高，淨效果是 rank 62.2；
把樣板修正之後那個抵銷消失，資料側的 Rician 抬高就裸露出來，
變成 rank 39.6。

**這代表 (a) 與 (c) 原本是互相掩蓋的。** 修掉 (a) 是正確的
（樣板本來就不該帶雜訊），但它同時暴露了 (c)。

## X.20.6 結論

**(a) 是真正的貢獻者，但不是唯一的問題。** 分開講：

1. **(a) 修正解決了四個參數的 rank 問題。** `M`（KS 0.0478 → 0.8354，
   rank 56.8 → 50.1）、`i`（0.0194 → 0.5875，極端 rank 0.287 → 0.158）、
   `ring_width_frac`（0.0970 → 0.7817）都從邊緣或不合格變成明確通過。
   `i` 的 coverage 從全場最差的 −5.6σ 收到 −1.5σ。
2. **coverage 的系統性偏低減輕了但沒有消失**：平均 0.800 → 0.839
   對上理論 0.891，六個參數仍全部 ≤ 理論值，`position_angle` 還有 −2.9σ。
   這個殘留與 (b)（`nact = 2` 鏈長不足導致後驗偏窄）的特徵一致，
   但本輪**沒有測 (b)**，所以是推論不是結論。
3. **流量的 rank 偏差換邊而非消失**，量級與方向都指向 **(c)**
   （在 Rician 分布的振幅上用高斯 likelihood），不是 (b)。

**這仍然不是「gr_eternal 已校準」。** 還有兩個獨立的殘留：
(b) 造成的後驗偏窄，與 (c) 造成的流量偏移。
**依指示未動手測 (b) 或 (c)。**

95 筆的 rank 與 CI 存在 `artifacts/image_sbc_fixA/`，
X.18 的 94 筆仍在 `artifacts/image_sbc_n20/`，兩批可直接對照。

---

# X.21 `gr_eternal` nact 鏈長對 coverage 影響的定向 pilot

測 X.20.6 第 2 點列出的候選 **(b)**：殘留的 coverage 偏低
（六參數 90% 平均 0.839 對上理論 0.891）是不是 `nact = 2` 鏈長不足、
使後驗偏窄造成的。**這是定向 pilot，不是完整 campaign**，也**沒有碰 (c)**、
沒有改動任何 likelihood 或機率模型程式碼。

## X.21.1 設計

| | `base` | `nact8` |
|---|---|---|
| 來源 | X.20 存檔 `artifacts/image_sbc_fixA/` | 本輪重跑 `artifacts/image_sbc_nact8/` |
| likelihood | 已修正的 `VisibilityLikelihood`（樣板用 `vis_signal`） | **同左** |
| `nlive` / `sample` / `bound` / `dlogz` | 250 / `rwalk` / `live` / 0.1 | **同左** |
| `nact` | 2（平均接受 4 步） | **8（平均接受 16 步）** |
| 每筆總時間上限 | 1500 s | **6000 s**（＝ 4×，與鏈長同比例放大） |

**抽樣方法（不是挑最差的）**：把 X.20 收斂的 95 筆依 `network_snr` 排序，
等分成 18 層，每層以 `default_rng(20260917)` 抽一筆。
選出的 18 筆 SNR 四分位是 3.8 / 12.6 / 36.4 / 161.1 / 397.8，
母體是 3.3 / 10.5 / 37.1 / 149.7 / 585.7 —— 分布相符。
**刻意與 G-1 的做法不同**：G-1 的 12 筆是依「在原 campaign 中 miss」挑出來的，
存在選擇效應而必須靠彙總比例判讀；本輪沒有這個問題，可以直接逐筆配對。

**配對成立的理由**：`run_one` 由 `seed = 700000 + idx` 重建真值與雜訊
（`sample_prior(default_rng(seed))` → `simulate(..., default_rng(seed+1))`），
兩組因此看到**完全相同的注入與資料**，唯一差異是鏈長。
已逐筆核對 `theta_true` 與 `network_snr` 完全一致。

## X.21.2 行為驗證：`nact` 確實生效

依 K.2.3，**檢查 kwargs 字典會給出假陰性**——`nact` 被 bilby 吃進自己的
sampler 物件，bilby result 的 `sampler_kwargs` 裡根本沒有這個鍵
（實測 X.20 的存檔 result JSON 只有 `walks: 100`，而那是無作用的 dynesty 預設）。
因此用兩個行為證據：

1. **bilby 自己的日誌**：
   `Using the bilby-implemented ensemble rwalk sampling method with ACT
   estimated chain length. An average of 16 steps will be accepted up to
   chain length 5000.`
   `16 = 2 × nact = 2 × 8` ✅（`base` 是 `2 × 2 = 4`）。
2. **ncall 量級**：配對 18 筆的 ncall 中位數上升 **3.00 倍**（見下節）。

## X.21.3 成本：18/18 收斂，ncall 3.00×、耗時 3.05×

| | `base` | `nact8` | 比例 |
|---|---|---|---|
| 收斂 | 18/18 | **18/18** | — |
| ncall 中位數 | 6.93×10⁴ | 2.23×10⁵ | **3.00×**（平均 2.98×） |
| 耗時中位數 | 185 s | 642 s | **3.05×**（平均 3.18×） |
| 18 筆總耗時 | 1.29 h | **4.00 h** | 3.11× |

逐筆比例落在 2.30–3.70×（唯一的離群是 `0097` 的耗時 6.48×，
但它的 ncall 比例是 2.97×，與其他筆一致，所以那是機器負載而非取樣行為）。
**與 G-1 不同的是，本通道沒有出現「最難樣本從分鐘級惡化到小時級」**：
最貴的 `0080` 從 641 s 變成 2058 s，仍在 6000 s 的 `_BudgetGuard` 上限內，
18 筆**沒有任何一筆被 cap 截斷**。

## X.21.4 主結果：後驗區間寬度**沒有改變**

(b) 的假說是「鏈長不足 → 後驗偏窄」。最直接的檢定是同一筆注入在兩種鏈長下的
90% 區間寬度比。108 個（注入 × 參數）配對：

| | 值 |
|---|---|
| `w90(nact8) / w90(base)` 中位數 | **0.9946** |
| 同上平均 | **0.9990**，95% CI **0.975 – 1.024** |
| Wilcoxon signed-rank（全部 108 對的 log 比） | **p = 0.516** |

逐參數的中位數比是 `M` 0.988、`a_star` 1.027、`i` 0.995、
`position_angle` 1.008、`ring_width_frac` 0.930、`log10_total_flux_jy` 0.990，
Wilcoxon p 值 0.197–0.865，**沒有一個參數達到顯著**。

**把鏈長從平均 4 步加到 16 步，對後驗寬度的效應是 0.0%，
上下界被壓在 ±2.4% 以內。** 這與 G-1 的情況不同：
G-1 至少量到 +3.6% 的系統性增寬（p = 0.0273），本通道**連方向都沒有**。

## X.21.5 coverage 的變化是雜訊，不是改善

| | `base`（18 筆） | `nact8`（18 筆） | 理論 |
|---|---|---|---|
| 90% coverage 平均 | 0.806 | 0.843 | 0.891 |
| 68% coverage 平均 | 0.574 | **0.565** | 0.673 |

90% 那一列看起來動了 +0.037，但**不能當成改善**，有三個理由：

1. **配對後只有 6/108 個指標翻面**（miss→hit 5、hit→miss 1），
   McNemar **p = 0.219**。
2. **68% coverage 往反方向動**（0.574 → 0.565，翻面 4:5，p = 1.000）。
   真的變寬的話兩個信賴水準應該同向。
3. **區間寬度沒變**（X.21.4）。寬度不變而 coverage 變動，
   只能來自後驗抽樣的隨機差異（每筆抽 L = 100 draws），不是機制性的改善。

這 18 筆在 `base` 下的 90% coverage 平均是 0.806，而全部 95 筆是 0.839——
子集本身就比母體低 0.033，這也是 n = 18 下的抽樣起伏量級。

## X.21.6 效應量對上缺口：差兩個數量級

把後驗近似成高斯、真實誤差比後驗寬 `k` 倍，
則名目 90% 區間的實際 coverage 是 `2Φ(1.645/k) − 1`。反解：

| 觀測 coverage | 名目 | 需要的區間放寬 |
|---|---|---|
| 0.839（X.20 全 95 筆，90%） | 0.90 | **+11.7%** |
| 0.614（X.20 全 95 筆，68%） | 0.68 | +14.7% |

**需要 +11.7%，量到的是 0.0%（±2.4%）。**

（**X.25.6 更正**：此處原寫 +17.3%，是我把名目 0.90 與 0.891 基準混用算錯的；一致地算是舊參考值 +14.4%、MC 參考值 **+11.7%**。結論不變。）
相比之下 G-1 是「需要 +43%，量到 +3.6%」。
本通道的情況比 G-1 更乾脆：不是效應太小，而是**量不到效應**。

## X.21.7 必須一併說明的限制

1. **n = 18，統計力有限。** 108 個 coverage 指標的 SE 約 0.03，
   所以這個 pilot 本來就分辨不出 0.05 量級的 coverage 變化。
   **真正有鑑別力的是配對寬度比**（±2.4%），結論建立在它上面，
   不是建立在 coverage 的點估計上。
2. **只測了 `nact` 2 → 8 這一個對照。** 沒有測 `nact = 32`，
   也沒有測 `nlive` 加倍。依 G-1 的機制論證，`nact` 是更直接的旋鈕，
   但「更大的 nact 也不會有效」是推論，**不是量測結果**。
3. **耗時比例會受機器負載影響**（`0097` 的 6.48×）。ncall 比例不受影響，
   所以成本結論以 ncall 為準。
4. **這只說明 (b) 不是殘留的主因，不表示殘留已經被解釋。**
   (c)（在 Rician 振幅上用高斯 likelihood）仍未測。

## X.21.8 一句話結論

**把 `nact` 從 2 加到 8（平均接受步數 4 → 16）對後驗寬度的效應是
0.0%（95% CI −2.5%…+2.4%，Wilcoxon p = 0.52），
而補上 0.05 的 coverage 缺口需要 +11.7% 的增寬（X.25.6 更正，原寫 +17.3%）——
效應量不是「杯水車薪」而是「量不到」，代價卻是 ncall 3.00 倍；
候選 (b) 因此在這個通道被排除，殘留的 under-coverage 另有來源。**

**全規模成本外推（僅供參考，不建議執行）**：X.20 的 95 筆在 `nact = 2`
下花了 8.15 h，依 3.0–3.2× 推算 `nact = 8` 的 N = 100 campaign 約
**25–26 小時**。既然效應量量不到，這筆支出沒有理由花。

## X.21.9 產出

- `artifacts/image_sbc_nact8/M87star/`（18 筆，18/18 收斂）
- `docs/calibration/image_gr_eternal_nact_pilot.csv`
  （逐筆 SNR、耗時、ncall，以及六參數的 `in90` / `w90` / `rank` 兩組並排）
- `scripts/analyse_nact_pilot.py`（配對比較）
- `scripts/run_image_sbc.py` 的 `--nact` / `--indices`
- `tests/test_sampler_settings_effective.py::TestChainLengthKnob::test_run_metadata_carries_the_cost_counter`

---

# X.22 `gr_eternal` nlive 對 coverage 影響的定向 pilot

測 X.21 排除 (b) 之後**唯一還沒測過的取樣器旋鈕**。動機是樣態：殘留的
under-coverage 在六個參數上**大致均勻**（X.20 的 90% coverage 0.800–0.863），
而不是集中在某個簡併方向上——這比 (b) 的「鏈長不足、脊上探索不夠」更像
nested sampling 在 `nlive` 不足時的一般性偏差。
**與 X.21 同樣是定向 pilot，沒有碰 (c)，沒有改動任何 likelihood 程式碼。**

## X.22.1 設計：與 (b) pilot 完全相同的方法論

用**同一批 18 筆**（X.21.1 依 `network_snr` 分 18 層、每層以
`default_rng(20260917)` 抽一筆的分層樣本），因此兩次 pilot 共用同一個
`base` 對照組，三組之間可直接互比。

| | `base` | `nlive500` |
|---|---|---|
| likelihood | 已修正（樣板用 `vis_signal`） | **同左** |
| `nlive` | 250 | **500** |
| `nact` | 2 | **2（維持不變）** |
| `sample` / `bound` / `dlogz` | `rwalk` / `live` / 0.1 | 同左 |
| 每筆總時間上限 | 1500 s | 6000 s |

真值與雜訊同樣由 `seed = 700000 + idx` 重建，逐筆配對。
**只動 `nlive` 一個旋鈕**，所以效應可單獨歸因。

## X.22.2 行為驗證：`nlive` 確實生效

這次不看 kwargs 字典（K.2.1 的 `walks` 被原樣回報卻毫無作用、
K.2.3 的 `nact` 即使生效也不出現在 bilby 的 `sampler_kwargs` 裡，
已經連續踩過兩次）。改用 nested sampling 本身的三個可預測後果：

| 量 | 理論預期 | 實測（18 筆中位數） | 逐筆範圍 |
|---|---|---|---|
| 迭代數 `niter` 比 | ×2（`niter ≈ nlive × H`） | **2.01** | 1.96 – 2.14 |
| `ln Z` 誤差比 | ×1/√2 = 0.707（`≈ √(H/nlive)`） | **0.727** | 0.691 – 0.751 |
| information gain `H` 比 | **×1.00**（後驗相對先驗的性質，與取樣器無關） | **1.008** | 0.944 – 1.095 |
| ncall 比 | ×2 | **2.00** | 1.85 – 2.12 |

三個量各自往**不同方向**、以各自的理論倍率移動，而應該不動的 `H` 沒動。
`nlive` 生效無疑義。

## X.22.3 成本：18/18 收斂，ncall 2.00×、耗時 1.96×

| | `base` | `nlive500` | 比例 |
|---|---|---|---|
| 收斂 | 18/18 | **18/18** | — |
| ncall 中位數 | 6.93×10⁴ | 1.42×10⁵ | **2.00×**（平均 1.97×） |
| 耗時中位數 | 185 s | 409 s | **1.96×**（平均 2.06×） |
| 18 筆總耗時 | 1.29 h | **2.33 h** | 1.81× |

**成本以 ncall 為準，不以耗時為準。** 兩筆的耗時比明顯脫隊
（`0097` 是 4.53×、`0080` 是 0.81×）而它們的 ncall 比分別是 1.98× 與 1.87×，
與其他筆一致——那是機器負載與分片重啟的產物，不是取樣行為。
`0080` 另有一個必須記錄的情況：容器在它跑到一半時重啟、resume 檔案損毀，
**該筆是從頭重跑的**（`error` 紀錄與殘檔已刪除後重建），
所以它的耗時不可與其他筆同樣解讀，ncall 仍然有效。

沒有任何一筆被 6000 s 的 `_BudgetGuard` 截斷。

## X.22.4 主結果：後驗區間寬度**沒有變寬，點估計還略窄**

沿用 X.21 證明有效的判讀：配對的 90% 區間寬度比，108 個（注入 × 參數）對。

| | 值 |
|---|---|
| `w90(nlive500) / w90(base)` 中位數 | **0.9900** |
| 同上平均 | **0.9894**，95% CI **0.959 – 1.020** |
| Wilcoxon signed-rank（全部 108 對的 log 比） | **p = 0.123** |

逐參數中位數比：`M` 0.994、`a_star` 1.007、`i` 0.990、`position_angle` 0.978、
`ring_width_frac` 0.954、`log10_total_flux_jy` 1.003，
Wilcoxon p = 0.265–0.865，**沒有一個達到顯著**。

**把 `nlive` 從 250 加到 500，後驗寬度的點估計是 −1.1%，
95% 信賴上界是 +2.0%。方向與「後驗偏窄需要變寬」相反。**

## X.22.5 coverage：沒有改善，90% 那一列還往下走

| | `base`（18 筆） | `nlive500`（18 筆） | 理論 |
|---|---|---|---|
| 90% coverage 平均 | 0.806 | **0.787** | 0.891 |
| 68% coverage 平均 | 0.574 | 0.574 | 0.673 |

配對翻面數（McNemar）：

- **90%：只有 2/108 個指標翻面**，而且**兩個都是 hit → miss**（p = 0.500）。
- **68%：12/108 翻面，6 上 6 下**（p = 1.000）。

90% 只有 2/108 翻面這件事本身就是結論的一部分：
**兩組的後驗幾乎是同一個後驗**，`nlive` 加倍沒有改變它的形狀，
只是把它取樣得更細（posterior draws 從約 800 增到約 1800，
但 rank 與 CI 都先 thin 到 L = 100，所以這不影響比較）。

## X.22.6 效應量對上缺口

沿用 X.21.6 的算法（後驗近似高斯、真實誤差寬 `k` 倍，
名目 90% 的實際 coverage 是 `2Φ(1.645/k) − 1`）：

| 觀測 coverage | 名目 | 需要的區間放寬 |
|---|---|---|
| 0.839（X.20 全 95 筆） | 0.90 | **+11.7%** |
| 0.806（本 18 筆子集） | 0.90 | +26.6% |

**需要 +11.7%，量到 −1.1%（95% 上界 +2.0%）。**

（**X.25.6 更正**：此處原寫 +17.3%，是我把名目 0.90 與 0.891 基準混用算錯的；一致地算是舊參考值 +14.4%、MC 參考值 **+11.7%**。結論不變。）

三組並排：

| pilot | 旋鈕 | 量到的寬度變化 | 95% CI | 成本（ncall） |
|---|---|---|---|---|
| X.21 | `nact` 2 → 8（鏈長 ×4） | **0.0%** | −2.5% … +2.4% | 3.00× |
| X.22 | `nlive` 250 → 500 | **−1.1%** | −4.1% … +2.0% | 2.00× |
| 需求 | — | **+11.7%**（X.25.6 更正） | — | — |

## X.22.7 必須一併說明的限制

1. **n = 18，與 X.21 相同。** 108 個 coverage 指標的 SE 約 0.03，
   所以結論同樣建立在**配對寬度比**上，不是 coverage 點估計。
   「90% coverage 掉到 0.787」不是「`nlive` 讓校準變差」的證據，
   它只有 2 個指標的位移。
2. **只測了 `nlive` 250 → 500 這一個對照。** 沒測 ×4 或更高。
   不過 `niter` 與 `ln Z` 誤差都精確地照 `nlive` 的理論倍率走，
   顯示這個區間裡取樣已經是收斂的；要主張 `nlive = 1000` 會突然有效，
   需要一個機制上的理由，本輪沒有找到。**這是推論，不是量測。**
3. **`0080` 因容器重啟而從頭重跑**（X.22.3），其耗時不可與其他筆同樣解讀。
4. **兩組共用同一個 `base` 對照組**，所以 X.21 與 X.22 的比較不是獨立的：
   `base` 那一側的抽樣起伏會同時影響兩份報告。

## X.22.8 一句話結論

**把 `nlive` 從 250 加到 500（迭代數 ×2.01、`ln Z` 誤差 ×0.727、
`H` 不變，行為驗證無疑義）對後驗寬度的效應是 −1.1%
（95% CI −4.1%…+2.0%，Wilcoxon p = 0.123，方向與需求相反），
而補上 0.05 的 coverage 缺口需要 +11.7%（X.25.6 更正，原寫 +17.3%）；
`nlive` 加倍因此與 `nact` 一樣被排除，代價 ncall 2.00× 也沒有理由花。**

由於效應量被排除，**不提出完整規模 campaign 的建議**
（僅記錄外推值供參考：N = 100 在 `nlive = 500` 下約 **15–17 小時**，
由 `nact = 2` 的 8.15 h 乘上 1.81–2.06 倍）。

## X.22.9 這代表候選清單需要重新想

到此為止三個候選的狀態：

| 候選 | 狀態 | 能解釋「六參數均勻偏窄」嗎 |
|---|---|---|
| (a) 模型樣板誤用含雜訊資料 | **已修正**（X.19/X.20） | 部分——修正後平均 0.800 → 0.839，但沒補完 |
| (b) `nact` 鏈長不足 | **已排除**（X.21，0.0% ± 2.4%） | 否 |
| (c) Rician 振幅上用高斯 likelihood | 未測 | **否**——它預測的是**流量單一參數的偏移**（方向與 0.45σ 的量級都已核對過，見 X.20.5），不是六個參數一致的區間過窄 |
| `nlive` 不足 | **已排除**（X.22，−1.1%，CI 上界 +2.0%） | 否 |

**兩個取樣器旋鈕都被排除，而且是以「取樣已收斂」的方式排除的**——
`niter`、`ln Z` 誤差、`H` 三個量都照理論走，後驗形狀在 4× 鏈長與 2× 活點下
都不動。這表示殘留的 under-coverage **不是取樣精度問題**。
**依指示，本輪不往下猜新的候選方向。**

## X.22.10 產出

- `artifacts/image_sbc_nlive500/M87star/`（18 筆，18/18 收斂）
- `docs/calibration/image_gr_eternal_nlive_pilot.csv`
  （逐筆 SNR、耗時、ncall、`niter`、`ln Z` 誤差、`H`，
  以及六參數的 `in90` / `w90` / `rank` 兩組並排）
- `scripts/analyse_sampler_pilot.py`（通用配對比較，X.21 的
  `analyse_nact_pilot.py` 保留不動）

---

# X.23 可見度雜訊 sigma 一致性核對

X.21 / X.22 用行為驗證排除了兩個取樣器旋鈕，代表取樣器忠實重建了
**likelihood 程式碼所定義的**後驗。那麼殘留的「六參數均勻偏窄」就必須來自
likelihood 定義的機率模型本身。全域的變異數尺度不匹配（假設的 `sigma`
比實際小）會讓所有參數**同等比例**地過窄，樣態正好吻合。

**本節是純粹的程式碼閱讀 + 數值核對，沒有跑任何取樣，沒有改動任何
production 程式碼。**

## X.23.1 兩側的 sigma 各自是什麼

**模擬器**（`ImageShadowSimulator.simulate`，`image_shadow.py:467, 491-494`）：

```python
thermal_noise_jy = float(context.get("thermal_noise_jy", 0.05))
...
noise_re = rng.standard_normal(len(vis_signal)) * thermal_noise_jy
noise_im = rng.standard_normal(len(vis_signal)) * thermal_noise_jy
visibilities = vis_signal + (noise_re + 1j * noise_im)
```

→ **實部與虛部各自** sd = `thermal_noise_jy`（單位 Jy）。
接著寫進 metadata：`"thermal_noise_jy": thermal_noise_jy`（第 518 行）。

**likelihood**（`VisibilityLikelihood.loglike`，`visibility.py:131-133`）：

```python
sigma_vis = float(meta.get("thermal_noise_jy", context.get("thermal_noise_jy", 0.05)))
sigma_arr = np.full(len(obs_vis), sigma_vis)
if "sigma" in meta:
    sigma_arr = np.asarray(meta["sigma"], dtype=float)
```

→ 套用在**振幅** `|V|` 上（第 191/193 行 `gaussian_loglike(obs_amp, model_amp, sigma_arr)`）。

## X.23.2 逐項核對

| 核對項 | 結果 |
|---|---|
| **是不是同一個數值路徑** | **是。** 全樹只有三處出現 `thermal_noise_jy`：模擬器讀 context、模擬器寫 metadata、likelihood 讀 metadata（metadata-first）。**likelihood 讀的是模擬器記錄下來的那個值本身**，不是獨立重新推導的——不存在「兩個各自定義、可能漂移」的數字。 |
| **同一個物理定義** | **是。** 兩側都是「每條基線的熱雜訊」。 |
| **同一組單位** | **是。** 兩側都是 Jy。 |
| **數值** | 兩側都是 `configs/instruments/eht.yaml` 的 `imaging.thermal_noise_jy: 0.05`，經 `dataio.eht.eht_imaging_config()` 進 context。 |
| **有沒有漏乘/多乘 √2 或 2** | **沒有。** 見 X.23.3 的數值驗證：高 SNR 下 `sd(\|V\|) / sigma_lik = 0.999–1.001`。若有 √2 錯誤，這個比值會落在 0.707 或 1.414。 |
| **熱雜訊 vs 系統雜訊的疊加** | **兩側都沒有系統雜訊項。** 全樹沒有 gain error / systematic 之類的第二個雜訊成分，所以不存在「一邊疊加了、一邊沒有」。 |
| **積分時間 / 頻寬假設** | **兩側都沒有用到。** `thermal_noise_jy` 是設定檔裡手設的常數，不由 SEFD / 頻寬 / 積分時間推導（見 X.23.5 的備註）。因此兩側不可能對它有不同假設。 |
| **測站數相關的正規化** | **兩側都沒有。** 每條基線一個獨立的雜訊抽樣，likelihood 每條基線一項殘差，`n = min(len(obs), len(model))` 實測是 16 = 16，沒有靜默丟資料。 |
| **另一個資料產生器** | `EHTLoader._mock_eht_data`（`dataio/eht.py:247`）用 `sigma = rng.uniform(0.02, 0.1, n)` 並以 `visibilities += sigma * (n_re + 1j*n_im)` 施加，**與模擬器同一個 per-component 慣例**，且透過 metadata 的 `"sigma"` 鍵被 likelihood 讀走。兩條資料路徑的慣例一致。SBC campaign 走的是 `ImageShadowSimulator`，它不設 `"sigma"`，所以用的是純量。 |

## X.23.3 數值驗證：慣例是對的，沒有 √2

模擬器給複數雜訊 `n = n_re + i·n_im`，兩個分量各自 sd = σ。
雜訊在訊號方向上的投影 `Re(n e^{-iφ}) = n_re cosφ + n_im sinφ`
的 sd 是 `σ·√(cos²+sin²) = σ`。
所以**高 SNR 下 `|V|` 的 sd 正好是 σ**——likelihood 把同一個 σ 用在振幅上是對的。

蒙地卡羅驗證（每筆 60000 次雜訊實現，用模擬器自己的雜訊配方）：

| 逐基線 SNR `|V|/σ` | `sd(|V|) / sigma_lik` |
|---|---|
| 109 | 0.9992 |
| 70 | 0.9996 |
| 24 | 0.9993 |
| 4.7 | 0.9857 |
| 1.6 | 0.8672 |
| 1.0 | 0.7810 |
| 0（Rayleigh 極限） | 0.655 |

高 SNR 端收斂到 1.000。**√2 或 2 的因子錯誤被明確排除。**

## X.23.4 唯一存在的差異：Rician 變異數虧損，而且**方向相反**

`|V_obs|` 服從 Rician 分布，其變異數**恆 ≤ σ²**，從高 SNR 的 σ²
單調降到零訊號極限的 `(2 − π/2)σ² = 0.429σ²`。
因此 **likelihood 用的 σ 永遠是高估，不是低估**。

在 X.21/X.22 pilot 的同一批 18 筆上（等權合併各基線）：

| | `r_eff = sd_true / sigma_lik` |
|---|---|
| 18 筆中位數 | **0.9612** |
| 最暗的一筆（網路 SNR 3.8） | 0.7605 |
| 最亮的一筆（網路 SNR 397.8） | 0.9993 |

在線性高斯近似下，區間半寬 ∝ 假設的 σ、真實誤差 ∝ 真正的 σ，
所以這個虧損讓區間**偏寬** `1/r_eff`：中位數 **1.040（+4.0%）**，
最暗的一筆到 1.31。

**這個方向不能解釋 under-coverage，只會造成 over-coverage。**
換句話說：實際的殘留缺口**比量到的還大一點**，因為 σ 的高估
正在部分遮蔽它。

用 campaign 真正的雜訊實現直接核對（18 筆 × 16 基線 = 288 個標準化殘差，
就是 `loglike` 實際除下去的那個量）：

```
mean z = +0.2120   sd z = 1.0399   (正確的高斯應為 0, 1)
```

`sd z` 與 1 相容（n = 288 時 sd 的標準誤約 0.042），
**沒有 17% 量級的尺度不匹配**。
`mean z = +0.21` 是已知的 Rician 均值偏移，即 X.20.5 的 (c)。

## X.23.5 一併記錄的 provenance 觀察（不是兩側的不匹配）

`configs/instruments/eht.yaml` 同時載有 `bandwidth_ghz: 2.0` 與每個測站的
`sefd_jy`，但 `thermal_noise_jy: 0.05` **不是**由它們推導出來的，是手設值。
用輻射計公式 `σ = √(SEFD₁·SEFD₂ / (2·Δν·t_int))` 對 8 站 28 條基線核算：

| `t_int` | 最小 | 中位數 | 最大 |
|---|---|---|---|
| 10 s | 0.0020 Jy | 0.0264 Jy | 0.0600 Jy |
| 300 s | 0.0004 Jy | 0.0048 Jy | 0.0110 Jy |

0.05 Jy 落在 10 s 積分的**悲觀端**，量級合理，但真實 EHT 的逐基線 σ
是高度不均勻的（ALMA 基線比 SMTO–JCMT 好約 30 倍），而這裡用的是均勻值。
**這是一個簡化，不是兩側的不一致**——模擬器與 likelihood 用的是同一個
簡化後的數字。記錄在此，是否要改由你決定。

## X.23.6 結論：**排除**

**模擬器與 likelihood 的 sigma 是同一個數字，走同一條路徑，
同一個物理定義、同一組單位，沒有任何因子差異。**
唯一存在的變異數差異是 Rician 虧損，它的**符號與需求相反**
（讓區間偏寬 +4.0% 中位數，而非偏窄），量級也遠小於缺口所需的 +11.7%（X.25.6 更正）。

**「likelihood 假設的 sigma 太小導致六參數均勻偏窄」這個候選被排除。**
依指示，本輪未修改任何程式碼，也未往下猜新的候選方向。

## X.23.7 產出

- `tests/test_image_noise_sigma_consistency.py`（6 個測試，釘住 per-component
  慣例、高 SNR 下無 √2、Rician 虧損的**符號**，以及 `loglike` 確實用了
  記錄下來的 sigma 而不是讀了卻不用）

---

# X.24 `gr_eternal` dlogz 對 coverage 影響的定向 pilot

排除 (b) `nact`、`nlive`（X.21 / X.22）與 sigma 不匹配（X.23）之後，
`dlogz` 是最後一個還沒測過、且理論上會造成**全局均勻**效應的取樣器旋鈕：
它決定 nested sampling 何時判定剩餘證據已夠小而停止，門檻偏鬆會讓後驗尾端
在充分探索之前就被截斷，所有參數同等比例偏窄——樣態與觀察到的殘留吻合。

**與 X.21 / X.22 同樣是定向 pilot，沒有碰 (c)，沒有改動任何 likelihood 程式碼。**

## X.24.1 設計與 `dlogz` 取值的依據

用**同一批 18 筆**分層樣本、同一組 seed 重建真值與雜訊，
`nact = 2`、`nlive = 250` 維持不變，**只改 `dlogz`**：0.1 → **0.01**。

取 0.01 的兩個理由：

1. **相對於統計誤差的位置。** base 的 `ln Z` 誤差中位數是 **0.218**，
   而 `dlogz = 0.1` 與它同量級——也就是說被截斷掉的尾端證據，原則上
   不能先驗地當成可忽略。收到 0.01 讓被忽略的剩餘量比統計誤差低一個數量級，
   truncation 確定不再是限制因素。
2. **成本可預期且負擔得起。** 剩餘證據大致以 `exp(−i / nlive)` 衰減，
   所以收緊 10 倍需要約 `nlive × ln(10) = 250 × 2.303 ≈ 576` 次額外迭代，
   在 base 的約 1900–5000 次迭代上是可接受的增量。

## X.24.2 行為驗證：三個可預測的量全部命中

| 量 | 理論預期 | 實測（18 筆） |
|---|---|---|
| **迭代數絕對增量** | **+576**（`nlive × ln 10`） | **中位數 +620、平均 +598（sd 81）** |
| `ln Z` 誤差比 | **< 1**（收斂更精確） | **中位數 0.845** |
| `ln Z` 點估計位移 | **≈ 0**（估得更精確，不是估得不同） | 平均 **−0.026**，距離 0 只有 **−0.35σ**，Wilcoxon **p = 0.640** |
| information gain `H` 比 | ×1.00（與取樣器無關） | 1.0022 |

**迭代數的絕對增量是這裡最鋒利的檢定**：它不是一個「有沒有變多」的定性
觀察，而是一個有數值預測的量，實測 +598 ± 81 對上預測 +576。
單筆最早的驗證跑（`0098`）是 **+579 對上 +576**。
`dlogz` 生效無疑義，且 `ln Z` 的點估計確實沒有系統性偏移。

## X.24.3 成本：18/18 收斂，ncall 1.38×、耗時 1.20×

| | `base`（dlogz=0.1） | `dlogz001` | 比例 |
|---|---|---|---|
| 收斂 | 18/18 | **18/18** | — |
| ncall 中位數 | 6.93×10⁴ | 9.61×10⁴ | **1.38×**（平均 1.39×） |
| 耗時中位數 | 185 s | 228 s | **1.20×**（平均 1.31×） |
| 18 筆總耗時 | 1.29 h | **1.49 h** | 1.16× |

**三個 pilot 裡最便宜的一個。** 與 X.22 相同的但書：成本以 ncall 為準，
`0097` 的耗時比 2.10× 而 ncall 比只有 1.08×，是機器負載的產物。
沒有任何一筆被 6000 s 的 `_BudgetGuard` 截斷。

## X.24.4 主結果：後驗區間寬度沒有變寬，點估計同樣略窄

108 個（注入 × 參數）配對的 90% 區間寬度比：

| | 值 |
|---|---|
| `w90(dlogz001) / w90(base)` 中位數 | **0.9913** |
| 同上平均 | **0.9826**，95% CI **0.959 – 1.006** |
| Wilcoxon signed-rank（全部 108 對的 log 比） | **p = 0.089** |

逐參數中位數比：`M` 0.979、`a_star` 0.994、`i` 0.991、`position_angle` 0.964、
`ring_width_frac` 1.021、`log10_total_flux_jy` 0.976，
Wilcoxon p = 0.108–0.640，**沒有一個達到顯著**。

**收緊 `dlogz` 十倍，後驗寬度的點估計是 −1.7%，95% 信賴上界是 +0.6%。
方向與「後驗偏窄需要變寬」相反，而且上界比前兩個 pilot 更緊。**

## X.24.5 coverage：同樣沒有實質改善

| | `base`（18 筆） | `dlogz001` | 理論 |
|---|---|---|---|
| 90% coverage 平均 | 0.806 | 0.824 | 0.891 |
| 68% coverage 平均 | 0.574 | **0.556** | 0.673 |

配對翻面數：**90% 只有 4/108**（miss→hit 3、hit→miss 1，McNemar p = 0.625）；
**68% 是 8/108**（3 上 5 下，p = 0.727）且**往下走**。
兩個信賴水準方向相反、翻面數都是個位數，與 X.21 / X.22 是同一種情形：
**兩組幾乎是同一個後驗。**

## X.24.6 效應量對上缺口：三個取樣器 pilot 並排

沿用 X.21.6 的算法：補上 0.839 → 0.891 需要 **+11.7%**（X.25.6 更正，原寫 +17.3%）。

| pilot | 旋鈕 | 量到的寬度變化 | 95% CI | 成本（ncall） |
|---|---|---|---|---|
| X.21 | `nact` 2 → 8（鏈長 ×4） | 0.0% | −2.5% … +2.4% | 3.00× |
| X.22 | `nlive` 250 → 500 | −1.1% | −4.1% … +2.0% | 2.00× |
| **X.24** | **`dlogz` 0.1 → 0.01** | **−1.7%** | **−4.1% … +0.6%** | **1.38×** |
| **需求** | — | **+11.7%**（X.25.6 更正） | — | — |

**三個旋鈕的 95% 上界合起來是 +0.6% 到 +2.4%，全部遠低於 +11.7%。**

一個必須誠實指出、但**不應過度解讀**的觀察：三個 pilot 的點估計都略低於 1
（0.999 / 0.989 / 0.983），也就是「把任何一個取樣器設定收緊，寬度都是微微
往下、從不往上」。但三者**單獨都不顯著**（p = 0.52 / 0.12 / 0.089），
而且**共用同一個 `base` 對照組**，所以三個數字不是獨立的證據，
不能把它們當成三次獨立確認。

## X.24.7 必須一併說明的限制

1. **n = 18，與 X.21 / X.22 相同。** 結論建立在配對寬度比，不是 coverage
   點估計；「90% coverage 0.806 → 0.824」只有 4 個指標的位移。
2. **只測了 `dlogz` 0.1 → 0.01 這一個對照。** 沒測 0.001。不過
   `ln Z` 點估計已經證實沒有系統性偏移（−0.35σ），代表 0.1 的截斷
   本來就沒有造成可量測的證據偏差；再收十倍要突然改變後驗形狀，
   需要一個機制上的理由。**這是推論，不是量測。**
3. **三個 pilot 共用同一個 `base` 對照組**，彼此不獨立。
4. **成本以 ncall 為準**，耗時比受機器負載影響（`0097`）。

## X.24.8 一句話結論

**把 `dlogz` 從 0.1 收緊到 0.01（迭代數增加 +598 ± 81 對上預測的 +576、
`ln Z` 誤差 ×0.845、`ln Z` 點估計位移只有 −0.35σ，行為驗證無疑義）
對後驗寬度的效應是 −1.7%（95% CI −4.1%…+0.6%，Wilcoxon p = 0.089，
方向與需求相反），而補上 0.05 的 coverage 缺口需要 +11.7%（X.25.6 更正）；
`dlogz` 因此與 `nact`、`nlive` 一樣被排除。**

由於效應量被排除，**不提出完整規模 campaign 的建議**
（僅記錄外推值供參考：N = 100 在 `dlogz = 0.01` 下約 **9.8–11.3 小時**，
由 8.15 h 乘上 1.20–1.39 倍——這是三個旋鈕裡最便宜的，但買不到東西）。

## X.24.9 候選清單真正用盡

| 候選 | 狀態 | 能解釋「六參數均勻偏窄」嗎 |
|---|---|---|
| (a) 模型樣板誤用含雜訊資料 | **已修正**（X.19/X.20） | 部分——0.800 → 0.839，沒補完 |
| (b) `nact` 鏈長不足 | **已排除**（X.21） | 否 |
| `nlive` 不足 | **已排除**（X.22） | 否 |
| likelihood 的 sigma 與模擬器不符 | **已排除**（X.23，兩側同一個數字） | 否 |
| **`dlogz` 門檻偏鬆** | **已排除**（X.24） | **否** |
| (c) Rician 振幅上用高斯 likelihood | 均值已核對、變異數已核對 | **否**——均值那一半預測流量單一參數的偏移；變異數那一半方向相反 |

**目前已知的三個取樣器精度旋鈕（`nact`、`nlive`、`dlogz`）全部被排除，
而且都是以「取樣已收斂」的方式排除的**：每一個旋鈕的行為驗證量
（`niter`、`ln Z` 誤差與點估計、`H`）都精確照 nested sampling 的理論走，
而後驗形狀在 4× 鏈長、2× 活點、10× 收斂門檻下都不動。

**「六參數均勻 coverage 過窄」現在沒有任何候選解釋。
依指示，本輪不往下猜新的機制。**

## X.24.10 產出

- `artifacts/image_sbc_dlogz001/M87star/`（18 筆，18/18 收斂）
- `docs/calibration/image_gr_eternal_dlogz_pilot.csv`
- `scripts/run_image_sbc.py` 的 `--dlogz`

---

# X.25 凍結參數乾淨對照組診斷

排除 (a)(b)(c) 與三個取樣器精度旋鈕之後，只剩一個方向沒測：
殘留的六參數均勻 coverage 過窄，究竟出在 likelihood / 模型物理層面，
還是出在 **SBC / coverage 計算機制本身**（重採樣、分位數、rank 正規化）。

bounce 通道有 `D_L` 這個天然的乾淨對照——它完全不進入那條 likelihood，
所以它的後驗必須精確等於先驗，成了整條 SBC 工具鏈的免費檢查。
`gr_eternal` 六個參數全是活的，沒有天然對照，**所以人工構造一個**。

**本節沒有改動任何 production likelihood / model 程式碼**，
診斷版 likelihood 以子類別的形式放在 harness 裡。

## X.25.1 構造方式

`scripts/run_image_sbc.py::_FrozenParameterLikelihood` 繼承
`VisibilityLikelihood`，每次 `loglike` 前把 `position_angle` 換成常數
**π/2 = 1.5707963**（先驗 `Uniform(0, π)` 的中點）：

```python
def loglike(self, theta, data, context):
    return super().loglike({**theta, **self.frozen}, data, context)
```

`parameter_names` **不動**，所以 `effective_parameter_names` 仍然回傳
`position_angle`，它**仍然被取樣**——這正是重點：
它對 lnL 毫無影響，因此邊際後驗必須精確等於先驗、SBC rank 必須精確均勻。

選 `position_angle` 的理由：它在原 campaign 裡資訊量很高
（見下節，逐筆 lnL 變化最高到 3553 nats），所以「本來很有資訊、
凍結後完全沒有資訊」這個對比最乾淨。

其餘設定與 X.21 / X.22 / X.24 完全相同：同一批 18 筆分層樣本、
同一組 seed 重建真值與雜訊、已修正的 (a) likelihood、
`nact = 2`、`nlive = 250`、`dlogz = 0.1`。

## X.25.2 凍結確實生效（逐筆驗證，不是彙總統計）

對每一筆注入，把 `position_angle` 在先驗範圍內掃 9 個點：

| 檢查 | 結果（18 筆逐筆） |
|---|---|
| 凍結版 lnL 的變化幅度 | **全部 18 筆都是 exactly 0.000e+00**（逐位元相同） |
| 未凍結版 lnL 的變化幅度 | **0.53 – 3553 nats**（確認這個參數本來很有資訊） |
| 模擬器 `vis_signal` 是否仍隨 PA 改變 | **是**，振幅最大差 7.4×10⁻⁴ – 2.4×10⁻¹ Jy |

第三列是關鍵：**凍結只影響 likelihood 端的模型建構，
生成注入資料的那條路徑完全不受影響**——注入仍然帶有真實的 PA 資訊，
只是 likelihood 看不見它。

## X.25.3 主結果：凍結參數的 rank 與 coverage 正常

18/18 收斂，總耗時 0.82 h（base 是 1.29 h——少一個有效參數，便宜 36%）。

| 參數 | 68% cov | 90% cov | rank 平均 | rank sd | KS p |
|---|---|---|---|---|---|
| **`position_angle`（凍結）** | **0.778** | **0.944** | **47.0** | **27.5** | **0.4462** |
| `M` | 0.500 | 0.667 | 69.1 | 29.4 | 0.0240 |
| `a_star` | 0.500 | 0.778 | 31.9 | 29.2 | 0.0230 |
| `i` | 0.833 | 0.889 | 45.3 | 22.5 | 0.1639 |
| `ring_width_frac` | 0.500 | 0.722 | 48.2 | 34.1 | 0.8236 |
| `log10_total_flux_jy` | 0.611 | 0.889 | 42.5 | 31.2 | 0.4392 |
| **參考值** | **0.667** | **0.883** | **50.0** | **29.2** | — |

凍結參數的 18 個 rank：
`3, 8, 17, 18, 19, 26, 39, 44, 48, 56, 56, 60, 60, 62, 64, 80, 92, 94`。

- **68% coverage 14/18 = 0.778**（參考 0.667，二項檢定 p = 0.455）
- **90% coverage 17/18 = 0.944**（參考 0.883，二項檢定 p = 0.713）
- **兩者都在參考值之上，沒有偏窄。**
- 90% 區間寬度中位數 **2.702 rad**，接近整個先驗（`0.90 × π = 2.827`），
  而未凍結時是 1.240 rad——**凍結讓它變寬 2.2 倍，後驗確實塌回先驗。**

**其餘五個參數的數字一併記錄，但不可與主 campaign 比較**：
凍結後模型對它們是**刻意錯誤指定**的（資料帶著真實 PA，模型一律用 π/2），
所以 `M`、`a_star` 的 KS p 掉到 0.024 是這個錯誤指定的後果，不是新發現。

## X.25.4 但 18 筆的對照組本身檢定力不足，所以又做了直接檢定

必須誠實指出：**如果計算機制真的讓區間普遍偏窄，18 筆能不能看出來？**

| 名目 | 參考值 | 若機制讓區間偏窄到需要 +17.3% 才補得回 | 實測 | 在該假說下的 p |
|---|---|---|---|---|
| 90% | 0.891 | 0.828 | 17/18 = 0.944 | **0.344** |
| 68% | 0.673 | 0.597 | 14/18 = 0.778 | **0.151** |

**p = 0.34 / 0.15 並不能否定那個假說。** 凍結對照組「通過」是好消息，
但它單獨不足以把計算機制排除。

所以直接檢定那兩個函式本身：從先驗抽 `L = 100` 個樣本當「完美後驗」、
另抽一個當真值，跑 `compute_credible_interval` 與 `compute_sbc_rank`，
**N = 200000 次**（SE ≈ 0.0007）：

| 先驗 | 68% coverage | 90% coverage | 極端 rank 比例 |
|---|---|---|---|
| uniform | 0.6674 | 0.8826 | 0.1185 |
| normal | 0.6667 | 0.8825 | 0.1187 |
| lognormal | 0.6672 | 0.8823 | 0.1189 |
| exponential | 0.6658 | 0.8825 | 0.1187 |
| **本專案一直用的參考值** | **0.6733** | **0.8911** | **0.1089** |

結果與先驗形狀無關（四個分布一致到 0.0002 以內）。

## X.25.5 這裡發現了兩個參考值上的錯誤（在分析／文件，不在程式碼）

**(1) `L = 100` 的分位數估計本身有小幅負偏。**
`compute_credible_interval` 用 `np.quantile` 的線性內插估分位數；
有限樣本下，完美後驗被這樣量測出來的 coverage 是
**0.8825（90%）與 0.6671（68%）**，不是專案一直當成理論值的
`level × L/(L+1)` = 0.8911 / 0.6733。差 **−0.0085 / −0.0062**，
在 N = 200000 下是 12σ / 6σ，是真實而可重現的效應。

**這佔掉了原本認定缺口的 16%（90%）與 10%（68%）。**

| 名目 | 實測（X.20 全 95 筆） | 舊參考值 → 缺口 | MC 參考值 → 缺口 |
|---|---|---|---|
| 90% | 0.839 | 0.8911 → +0.0521 | **0.8825 → +0.0436** |
| 68% | 0.614 | 0.6733 → +0.0593 | **0.6671 → +0.0534** |

**(2) 極端 rank 的參考值是 off-by-one。**
`compute_sbc_rank` 回傳 `#{samples < truth}`，值域是 `0…100`，共 **101** 個。
「≤5 或 ≥95」涵蓋 `0,1,2,3,4,5`（6 個）與 `95,…,100`（6 個），
合計 **12** 個，所以參考值是 **12/101 = 0.1188**，
不是文件裡寫的 `11/101 = 0.109`。MC 量到 0.1185–0.1189，吻合 12/101。

影響：X.18 / X.20 報告的「極端 rank 超標」是對著一個偏小 9% 的參考值算的。
**兩個都是參考值的錯誤，不是 `math_utils.py` 的 bug**——
函式本身的行為與它的 docstring 一致。

**（順帶記錄，未處理）** `compute_credible_interval` 的 docstring 寫
"Return the highest posterior density credible interval"，
但實作是等尾分位數（`alpha` 與 `1-alpha`）。對稱單峰時兩者重合，
偏斜時不同。等尾區間本身是合法的，SBC coverage 對它同樣成立，
所以這不影響任何結論，但**名稱與實作不符**，記錄在此。

## X.25.6 我自己先前算錯的一個數字，一併更正

X.21 / X.22 / X.24 都引用「補上缺口需要 **+17.3%** 的區間增寬」。
**那個數字算錯了**：我用名目的 0.90 去取 `z = Φ⁻¹(0.95) = 1.645`，
卻拿 0.891 當比較基準——兩者不一致。一致地算應該是：

| 基準 | 需要的增寬 |
|---|---|
| 舊參考值 0.8911 | **+14.4%** |
| MC 參考值 0.8825 | **+11.7%** |

**三個取樣器 pilot 的結論完全不變**：量到的效應是
0.0% / −1.1% / −1.7%，95% 上界是 +2.4% / +2.0% / +0.6%，
對上 +11.7% 仍然是決定性的排除。但數字本身要更正為 **+11.7%**。

## X.25.7 結論

**凍結參數對照組通過，而且直接檢定證實 SBC / coverage 計算機制
基本上是健全的**：rank 統計精確（平均 50.09、sd 29.139 對上理論
50.00、29.155），分位數估計只有一個 −0.0085 的小幅有限樣本負偏。

**但那個負偏是真的，而且佔掉原本認定缺口的 16%。**
扣掉之後，殘留仍有 **+0.0436**（90%）與 **+0.0534**（68%），
對應需要 **+11.7%** 的區間增寬——**絕大部分的缺口沒有被計算機制解釋掉**。

依 X.25 的設計，這指向第一個分支：
**bug 還在 likelihood，或在六個參數彼此互動的某個環節，
需要回頭重新檢視 likelihood 程式碼本身。**

**依指示，本輪不往下動手修任何一邊**——不修 likelihood，
也不修那兩個參考值與那個 docstring，先回報。

## X.25.8 產出

- `artifacts/image_sbc_frozenPA/M87star/`（18 筆，18/18 收斂）
- `docs/calibration/image_gr_eternal_frozen_pa_pilot.csv`
  （逐筆 rank / `in68` / `in90` / `w90`，凍結組與 base 並排）
- `scripts/run_image_sbc.py` 的 `--freeze` 與 `_FrozenParameterLikelihood`

---

# X.26 候選清單更新：`n_pixels` 結構性排除、新增 `f_pix` 階梯效應候選

**本節沒有跑任何運算，也沒有改動任何程式碼**，是候選清單的狀態更新。

## X.26.1 `n_pixels`：假說前提不成立，結構性排除

**原假說**：`build_ring_image()` 在有限像素網格上算模型影像，參數連續變化時
影像的像素化表示帶有系統性量化誤差（尤其環邊緣、特徵尺度接近像素大小處），
這個誤差沒有被 likelihood 假設的高斯雜訊模型（`thermal_noise_jy`）
考慮進去，導致 likelihood 系統性高估模型的精確度，後驗因此系統性過窄。
這會是一個影響整張影像的全局效應，不偏向任何單一參數，
吻合觀察到的六參數大致均勻偏窄樣態。

**排除理由**：likelihood 的模型樣板與生成資料用的無雜訊訊號，
在真值處呼叫的是**同一個 `build_ring_image()`、用同一個網格，
逐位元完全相同**：

```
model template vs data's noise-free signal at the TRUE theta:
  max |difference| = 0.0
  bit-identical    = True
  images identical = True
```

這代表 forward model `f` 的定義本身就是

> 資料 = `f_pixelated(θ_true)` + noise

而**不是** `f_continuous(θ_true)` + noise。**真值處沒有任何未被建模的
量化殘差。** 原假說的機制在這個結構下不成立，因為不存在
「模型與真實的不匹配」這種落差可言。

**因此沒有跑任何解析度測試。** 當時做的成本估算一併記錄，供日後參考
（同一筆注入、12 次取平均、FoV 固定 50 μas）：

| `n_pixels` | 影像陣列形狀 | 每次 `loglike` | 倍率 | 18 筆 pilot 推算 |
|---|---|---|---|---|
| 128（現值） | (128, 128) | 1.86 ms | 1.00× | 1.29 h（實測） |
| 256 | (256, 256) | 6.89 ms | 3.70× | 4.8 h |
| 512 | (512, 512) | 30.06 ms | **16.13×** | **20.8 h** |

倍率接近像素數平方，與 `build_ring_image` ＋直接 DFT 的複雜度相符
（`loglike` 目前還會建構影像兩次——X.19.5 記錄的已知重複，未修——
所以倍率是實打實的）。

**排除方式與 X.21 / X.22 / X.24 不同，稽核時請按這個差別讀**：
那三項是**配對實測**排除的，各自有 18 筆、量到的效應量與 95% 信賴區間；
這一項是**結構性論證**排除的，**沒有量到的效應量**。

## X.26.2 新候選：`f_pix(θ)` 階梯狀跳動可能扭曲曲面形狀

X.26.1 排除的是「量化誤差是模型與真實的落差」。**較弱的版本仍然成立**，
而且是**不同的機制**：`f_pix(θ)` 隨 θ 連續變化時是**階梯狀**的，
這可能**扭曲真值附近 likelihood 曲面的形狀**，
而不是製造一個未被建模的量化誤差。

**為什麼先前沒有任何測試排除它**：X.25 的凍結參數對照組只切斷**單一**
參數對 lnL 的影響，其他五個參數在連續變動時仍然會撞到像素量化階梯，
所以那個對照組測不到這個效應。三個取樣器旋鈕測的是取樣精度，也無關。

**建議的初步驗證方式（比完整 SBC pilot 便宜得多；本輪依指示未執行）**：
直接對 `build_ring_image()` 在幾組真值附近、沿各參數方向做**細密的數值掃描**
——不跑任何取樣器，只評估 lnL（或影像本身）隨參數的變化是平滑的，
還是有非連續的階梯跳動。判讀：

- **明顯的非平滑跳動** → 再決定是否值得投入完整 pilot 驗證它對 coverage
  的實際影響。
- **平滑（跳動遠小於雜訊尺度 `thermal_noise_jy`）** → 可以直接排除，
  **不用跑完整 pilot**。

**狀態：未測試，已識別但尚未驗證。** 這是目前候選清單上唯一還開著的項目。

## X.26.3 候選清單現況

| 候選 | 狀態 | 排除方式 |
|---|---|---|
| (a) 模型樣板誤用含雜訊資料 | **已修正**，但沒補完（0.800 → 0.839） | 實測（X.19 / X.20） |
| (b) `nact` 鏈長不足 | **已排除**（0.0%，CI −2.5%…+2.4%） | 配對實測（X.21） |
| `nlive` 不足 | **已排除**（−1.1%，CI −4.1%…+2.0%） | 配對實測（X.22） |
| `dlogz` 門檻偏鬆 | **已排除**（−1.7%，CI −4.1%…+0.6%） | 配對實測（X.24） |
| likelihood 的 sigma 與模擬器不符 | **已排除**（兩側同一個數字） | 程式碼閱讀＋數值核對（X.23） |
| (c) Rician 振幅上用高斯 likelihood | 均值與變異數都已核對 | 解釋流量單一參數的偏移，變異數那一半方向相反（X.20.5 / X.23.4） |
| SBC / coverage 計算機制本身 | **大致排除**，只解釋 16% | 凍結參數對照組＋N=200000 直接檢定（X.25） |
| `n_pixels` 量化誤差未進雜訊模型 | **已排除** | **結構性論證，非實測**（X.26.1） |
| **`f_pix(θ)` 階梯狀跳動扭曲曲面形狀** | **未測試** | — （X.26.2） |

缺口基準：**+11.7%**（X.25.6 更正後的值；更正前寫作 +17.3%）。

---

# X.27 `f_pix(θ)` 階梯效應數值掃描

驗 X.26.2 登記的候選 I-6k：`build_ring_image()` 在有限像素網格上算影像，
`f_pix(θ)` 隨 θ 連續變化時是否**階梯狀跳動**，進而扭曲真值附近
likelihood 曲面的形狀。**沒有跑任何取樣器，沒有改動任何 production 程式碼。**

## X.27.1 方法：網格細化測試，不是看單一次掃描

**只看一次掃描說「很粗糙」無法分辨兩件事**：真正的不連續，
與「平滑但很陡、取樣太疏」。image 通道的 lnL 在某些方向確實極陡
（見 X.27.4），所以這個分辨是必要的。把步長 `h` 減半就能分開：

| | `max|Δy|` 隨 h 減半 | `max|Δy| / h` |
|---|---|---|
| 平滑可微 `f` | **減半（比值 0.500）** | 收斂到 `|f'|max` |
| 真正的不連續 | **不變（比值 1.000）** | 每次細化加倍 |

對 5 筆注入（29 / 76 / 62 / 96 / 80，網路 SNR 3.8 – 397.8）
的每一個參數，在真值附近的視窗上跑 `N = 201 / 401 / 801 / 1601` 四個網格。

**掃描視窗必須裁切到先驗支撐。** 第一版沒有裁切，
`truth ± 1.5 × w90` 讓 `ring_width_frac` 跑到**負值**、
`i` 越過 edge-on，那裡 `build_ring_image` 退化、lnL 擺動數百 nats
（`96 / ring_width_frac` 一度量到 `max|d2| = 617.7`，甚至超過該次掃描的
lnL 全距）。**取樣器從不在先驗外取值，所以那些擺動什麼都不代表。**
裁切後 `ring_width_frac` 的數字掉到 0.0096–1.61。這是我第一版掃描的設計
錯誤，記錄在此以免日後誤讀。

## X.27.2 結果：30 個掃描裡 29 個乾淨平滑

| idx | SNR | 六個參數的細化比值 |
|---|---|---|
| 29 | 3.8 | 0.501 / 0.500 / **0.500** / 0.500 / 0.502 / 0.500 |
| 76 | 11.2 | 0.500 / 0.502 / **1.167** / 0.503 / 0.500 / 0.500 |
| 62 | 30.3 | 0.502 / 0.501 / **0.952** / 0.500 / 0.501 / 0.500 |
| 96 | 102.3 | 0.500 / 0.513 / **1.160** / 0.501 / 0.512 / 0.500 |
| 80 | 397.8 | 0.501 / 0.502 | **0.779** / 0.501 / 0.507 / 0.501 |

（欄序：`M` / `a_star` / `i` / `position_angle` / `ring_width_frac` /
`log10_total_flux_jy`；粗體是 `i`。）

**除了 `i` 之外的 25 個掃描，細化比值全部落在 0.500 – 0.513**，
也就是 `max|Δy| ∝ h`——標準的平滑可微行為，**沒有任何階梯結構**。

**影像側同樣乾淨**：相鄰掃描點之間總流量的最大變化，
以 `thermal_noise_jy = 0.05 Jy` 為單位是

| 參數方向 | `最大流量跳動 / σ` |
|---|---|
| `M` / `a_star` / `i` / `position_angle` / `ring_width_frac` | **1.1×10⁻¹⁵ – 1.1×10⁻¹³** |
| `log10_total_flux_jy` | 3.7×10⁻³ – 5.2×10⁻³ |

前五個方向比雜訊尺度低 **13–15 個數量級**（是浮點捨入層級，
因為 `build_ring_image` 把影像歸一化到總流量，流量本來就不隨幾何改變）；
`log10_total_flux_jy` 方向的 5×10⁻³ 是**流量本來就該隨這個參數變**的
連續變化，也比雜訊尺度低 200 倍，而且同樣隨 h 縮小。

## X.27.3 `i` 是唯一的例外，但細化後證明它也是平滑的

`i` 在 `±0.5 × w90` 的視窗上比值是 0.779 – 1.167，看起來像跳動。
但這是 X.27.1 講的那個陷阱的第二種形式：**`i` 的 `w90` 幾乎是整個先驗**
（資料對它約束很弱），所以「半個後驗寬度」仍是很大的範圍、`h` 仍然很粗。

把視窗縮到真值附近 `±0.05 rad` 重跑細化，四筆全部乾淨：

| | `N=101` | `201` | `401` | `801` | `1601` | 比值 |
|---|---|---|---|---|---|---|
| 96 `i` `max|Δy|` | 0.7412 | 0.3707 | 0.1854 | 0.0927 | 0.0464 | **0.500** |
| 80 `i` | 0.0581 | 0.0291 | 0.0146 | 0.0073 | 0.0037 | **0.500/0.501** |
| 62 `i` | 0.0065 | 0.0033 | 0.0016 | 0.0008 | 0.0004 | **0.501** |
| 96 `M`（對照） | 0.2690 | 0.1358 | 0.0682 | 0.0342 | 0.0171 | **0.501** |

再把最陡的那個特徵單獨挑出來細化——它落在 **`i ≈ 1.5825 rad = 90.7°`，
也就是 edge-on**，不是真值附近（該筆真值是 51.67°）：

| `N` | `h` | `max|Δy|` | `max|Δy| / h` | 比值 |
|---|---|---|---|---|
| 201 | 4.0×10⁻⁵ | 197.27 | 4.932×10⁶ | — |
| 401 | 2.0×10⁻⁵ | 100.70 | 5.035×10⁶ | 0.510 |
| 801 | 1.0×10⁻⁵ | 50.487 | 5.049×10⁶ | 0.501 |
| 1601 | 5.0×10⁻⁶ | 25.289 | 5.058×10⁶ | 0.501 |
| 3201 | 2.5×10⁻⁶ | 12.645 | 5.058×10⁶ | **0.500** |

`max|Δy|` 精確減半、`max|Δy|/h` 收斂到 **|f'| ≈ 5.06×10⁶ nats/rad**。
**這是平滑函數配上一個很大的導數，不是跳躍。**
該視窗內 `UnrepresentableRingError` 出現 **0/201** 次，
所以也不是可表示性邊界造成的。

`i ≈ 90°` 的陡峭是**物理的**：環在 edge-on 投影成一條線，
影像結構在那裡對 `i` 極度敏感。它與像素量化無關。

## X.27.4 量化：跳動對上雜訊尺度

| 量 | 值 |
|---|---|
| lnL 相鄰步階，最細網格，25 個非 `i` 掃描 | 6.6×10⁻⁵ – 0.159 nats |
| 同上，以「1σ ⇒ ΔlnL = 0.5」為單位 | **1.3×10⁻⁴ – 0.32** |
| **而且全部隨 h 線性縮小**（比值 0.500–0.513） | → `h → 0` 時歸零 |
| 影像總流量跳動 / `thermal_noise_jy` | **1.1×10⁻¹⁵ – 5.2×10⁻³** |

關鍵不是「這些數字小」，而是**它們隨網格細化而消失**。
真正的階梯跳動不會隨 `h` 縮小；這裡的每一個都會。

## X.27.5 一句話結論

**`f_pix(θ)` 在真值附近（以及先驗支撐內所有測到的地方）是平滑可微的：
30 個「注入 × 參數」掃描中，25 個的細化比值直接落在 0.500–0.513，
另外 5 個（全是 `i`）在縮到有界視窗後同樣收斂到 0.500，
最陡的特徵（edge-on，`|f'| ≈ 5.06×10⁶ nats/rad`）經五級細化確認是
平滑而非跳躍；影像側的流量跳動比雜訊尺度低 3–15 個數量級，
且同樣隨 h 線性消失——**沒有觀察到任何階梯跳動，這個候選排除，
不需要再跑完整 pilot。**

## X.27.6 候選清單到此全部處理完

| 候選 | 狀態 | 排除方式 |
|---|---|---|
| (a) 模型樣板誤用含雜訊資料 | **已修正**，但沒補完（0.800 → 0.839） | 實測（X.19 / X.20） |
| (b) `nact` 鏈長不足 | **已排除**（0.0%，CI −2.5%…+2.4%） | 配對實測（X.21） |
| `nlive` 不足 | **已排除**（−1.1%，CI −4.1%…+2.0%） | 配對實測（X.22） |
| `dlogz` 門檻偏鬆 | **已排除**（−1.7%，CI −4.1%…+0.6%） | 配對實測（X.24） |
| likelihood 的 sigma 與模擬器不符 | **已排除**（兩側同一個數字） | 程式碼閱讀＋數值核對（X.23） |
| (c) Rician 振幅上用高斯 likelihood | 均值與變異數都已核對 | 均值解釋流量單一參數的偏移；變異數方向相反（X.20.5 / X.23.4） |
| SBC / coverage 計算機制本身 | **大致排除**，只解釋 16% | 凍結參數對照組＋N=200000 直接檢定（X.25） |
| `n_pixels` 量化誤差未進雜訊模型 | **已排除** | 結構性論證，非實測（X.26.1） |
| **`f_pix(θ)` 階梯狀跳動** | **已排除** | **網格細化數值掃描（X.27）** |

**八個候選全部處理完畢，殘留的 +11.7% 缺口（六參數 90% coverage
平均 0.839，對上有限樣本修正後的參考值 0.8825）沒有任何候選能解釋。**

這代表需要重新思考整個調查方向。**依指示，本輪不往下猜新方向。**

## X.27.7 產出

- `docs/calibration/image_gr_eternal_fpix_scan.csv`
  （30 個掃描的視窗、四級網格的最大步階、細化比值、影像側流量與像素差異）
- `scripts/scan_fpix_steps.py`

---

# X.28 殘留 coverage 缺口統計顯著性正式檢定

八個候選機制全部處理完之後，補上一件**整個調查過程中從來沒有正式做過**的事：
「+11.7% 缺口」本身，在 N = 95 的樣本數下，是否達到統計顯著。
**純統計檢定，不涉及任何機制假說，沒有跑任何新的取樣。**
資料是 X.20 的存檔（100 筆起跑、95 筆收斂）。

## X.28.1 檢定設計：為什麼合併檢定要以「注入」為獨立單位

六個參數的 hit/miss 指標**不是互相獨立的**：同一筆注入的六個指標
來自同一組資料、同一個後驗。把 6 × 95 = 570 個指標直接丟進一個二項式
檢定會低估變異數、給出偏小的 p 值。

**獨立的單位是「注入」，不是「注入 × 參數」。**
所以合併檢定的做法是：對每一筆注入數出被涵蓋的參數個數
`k_j ∈ {0,…,6}`（95 個互相獨立的觀測），檢定其平均是否顯著低於
`6 × 參考值`，p 值用**對注入重抽的 bootstrap**（200000 次）算，
這樣不需要假設參數之間獨立。

實測的組內相關很小（ICC = +0.023 於 90%、−0.030 於 68%，
變異數膨脹 1.115 / 0.849），所以叢集效應本來就不大——
但這是**量出來的**，不是假設的。

參考值用 X.25 的有限樣本修正值 **0.8825（90%）/ 0.6671（68%）**，
不是 `level × L/(L+1)`。

## X.28.2 逐參數：沒有任何一個單獨顯著

| 參數 | hits | coverage | 單側 p | Bonferroni ×6 |
|---|---|---|---|---|
| `M` | 82/95 | 0.863 | 0.3231 | 1.0000 |
| `a_star` | 79/95 | 0.832 | 0.0880 | 0.5280 |
| `i` | 80/95 | 0.842 | 0.1443 | 0.8659 |
| `position_angle` | **76/95** | **0.800** | **0.0139** | 0.0836 |
| `ring_width_frac` | 80/95 | 0.842 | 0.1443 | 0.8659 |
| `log10_total_flux_jy` | 81/95 | 0.853 | 0.2226 | 1.0000 |

**六個參數逐一檢定，沒有一個在多重比較修正後達到顯著。**
最極端的 `position_angle` 原始 p = 0.0139，Bonferroni 後 0.0836。

這一點值得記下來：X.18 / X.20 一路上把逐參數的 coverage 偏低
當成需要解釋的現象，但**逐參數而言它們本來就都在雜訊範圍內**。

## X.28.3 合併檢定：缺口**確實顯著**

| | 90% | 68% |
|---|---|---|
| 每筆注入平均涵蓋參數數 | 5.0316 | 3.6842 |
| 期望（6 × 參考值） | 5.2950 | 4.0026 |
| 差 | **−0.2634** | **−0.3184** |
| 叢集穩健 SE | 0.0975 | 0.1123 |
| **z** | **−2.70** | **−2.84** |
| **bootstrap 單側 p** | **0.00346** | **0.00192** |

對照（都假設獨立，因此偏鬆）：naive 合併二項式 p = 0.00111 / 0.00443；
Fisher 合併六個 p 值 p = 0.00937 / 0.01223。**三種算法都落在顯著側。**

**兩個信賴水準各自獨立地給出同方向、同量級的結果**，這本身是
「不是雜訊」的佐證：如果只是抽樣起伏，沒有理由 90% 與 68% 同時
偏低同樣的幅度。

## X.28.4 效應量與它的不確定性

| 名目 | coverage | bootstrap 95% CI | 缺口 | **需要的區間增寬** | 其 95% CI |
|---|---|---|---|---|---|
| 90% | 0.8386 | [0.8070, 0.8702] | +0.0439 | **11.8%** | **[3.3%, 20.2%]** |
| 68% | 0.6140 | [0.5772, 0.6509] | +0.0531 | **11.7%** | **[3.4%, 20.8%]** |

**兩個信賴水準給出的所需增寬幾乎完全一致（11.8% 與 11.7%）**，
而它們是從不同的分位數算出來的。這與「全域、與參數無關的區間過窄」
的樣態一致。

**這同時讓前面三個取樣器 pilot 的排除更紮實**：它們的 95% 上界是
**+2.4% / +2.0% / +0.6%**，全部低於所需增寬 CI 的**下界 +3.3%**。
也就是說，即使缺口實際上落在它不確定性範圍內最有利的那一端，
三個旋鈕的效應量仍然不足。

## X.28.5 必須承認的一件事：先前的信心沒有統計基礎

X.25.7 寫的是「**絕大部分的缺口沒有被計算機制解釋掉**」。
那句話的依據**只有點估計的算術**：0.0521 的原始缺口裡，
計算機制解釋掉 0.0085，剩下 0.0436，所以「84% 沒被解釋」。

**那個算術是對的，但它完全沒有說到「剩下的部分能不能與雜訊區分」。**
當時並沒有做任何顯著性檢定就下了那個判斷。

現在補做的結果**恰好支持了那個判斷**（p = 0.0035），
但那個支持在當時並不存在。這是方法上的疏漏，記錄於此。
同樣地，X.18 / X.20 判讀逐參數 coverage 偏低時用的也是
「觀察值減理論值除以粗略標準誤」，而不是正式檢定——
X.28.2 顯示那樣的判讀**高估了逐參數證據的強度**。

## X.28.6 必須一併說明的限制

1. **分母有收斂選擇效應。** 95 筆是「收斂的」那些；被排除的 5 筆
   `timeout_capped` 的 SNR 是 **583.5 / 349.3 / 791.9 / 572.8 / 140.4**，
   **全部落在收斂集的上四分位（149.7）之上**。
   也就是說被排除的是最亮、最難取樣的樣本，收斂集因此偏向低 SNR。
   這個偏差的方向對 coverage 的影響未知，**本輪沒有檢定它**。
2. **參考值 0.8825 / 0.6671 本身是蒙地卡羅量的**（N = 200000，
   SE ≈ 0.0007），相對於 0.0439 的缺口可以忽略，但不是解析值。
3. **顯著不等於大。** 所需增寬的 95% CI 是 [3.3%, 20.2%]，
   相當寬；效應的**存在**有證據，效應的**大小**只被鬆散地決定。
4. 合併檢定假設 95 筆注入互相獨立。它們用不同的 seed 生成，
   這個假設應該成立，但沒有另外檢定。

## X.28.7 結論與對 I-6 這條調查線的建議

**缺口是統計顯著的**（合併 z = −2.70 / −2.84，bootstrap p = 0.0035 / 0.0019，
兩個信賴水準同方向），**但沒有任何單一參數單獨顯著**（Bonferroni 後全部不顯著）。
這正是「全域、均勻、小幅」的效應該有的樣子：
逐參數看不出來，合併起來才看得到。

**因此不能把這條線標記為「殘留差異在統計雜訊範圍內」而收尾。**

調查已經窮盡的類別：

| 類別 | 涵蓋的候選 |
|---|---|
| **likelihood 內部** | 模型樣板誤用含雜訊資料 (a)；雜訊 `sigma` 與模擬器的一致性；振幅機率模型 (c) 的**均值**與**變異數**兩半 |
| **取樣器精度** | `nact`（鏈長 ×4）、`nlive`（活點 ×2）、`dlogz`（門檻 ×10） |
| **計算機制** | credible interval 估計、rank 統計；凍結參數乾淨對照組＋N = 200000 直接檢定 |
| **成像／像素化** | `n_pixels` 量化殘差（結構性論證）、`f_pix(θ)` 階梯效應（網格細化掃描） |

**依指示，本輪不往下猜新的候選機制。**
要不要從一個全新的角度開一條調查線，或是先接受目前的狀態，由你決定。

## X.28.8 產出

- `docs/calibration/image_gr_eternal_coverage_significance.csv`
- `scripts/test_coverage_significance.py`
