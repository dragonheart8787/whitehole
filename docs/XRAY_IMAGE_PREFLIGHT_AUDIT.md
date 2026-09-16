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
