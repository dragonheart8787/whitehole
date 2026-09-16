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
