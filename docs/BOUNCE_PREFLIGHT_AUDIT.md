# `discrete_uniform` 先驗映射修正，與 `bounce` 模型校準前置稽核

## 範圍聲明

**這份文件記錄的是 (Part A) 一個先驗映射 bug 的修正，與 (Part B) 對 `bounce` 模型執行
SBC/coverage 之前的前置稽核結果。不構成任何白洞訊號偵測或未偵測的科學宣稱，也不包含任何
天文物理結論。**

WhiteSearch 是 candidate ranking engine（候選訊號排序引擎），不是白洞證明器。所有資料皆為
mock 模擬資料，未使用任何真實 GWOSC 觀測資料。

**文件狀態（2026-09-06 更新）**：Part A、Part B 是原始紀錄，內容不變——Part B 的
campaign 當時未執行，因為前置稽核發現了與 `docs/BH_RINGDOWN_SBC_COVERAGE_REPORT.md`
發現 (B)/(D) 同等級的阻斷性問題，依 fail-closed 原則先回報現況與修法選項。
決策後的四項修正（B.1／B.3-B3-3／B.4-B4-2／B.5）與修正後的 SBC/coverage 驗證結果見
**Part C**：六個取樣參數的 SBC rank 均勻性全部通過，`eps_Q` 有一項需要留意的訊號。
之後改採 B3-2（爆發時序重新參數化）的實作與其校準驗證見 **Part D**——
**該次驗證失敗**：8 個取樣參數全部嚴重不校準，如實回報，未進一步修正。

---

# Part A：`discrete_uniform` → `bilby.core.prior.DiscreteValues`

## A.1 問題

`ParameterSpec.sample()` 的 `discrete_uniform` 從給定的 `values` 均勻抽樣，但
`to_bilby_prior()` 回傳 `DeltaFunction(values[0])`——在 dynesty 下等於把
`bounce` 的 `p_lifetime`（`values=[4, 5]`）**釘死在 4，完全沒有被推論**。
這與 `BH_RINGDOWN_SBC_COVERAGE_REPORT.md` 發現 (A) 是同一個 bug 家族。

## A.2 更正上一輪報告的一個錯誤陳述

上一輪報告 §10.7 寫「bilby 2.8 的 `Categorical` 只涵蓋 `0..n-1`，對 `[4, 5]` 這種偏移集合
沒有等價類別」，**這是錯的**。當時只查了幾個猜到的類別名稱就斷言不存在，沒有列舉
`bilby.core.prior` 的內容。

實際列舉後（bilby 2.8.2）：

```
Categorical, ConditionalDeltaFunction, ConditionalInterped, DeltaFunction,
DiscreteValues, Interped, MultivariateGaussian, ..., WeightedCategorical,
WeightedDiscreteValues
```

`bilby.core.prior.DiscreteValues(values=[...])`（在 `bilby.core.prior.analytical`）語意
正好就是「在任意有限值集合上均勻」：

| 檢查 | 結果 |
|---|---|
| `DiscreteValues(values=[4,5])` 的 `minimum` / `maximum` | 4 / 5 |
| `prob(4)`, `prob(5)`, `prob(4.5)`, `prob(3)` | 0.5, 0.5, 0.0, 0.0 |
| `rescale([0, 0.25, 0.49, 0.51, 0.99, 1.0])` | `[4 4 4 5 5 5]` |
| 三值集合 `[2, 7, 11]` 抽 30000 次的頻率 | 0.333 / 0.336 / 0.331 |

## A.3 選了哪個做法、為什麼

任務列出的兩個方向分別是 (a) `Categorical` + index↔value 轉換層、(b) 自訂繼承
`bilby.core.prior.Prior` 的類別。**兩個都不需要**：bilby 內建的 `DiscreteValues` 已經直接
支援有偏移的任意離散集合。

依任務給的優先順序評估：

| 準則 | `DiscreteValues`（採用） | (a) Categorical + index 層 | (b) 自訂 prior 類別 |
|---|---|---|---|
| 改動範圍 | 只改 `to_bilby_prior()` 一個 case | 要在 model／likelihood 層加 index↔value 轉換，且 forward/backward 都要一致 | 要新增並自行維護一個 prior 類別（`rescale`/`prob`/`ln_prob`/`sample`） |
| 是否需要 index 轉換 | 不需要 | 需要 | 不需要 |
| 能否被既有一致性測試驗證 | 可以，直接比較經驗分布 | 可以，但測試要跨越轉換層 | 可以 |
| 正確性風險 | 用上游已測試過的實作 | 轉換層是新的錯誤來源（posterior 存的是 index 不是 value） | 自己實作 `rescale` 的邊界條件容易出錯 |

因此採用 `DiscreteValues`。對太舊、沒有這個類別的 bilby，改為 **fail-closed 拋
`ImportError`**，而不是退回一個沒人要求的分布。

## A.4 驗證

`tests/test_prior_mapping.py`：

- 移除上一輪的 `xfail(strict=True)` marker；`discrete_uniform` 現在走正式斷言。
- 主一致性測試對離散型改用**卡方檢定**（KS 對有限支撐不適用），並先斷言兩邊的**值集合相同**。
- 新增 `test_discrete_uniform_covers_arbitrary_offset_value_sets`，涵蓋
  `[4,5]`、`[2,7,11]`、`[-3,0,1,9]`——確認不是只對「兩個值」或「連續整數」剛好正確；
  同時斷言 `prob(v) == 1/len(values)`（讓 nested sampling 看到真的密度，不是點質量）。
- 新增 `test_discrete_uniform_is_not_a_point_mass`（直接鎖住 `DeltaFunction` 的 regression）。
- 新增 `test_empty_discrete_uniform_raises`。

`bounce` 的完整先驗映射現在是：

```
M LogUniform | a_star Uniform | log10_tau_bounce_yr Uniform | log10_ell_q Uniform
p_lifetime DiscreteValues | eps_f Uniform | eps_Q Uniform | log10_A_bounce Uniform
D_L PowerLaw | i Sine | eta_r LogUniform | eta_gamma LogUniform
```

---

# Part B：`bounce` 校準前置稽核 —— 發現阻斷性問題，campaign 未執行

方法與上一輪對 `bh_ringdown` 相同：對每一個宣告的參數做擾動，量測 (a) mock 模擬器輸出是否改變、
(b) likelihood 的 lnL 是否改變；再檢查先驗支撐是否落在 likelihood 可表示的範圍內。
context 用 `sample_rate=4096, duration=4.0, t_merger=1.0, low_freq_cutoff=20.0`。

## B.1 參數數量本身就對不上

| 來源 | 參數數 | 內容 |
|---|---|---|
| `BlackToWhiteBounce.parameter_names` | **12** | `M, a_star, log10_tau_bounce_yr, log10_ell_q, p_lifetime, eps_f, eps_Q, log10_A_bounce, D_L, i, eta_r, eta_gamma` |
| `GWLikelihood("bounce").parameter_names` | **8** | `M, a_star, eps_f, eps_Q, log10_A_bounce, log10_tau_bounce_yr, D_L, i` |
| 只在 model 宣告、likelihood 沒宣告 | 4 | `log10_ell_q, p_lifetime, eta_r, eta_gamma` |

（任務描述裡寫的 8 個是 likelihood 那一份；模型本身宣告 12 個。`BilbyRunner.run()` 用
`model.to_bilby_priors()` 建先驗，所以 **dynesty 實際取樣的是 12 維**。）

## B.2 逐參數活性實測（擾動一個參數，看資料與 lnL 有沒有變）

基準點 `M=60, a_star=0.6, log10_tau_bounce_yr=0, log10_ell_q=3, p_lifetime=4, eps_f=0.05,
eps_Q=-0.1, log10_A_bounce=-21, D_L=400, i=0.5, eta_r=1e-5, eta_gamma=1e-6`；
模擬器與 likelihood 都用同一個 seed 的資料。

| 參數 | `max\|d − d₀\|`（模擬器） | ΔlnL | 判定 |
|---|---|---|---|
| `M` | 3.08e-22 | +4.69e+02 | live |
| `a_star` | 1.32e-21 | −3.08e+03 | live |
| `log10_tau_bounce_yr` | **0.0** | **0.0** | **DEAD（兩邊都不用）** |
| `log10_ell_q` | **0.0** | **0.0** | **DEAD（兩邊都不用）** |
| `p_lifetime` | **0.0** | **0.0** | **DEAD（兩邊都不用）** |
| `eps_f` | 1.58e-21 | −5.38e+03 | live |
| `eps_Q` | 9.25e-22 | +1.76e+03 | live |
| `log10_A_bounce` | **0.0** | **0.0** | **DEAD（兩邊都不用）** |
| `D_L` | 8.05e-21 | −4.03e+04 | live |
| `i` | 5.31e-21 | −1.41e+04 | live |
| `eta_r` | **0.0** | **0.0** | **DEAD（兩邊都不用）** |
| `eta_gamma` | **0.0** | **0.0** | **DEAD（兩邊都不用）** |

**12 個參數裡有 6 個對 GW 通道完全沒有作用。**

要注意這裡跟上一輪 `bh_ringdown` 的 `D_L`/`i` 不同：那時是「likelihood 不用、但模擬器用」的
**不一致**；這裡的 6 個是**兩邊一致地都不用**。一致地無用不會破壞 SBC 的前提（posterior 就等於
prior，rank 會均勻，就像上一輪的 `D_L` 正對照組），但它讓 dynesty 白跑 6 個維度，也代表模型
宣告的物理內容有一半沒有進入 GW 推論。

`eta_r` / `eta_gamma` 是 radio／gamma 通道的效率參數，`log10_ell_q` / `p_lifetime` 是量子尺度與
壽命標度指數——這四個在 GW-only 分析裡沒有作用是可以理解的（雖然仍應該從 GW 的取樣維度移除）。
**真正的問題是另外兩個。**

## B.3 阻斷性問題 (I)：bounce burst 在任何先驗抽樣下都不可能落在分析片段內

`log10_A_bounce` 與 `log10_tau_bounce_yr` 是 bounce 模型**相對於 `bh_ringdown` 的唯一區別**
（多出來的那個爆發），而實測它們**完全沒有作用**。追查原因：

`log10_tau_bounce_yr` 的先驗是 `uniform(-3, 10)`，而 `tau_bounce_s()` 的換算是
`10**log10_tau_yr * GYR_S / 1e9`，也就是**以「年」為單位**。因此：

```
tau ∈ [10^-3 yr, 10^10 yr] = [3.156e+04 s, 3.156e+17 s]
```

模擬器只有在 `t_merger + tau < duration` 時才注入爆發，likelihood 也只有在
`t_bounce < times[-1]` 時才把爆發加進樣板。分析片段是 **4 s**（真實資料路徑是 32 s）：

| 片段長度 | 需要 `log10_tau_bounce_yr <` | 先驗中滿足的比例 |
|---|---|---|
| 4 s | −6.897 | **0.0000** |
| 32 s | −5.994 | **0.0000** |

先驗下界是 −3，離需要的 −6.9 差了近 4 個數量級。**沒有任何一組先驗抽樣會讓爆發進入片段**，
所以這兩個參數在 GW 通道上恆為死參數。

**這不是「複雜模型本來就會這樣」**——它代表目前這條 GW 路徑上，`bounce` 與 `bh_ringdown`
的差別只剩下 `eps_f` / `eps_Q` 兩個 ringdown 偏移量，爆發成分從來沒有被注入或被擬合過。
在這個狀態下對 `bounce` 跑 SBC，`log10_A_bounce` 與 `log10_tau_bounce_yr` 的 rank 會是均勻的
（因為 posterior = prior），**校準檢查會「通過」，但通過的是一個空的模型成分**。

## B.4 阻斷性問題 (II)：模擬器與 likelihood 的天線投影公式不同

`simulators/grav_wave.py`（bounce 走的物理振幅路徑）：

```python
fp, fc = antenna_response(i)          # fp = 0.5(1+cos^2 i),  fc = cos i
A_rd = h0 * np.sqrt(fp**2 + fc**2)
```

`likelihoods/gw_likelihood.py::_build_template()`（bounce 分支）：

```python
A_rd = h0 * 0.5 * (1.0 + cos_i**2)    # 只有 fp，沒有 fc
```

兩者的比值隨傾角變化：

| `i` [rad] | 模擬器 `√(fp²+fc²)` | likelihood `0.5(1+cos²i)` | 比值 |
|---|---|---|---|
| 0.0000 | 1.414214 | 1.000000 | **1.4142** |
| 0.5000 | 1.246399 | 0.885076 | 1.4082 |
| 1.0000 | 0.842137 | 0.645963 | 1.3037 |
| 1.5708 | 0.500000 | 0.500000 | 1.0000 |
| 3.0000 | 1.400096 | 0.990043 | **1.4142** |

也就是模擬器注入的振幅在面對面（face-on）時比樣板所模型化的**大 41%**。這與上一輪發現 (B)
是同一類的 forward-model 不一致。

**實測它確實造成偏差**（無雜訊注入，真值 `D_L = 100 Mpc`，沿 `D_L` 掃描 lnL 找最大值；
`h0 ∝ 1/D_L`，所以樣板必須把距離縮小同樣的倍數才能對上振幅）：

| `i` 真值 | 真值 `D_L` | 最大似然 `D_L` | 真值/最大似然 | 由天線比值預測 |
|---|---|---|---|---|
| 0.0000 | 100.0 | 70.83 | **1.4118** | 1.4142 |
| 0.5000 | 100.0 | 71.14 | **1.4057** | 1.4082 |
| 1.5708 | 100.0 | 100.05 | 0.9995 | 1.0000 |

三位有效數字吻合，成因確定。這會系統性地把 `D_L` 的 posterior 拉低（並與 `i` 耦合），
在 SBC 上會表現成 `D_L`／`i` 的 rank 非均勻——正是上一輪 `log10_A` 的失敗模式。

## B.5 阻斷性問題 (III)：`M` 的先驗支撐超出可分析頻帶（上一輪 (D) 在 bounce 上的版本）

`bounce` 的 `M` 先驗仍是 `log_uniform(5, 1000)`（上一輪只收窄了 `bh_ringdown` 的），而且
bounce 的 ringdown 頻率還多乘一個 `(1 + eps_f)`，`eps_f ∈ [-0.3, 0.3]`：

```
f_rd 在先驗上的範圍：[8.97, 6860.64] Hz     （可分析頻帶是 [20, 1700] Hz）
```

20000 次先驗抽樣：

| 情況 | 比例 |
|---|---|
| 被 `_build_template()` 直接拒絕（`f_rd < 20` 或 `f_rd > 1945.6`）→ lnL = `ll_min` | **0.1406** |
| 落在 band mask `[20, 1700]` 之外（含下面那個死區） | **0.1646** |
| 落在 1700–1945.6 Hz 死區（樣板建得出來但完全在遮罩外） | 0.0240 |

依上一輪的推導方式（`f_rd = k(a)/M · (1+eps_f)`，`k ∈ [11897.6, 32525.0] Hz·M⊙`），
要讓每一組 `(a_star, eps_f)` 都落在 `[20, 1700]` 內，需要 **`M ∈ [24.87, 416.42]`**。

另外 `_build_template()` 只檢查 ringdown 的 `f_rd`，**沒有檢查爆發成分的 `0.8 × f_rd`**；
先驗中有 16.13% 的抽樣其爆發頻率落在頻帶外。目前因為 B.3（爆發永遠不在片段內）而不影響結果，
但只要 B.3 被修好，這一項就會立刻變成真的問題。

## B.6 為什麼現在不跑 campaign

依任務的第 3 點：B.3 與 B.4 屬於與上一輪 (B)/(D) 同等級的 forward-model 不一致，
所以停在回報這一步。具體理由：

- **B.4 未修就跑，`D_L`／`i` 必定不校準**，而且原因已經確定，跑 200 組 dynesty（上一輪同規模
  約 3 小時，12 維只會更久）只會重新量一次已知答案。
- **B.3 未修就跑，`log10_A_bounce`／`log10_tau_bounce_yr` 的 rank 會漂亮地均勻**，
  但那個「通過」沒有意義——校準的是一個從未被注入、也從未被擬合的模型成分。
  把這種結果寫成「bounce 校準良好」會是實質上的誤導。
- **B.5 未修就跑，`M` 會重演上一輪 0.795 的 under-coverage**，成因也已經確定。

## B.7 修法選項與 trade-off（等決策，未動手）

### 針對 B.3（爆發永遠不在片段內）

| 選項 | 做法 | Trade-off |
|---|---|---|
| **B3-1** | 收窄 `log10_tau_bounce_yr` 先驗到讓爆發落在片段內（4 s 片段需 `< -6.90`，32 s 需 `< -5.99`） | 改動最小、SBC 立刻有意義；但先驗會被資料片段長度綁死，且 `τ ~ 10⁻⁷ 年` 與模型文獻裡「BH 壽命」的物理量級完全脫節——等於把參數重新定義成別的東西。**需要你判斷這是否可接受**。 |
| **B3-2** | 改成以「相對於 merger 的秒數」重新參數化（例如 `log10_dt_bounce_s ∈ [-3, log10(duration)]`），把「宇宙學壽命 τ」與「片段內爆發延遲」分成兩個概念 | 物理上最誠實：GW 片段本來就只能看到「爆發相對 merger 的延遲」，看不到 10¹⁰ 年的壽命；但這是模型規格改動，會影響 `summary_stats()`、`configs/models/bounce.yaml`、既有測試。 |
| **B3-3** | 維持現狀，明確把 `log10_A_bounce`／`log10_tau_bounce_yr`／`log10_ell_q`／`p_lifetime`／`eta_r`／`eta_gamma` 標成「非 GW 通道參數」，從 GW 取樣維度移除（像上一輪對 `bh_ringdown` 的 `D_L`/`i` 那樣） | 承認目前的 GW `bounce` 就是「ringdown + `eps_f`/`eps_Q` 偏移」；取樣從 12 維降到 6 維，dynesty 會快很多；但等於正式記錄「爆發成分在 GW 通道未實作」。 |

### 針對 B.4（天線投影不一致）

| 選項 | 做法 | Trade-off |
|---|---|---|
| **B4-1** | 改 likelihood，讓樣板用 `√(fp²+fc²)`，與模擬器一致 | 改動小；但 `√(fp²+fc²)` 混合了 plus/cross 兩個極化的振幅，對單一實值樣板而言不是標準做法。 |
| **B4-2** | 改模擬器，只注入 plus 極化 `fp = 0.5(1+cos²i)`，與 likelihood 一致 | 與 `ringdown_waveform()` 只回傳 `h_+` 的事實一致（函式 docstring 明寫「plus-polarization」），物理上比較自洽；但會改變所有既有 mock GW 資料的振幅（約 −29%），可能影響既有測試的數值斷言。 |
| **B4-3** | 兩邊都改成明確的 `F₊h₊ + F×h×` 雙極化投影 | 最正確；但要動 `ringdown_waveform()` 產生 cross 極化，改動範圍最大。 |

我的傾向是 **B4-2**（`ringdown_waveform()` 目前就只產生 `h_+`，所以只注入 `F₊h₊` 才是自洽的），
但這牽涉物理慣例，**由你決定**。

### 針對 B.5（`M` 先驗超出頻帶）

直接套用上一輪對 `bh_ringdown` 的做法：把 `bounce` 的 `M` 先驗收窄到 `[24.87, 416.42]` 之內
（例如 `[25, 415]`），並把 `eps_f` 一併納入界限推導。同樣要標註它是由最壞情況
（最高自旋 × `eps_f` 上界）決定的保守值。若同時修 B.3-2，爆發頻率 `0.8 × f_rd` 的頻帶檢查
也要一起加進 `_build_template()`。

## B.8 現況總結

`bounce` 目前**還不具備做有意義 SBC/coverage 的條件**。在 B.3 與 B.4 沒有決策之前跑 campaign，
得到的會是一份「三個參數不校準（原因已知）、兩個參數假性通過（模型成分是空的）」的報告，
沒有增加任何新資訊。等你決定方向後，再另開一輪執行 Part B 的 campaign。

## B.9 稽核使用的腳本

稽核腳本放在 session scratchpad（不進版控）：`bounce_audit.py`（逐參數活性）、
`bounce_audit2.py`（爆發時序／天線投影／頻帶覆蓋）、`bounce_audit3.py`（天線不一致造成的
`D_L` 偏差實測）。本文件中的每個數字都可由這三個腳本重現。

---

# Part C：修正後的 `bounce` GW 通道校準驗證（2026-09-06）

> 本節記錄依 §B.7 決策實作的四項修正（commit `fdf8870`），以及修正後對 `bounce` 執行的
> SBC/coverage 驗證結果。同樣不含任何天文物理宣稱，只描述 bounce 模型 GW 通道規格對齊與
> 校準驗證。

## C.1 實作的四項修正

| 對應發現 | 決策 | 檔案 | 改動 |
|---|---|---|---|
| **B.1** 結構性 | 取樣維度由 likelihood 決定 | `inference/bilby_runner.py` | 新增 `effective_parameter_names(model, likelihood)`：model 參數順序下與 `likelihood.parameter_names` 的交集；likelihood 未宣告參數時不做限制（維持既有行為）；likelihood 要求 model 沒有的參數時 **raise**。`_restrict_priors()` 套用到 dynesty 的先驗字典，metadata 新增 `sampled_parameters` / `model_parameters` / `likelihood_parameters` |
| **B.3** | B3-3 | `likelihoods/gw_likelihood.py` | `GWLikelihood("bounce").parameter_names` 8 → **6**（移除 `log10_A_bounce`、`log10_tau_bounce_yr`）；`_build_template()` 爆發分支加註原因（§B.3 數字）並保留分支本身 |
| **B.4** | B4-2 | `simulators/grav_wave.py` | `A_rd = h0·√(fp²+fc²)` → `A_rd = h0·fp`，`fp = 0.5(1+cos²i)`，與樣板公式及 `ringdown_waveform()` 只回傳 h₊ 一致 |
| **B.5** | 收窄 M 先驗 | `models/bounce.py` | 新增 `_m_prior_bounds_for_band()`（含 `eps_f` 因子）；`M` 先驗 `log_uniform(5, 1000)` → `log_uniform(25, 415)` |

`BlackToWhiteBounce.parameter_names` 維持 **12 個不變**——`log10_ell_q`、`p_lifetime`、
`eta_r`、`eta_gamma` 仍宣告在 model 上，只是不進入 GW 通道的取樣維度。

**附帶查證**（依指示實際讀了其他通道的 likelihood，未假設）：這四個參數目前**沒有任何**
likelihood 讀取。`RadioBurstLikelihood` 用的是 `log10_eta_r`、`XRayBurstLikelihood` 用的是
`log10_eta_gamma`，兩者都屬於 `PBHTunnelingWhiteHole`，與 bounce 的 `eta_r`／`eta_gamma`
是不同參數；`VisibilityLikelihood` 也沒有用到其中任何一個。B.1 的交集邏輯是通用的，
未來若某個通道真的開始使用它們，會自動被納入取樣維度。

### M 先驗界限的重新推導

未照抄稽核報告數字，從程式碼裡的 `kerr_qnm_frequency`、`eps_f` 實際先驗與
`BAND_LOW_HZ`／`BAND_HIGH_HZ` 重新推導：`f = k(a)/M · (1+eps_f)`，`k` 對自旋單調遞增，
要讓每組 `(a_star, eps_f)` 都在 `[20, 1700]` Hz 內需要

```
M >= k(0.998)·(1+0.3)/1700 = 24.872086
M <= k(0)·(1-0.3)/20       = 416.417493
```

**重新推導後確認與稽核報告的 `[24.87, 416.42]` 一致。** 採用 `[25, 415]` 留安全邊界。
這組界限由最壞情況（最高自旋 × `eps_f` 上界）決定，屬於保守值：低自旋或負 `eps_f` 的
更輕殘骸其實仍在頻帶內，要放寬需要 `(M, a_star, eps_f)` 聯合先驗，現有機制表達不了。

依 §B.7 的決定，**沒有**額外加 `0.8 × f_rd` 的頻帶檢查——爆發參數已不在取樣 theta 裡，
該分支不會被觸發。

## C.2 執行設定與 provenance

`InjectionRecovery` N=200、`ci_level=0.90`、dynesty `nlive=250`、`likelihood_mode=full`、
seed=20260906，context 與前幾輪相同。取樣維度 **6**（`M, a_star, eps_f, eps_Q, D_L, i`），
由 B.1 的交集邏輯決定並記錄在 metadata 裡。總耗時 24207 s（約 6.7 小時）。
posterior 樣本數 min/median/max = **516 / 777 / 1417**，全部遠高於 rank 分母 L=100。
median `ln_Z` = −9.63×10¹⁰，median `ln_Z_err` = 449.1 nats（發現 (C) 未處理，見 §C.6）。

**與前幾輪的一項方法差異，如實標註**：這一輪的 rank 是從 `InjectionRecovery` 保留的
posterior 算出來的，不是由 `SBCRunner` 直接產出。原本兩條 campaign 平行跑（`SBCRunner`
給 rank、`InjectionRecovery` 給 coverage），但兩者用同一 seed 抽同一批真值、產生同一批 mock
資料、用同一組 sampler 設定，等於同一件事做兩遍互相搶 CPU；為了縮短總時間砍掉了 `SBCRunner`
那條。rank 估計量（`utils.math_utils.compute_sbc_rank`）、thinning 規則
（`default_rng(seed + i).choice(n, size=min(L, n), replace=False)`）與均勻性判定
（`SBCResult` 的確定性 `kstest`）都與 `SBCRunner` 完全相同，而且 rank 與 coverage 現在
來自同一批 200 組抽樣，比兩條獨立 stream 更一致。

（順帶記錄一個被推翻的假設：原本以為兩條 stream 會給出逐位元相同的結果，實測**不是**——
同 index 的 posterior 樣本數不同、中位數最大差到 63%。`bilby.run_sampler(seed=...)` 在兩個
全域 RNG 使用歷史不同的進程裡並未完全決定化。所以兩者是獨立實現，只能擇一，不能互相推導。）

## C.3 SBC rank statistics

L=100，rank ∈ {0,…,100}；`kstest p` 是 `SBCResult` 的確定性單樣本檢定，
`χ² p` 是 20 個等寬 bin 的適合度檢定。

| 參數 | rank 平均（期望 50.0） | rank 標準差（期望 29.15） | frac rank=0 | frac rank=L | **kstest p** | χ² p |
|---|---|---|---|---|---|---|
| `M` | 46.56 | 28.60 | 0.015 | 0.025 | **0.100** | 0.924 |
| `a_star` | 48.76 | 30.58 | 0.010 | 0.025 | **0.412** | **0.018** |
| `eps_f` | 46.30 | 30.43 | 0.010 | 0.010 | **0.058** | 0.590 |
| `eps_Q` | 48.27 | 30.07 | 0.015 | 0.005 | **0.186** | **0.008** |
| `D_L` | 47.76 | 29.46 | 0.005 | 0.015 | **0.580** | 0.748 |
| `i` | 51.22 | 29.75 | 0.005 | 0.020 | **0.469** | 0.509 |

Rank 直方圖（20 bins，N=200，均勻期望每格 10）：

```
M        12 10 13 13 10 10  8 13 12 15  9  9  7  8 11  6 10  7  9  8
a_star   20  9 11 10  6  6 11 10 14 12  3  7  6 14 15 12  6  7  6 15
eps_f    16 13 12 11 12  7 15 10 10  7  9  8  6  8 10 11  4  9 13  9
eps_Q    17 15 10  7  8  3  8 20  8 12  8  6 12  7 11  8  6 15 14  5
D_L      14 12  8 15 10  7 11 13  9  6  8 16  7  9  8 10 11  9  9  8
i        12 14 10 12  5  5  9 10  8  6 12 12 12  9 15 12  6  6 12 13
```

**六個參數的 KS 均勻性檢定全部通過（p > 0.05）**，沒有任何一個出現前幾輪那種
端點堆積（最大的 `frac_rank_0` / `frac_rank_L` 都 ≤ 0.025，均勻期望值是 0.0099）。

但 `a_star`（χ² p = 0.018）與 `eps_Q`（χ² p = 0.008）在 20-bin 適合度檢定上偏離：
KS 對 CDF 層級的位移敏感、χ² 對局部起伏敏感，兩者不一致代表這兩個參數的 rank 分布
在中段有結塊（`eps_Q` 的第 8 個 bin 有 20 個、第 6 個 bin 只有 3 個）而不是整體位移。
**多重比較的處理**：本節共做了 12 個檢定（6 KS + 6 χ²），Bonferroni 的 5% 門檻是
p = 0.0042，**這兩個值都在門檻之上**，所以在校正多重比較之後不構成統計顯著的不校準。

## C.4 Credible interval coverage

括號內為偏離名目值的倍數（以二項式標準誤計）。

| 參數 | 名目 50% | 名目 68% | 名目 90% |
|---|---|---|---|
| `M` | 0.490 (−0.28σ) | 0.685 (+0.15σ) | **0.905** (+0.24σ) |
| `a_star` | 0.490 (−0.28σ) | 0.645 (−1.03σ) | 0.855 (−1.81σ) |
| `eps_f` | 0.470 (−0.85σ) | 0.630 (−1.46σ) | 0.895 (−0.23σ) |
| `eps_Q` | 0.485 (−0.42σ) | **0.590 (−2.59σ)** | 0.885 (−0.67σ) |
| `D_L` | 0.510 (+0.28σ) | 0.665 (−0.45σ) | **0.905** (+0.24σ) |
| `i` | 0.520 (+0.57σ) | 0.675 (−0.15σ) | 0.875 (−1.07σ) |

18 個數字中 17 個落在名目值的 1.9σ 以內。**唯一的例外是 `eps_Q` 在 68% 名目下的
0.590（−2.59σ）**，見 §C.5。`calibration_report.py::_evaluate_coverage` 的 PASS 條件
（`0.8 ≤ cov90 ≤ 1.0`）六個參數全部通過。

## C.5 一句話結論

**修正 B.1／B.3／B.4／B.5 之後，`bounce` 模型在 GW 通道的 6 個取樣參數
（`M, a_star, eps_f, eps_Q, D_L, i`）SBC rank 均勻性全部通過（kstest p = 0.100 / 0.412 /
0.058 / 0.186 / 0.580 / 0.469），90% coverage 全部落在 0.855–0.905、18 個 coverage 數字中
17 個在 1.9σ 以內；唯一需要留意的是 `eps_Q` 在 68% 名目下實測 0.590（−2.59σ），
以及 `a_star`／`eps_Q` 的 20-bin χ² 檢定分別為 0.018／0.008（校正 12 個檢定的多重比較後
不顯著）——這三個訊號一致地指向 `eps_Q`，如實回報，未做任何進一步修正。**

### `D_L` 的對比：從「乾淨正對照組」變成真正的檢驗

上一輪 `bh_ringdown` 的 `D_L` 完全不進 likelihood，posterior 就是 prior，所以它校準完美
只證明了 SBC 工具鏈本身沒問題，對模型是否對齊沒有任何說服力。

這一輪不同：bounce 的 `D_L` 透過 `h0 = GM/(c²D_L)` **真正進入 likelihood**，而且它正是
B.4 天線不一致最直接的受害者——修正前的實測是「真值 100 Mpc，最大似然落在 70.83 Mpc」，
偏差倍數 1.4118 與天線比值 1.4142 吻合到三位有效數字。

修正後 `D_L` 的結果是 **kstest p = 0.580、χ² p = 0.748、coverage 0.510 / 0.665 / 0.905
（+0.28σ / −0.45σ / +0.24σ）**，是六個參數裡最乾淨的一個。這是 B4-2（模擬器改成只注入
plus 極化）確實把模擬器與樣板對齊了的直接證據——如果天線公式仍然不一致，
`D_L` 的 posterior 會系統性偏低、rank 會往上界堆積，而實測 `frac_rank_L` 只有 0.015。

## C.6 仍然成立、未在本輪處理的事項

1. **發現 (C)（第五輪 taper 在 mock 路徑失效）依舊未處理**：本輪 median `ln_Z_err` 是
   **449.1 nats**。如 `BH_RINGDOWN_SBC_COVERAGE_REPORT.md` §6(C) 所述，這不影響 posterior
   的校準（本節數字就是證據），但 evidence 的精度在 mock 上仍然是壞的。
2. **`eps_Q` 的 68% under-coverage 與 χ² 結塊**（§C.4、§C.5）——如實回報，未動手。
3. **爆發成分在 GW 通道仍未實作**（B3-3 的定義就是如此）。要在 GW 通道推論爆發，需要先做
   B3-2 的重新參數化（把「相對 merger 的延遲」與「宇宙學壽命」分開）。目前 GW 通道的
   `bounce` 與 `bh_ringdown` 的差別只有 `eps_f`／`eps_Q` 兩個 ringdown 偏移量。
4. **`M` 先驗界限的保守性**（§C.1）：需要聯合先驗才能放寬。
5. ~~**`InjectionRecoveryResult._compute_coverage` 對 posterior 裡不存在的參數會退回
   `(-inf, +inf)`**~~ **（已於 2026-09-07 修正，見 §C.8）**。舊行為是**無條件算作被覆蓋**
   （coverage = 1.0）且無任何警告。Part C 的分析因為只取 6 個取樣參數而未受影響，
   但這是一個與 fail-closed 原則相衝突的靜默預設值。
6. **N=200 的檢定力限制**：`SBCRunner` docstring 建議 N ≥ 1000。

## C.7 第三輪產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `docs/calibration/bounce_sbc_rank_statistics.csv` | 6 個參數的 rank 統計、kstest／χ² p 值、三個層級的 coverage | 是 |
| `docs/calibration/bounce_coverage.csv` | 50%/68%/90% coverage 與二項式標準誤 | 是 |
| `docs/calibration/bounce_coverage_90_native.csv` | `InjectionRecoveryResult.summary()` 原生輸出 | 是 |
| `artifacts/calibration/bounce_sbc/rank_hist_*.png`, `ranks.json`, `theta_true.csv`, `evidences.csv`, `meta.json` | rank 直方圖與每次注入的 provenance | 否（`artifacts/` 在 `.gitignore` 內） |


## C.8 後續修正：coverage 對未取樣參數改為 fail-closed（2026-09-07）

處理 §C.6 第 5 項。**這是工具面的 fail-closed 補強，不影響已經跑完的 bh_ringdown 或
bounce 校準結果**——Part C 的分析本來就只取 6 個實際取樣的參數，所以沒有重跑任何 campaign。

### 舊行為

`validation/injection.py::InjectionRecoveryResult._compute_coverage()` 的
`lo, hi = ci.get(p, (-np.inf, np.inf))`：對 posterior 裡不存在的參數，區間退回
`(-inf, +inf)`，於是**每一個真值都落在區間內**，覆蓋率算出來是 1.0，而且分母仍然除以全部
`n` 次注入。結果是一個被宣告但從未被取樣的參數會回報「完美覆蓋」，沒有警告也沒有錯誤。

### 新行為

依每個注入參數實際拿到 credible interval 的次數把參數分成三類：

| 類別 | 條件 | 處置 |
|---|---|---|
| 可計分 | 全部 `n` 次注入都有區間 | 進入 `coverage` / `sbc_ranks`，數值行為完全不變 |
| `unsampled_parameters` | **0** 次有區間 | 排除，不計分；`coverage_of()` 查詢時 raise |
| `partially_sampled_parameters` | 介於 1 與 `n-1` 之間 | 排除（否則分母不一致）；查詢時 raise |

- `ci.get(p, ...)` 的預設值整個刪除：迴圈只對「保證存在」的參數取 `ci[p]`，所以這裡再出現
  `KeyError` 就是真正的不一致，而不是被預設值蓋掉。
- 新增公開方法 `coverage_of(param)`：可計分的參數回傳數值；未取樣／部分取樣／根本沒注入的
  參數分別 raise 帶有不同訊息的 `KeyError`，由呼叫端自己決定要不要 catch。
- 被排除的參數會用 `logger.warning` 明確列出，不是靜默消失。

另外，`run_injections()` 在單次推論失敗時會塞一個「posterior 只有一列、且就位在真值上」的
placeholder，它的區間必然包含真值。這個 placeholder 保留（維持各串列長度對齊），但
metadata 新增 `failed_indices` 與 `n_failed`，讓「這次 coverage 裡含有幾筆是失敗的
placeholder」變成可見的 provenance 而不是隱含假設。

### 呼叫端檢查

依要求逐一確認，沒有任何既有邏輯依賴「查詢未取樣參數」這個行為：

| 呼叫端 | 用到什麼 | 影響 |
|---|---|---|
| `injection.py::summary()` | `self.coverage` / `self.sbc_ranks` | 兩者鍵集合一致，改動後對 bounce 從 12 列變成 6 列——正是這次要修掉的虛胖 |
| `calibration_report.py::_evaluate_coverage()` | `summary()` 的 `coverage_ok` | 判定條件是 `n_ok == n_total`，被移除的都是舊行為下必然 `ok=True` 的虛列，所以 PASS/FAIL 結論不變；若全部參數都未取樣則 `cov_df` 為空，回傳 `pass=False`（fail-closed） |
| `calibration_report.py::_plot_coverage()` | 同上 | 少畫幾根虛胖的長條 |
| `cli.py` 的 sensitivity 曲線、`compute_sensitivity_curve()` | 只用 `theta_true` / `evidences` | 不受影響 |

### 測試

新增 `tests/test_coverage_fail_closed.py`（11 個）：未取樣參數被排除且不等於 1.0、查詢時
raise、部分取樣同樣排除並 raise、查詢未注入參數 raise、排除時會發出警告；以及鎖住既有正常
數值的 regression（10 次注入中 7 次覆蓋 → `coverage == 0.7`、SBC rank 全部等於 50、
`summary()` 只列出計分參數、全覆蓋仍回傳 1.0、空 campaign 仍可處理）；最後一個測試用
真的 `InjectionRecovery` 跑 bounce，確認 12 個宣告參數中 6 個計分、6 個進
`unsampled_parameters`。

`pytest`：**192 passed, 1 skipped, 1 deselected**（`d4bce4e` 基準是 181 passed；差額為新增的
11 個測試），無非預期 regression。

---

# Part D：B3-2 爆發時序重新參數化與校準驗證（2026-09-07）

> 本節記錄依 B3-2 方向做的爆發時序重新參數化（commit `5e2e94c`），以及修正後對 `bounce`
> 執行的 SBC/coverage 驗證結果。同樣只描述 bounce 模型 GW 通道規格與校準驗證，
> 不含任何天文物理宣稱。
>
> **結論先講：這次校準驗證結果是失敗的。** 8 個取樣參數全部嚴重不校準
> （kstest p 介於 5×10⁻⁶ 與 2×10⁻²⁵ 之間，90% coverage 只有 0.39–0.48）。
> 依 fail-closed 原則如實回報數字與診斷，未做任何進一步修正。

## D.1 重新參數化內容

`log10_tau_bounce_yr`（宇宙學尺度的黑洞壽命）與「爆發相對 merger 的延遲」是兩個不同的
物理量，先前被混為一談。這次把後者獨立出來成為新參數 `log10_dt_bounce_s`：

| 項目 | 內容 |
|---|---|
| 新參數 | `log10_dt_bounce_s`，`uniform(-3.0, 0.4)`，單位 log10(秒) |
| 界限推導 | `_dt_bounce_prior_bounds()`：爆發要進入樣板需 `t_merger + dt < times[-1]`，即 `dt < duration - t_merger - 1/sample_rate`；下界是取樣率可解析的最短延遲（4 個取樣點）。在 `configs/runs/gw_run.yaml` 的 4096 Hz / 4 s / t_merger=1 s 幾何下得 **[-3.0103, +0.4771]** |
| `log10_tau_bounce_yr` | **保留在 model 上，維持不變**，仍標記為 GW 通道不使用 |
| GW 取樣維度 | 6 → **8**（加回 `log10_A_bounce` 與 `log10_dt_bounce_s`） |
| 模擬器 | 用同一個時序定義注入爆發、同一個 `times[-1]` 守衛 |
| 爆發頻帶檢查 | `_build_template()` 新增 `0.8×f_rd` 的頻帶檢查（B.5 記錄的死角），band 外回傳 `None` |
| `M` 先驗 | `[25, 415]` → **`[25, 330]`**：補上爆發頻帶檢查後，f_rd=20 Hz 的爆發落在 16 Hz（低於截止），有效 ringdown 頻窗變成 `[25, 1700]` Hz，重新推導得 `[24.872086, 333.133994]` |
| 共用常數 | `BOUNCE_BURST_FREQ_FACTOR` / `Q_FACTOR` / `Q_MIN` 移入 `utils/constants.py`，模擬器、likelihood、先驗推導共用一份 |

寫「模擬器與樣板逐樣本一致」測試時另外抓到一個 bug：模擬器把 D_L 用寫死的 `3.086e22`
換算，而 likelihood 用 `utils.constants.MPC_M = 3.085677581e22`，相對差 **1.0449e-04**，
注入與模型化的振幅因此差這個倍數。與 B.4 同一類但小四個數量級，已改用 `MPC_M`。

`pytest`：**198 passed, 1 skipped, 1 deselected**（基準 `13bb132` 是 192 passed；移除 3 個
記錄 B3-3 舊狀態的過時測試、新增 9 個），無非預期 regression。

## D.2 執行設定與實際完成度

N=100（依決策從 200 下修；保留 `nlive=250` 以免窄峰維度上 live points 不足產生偽陽性的
不均勻 rank）、L=100、dynesty `nlive=250`、`likelihood_mode=full`、seed 20260907。

為縮短 wall clock，campaign 切成 4 個 shard 平行執行，shard s 用 `rng_seed = 20260907 + s`、
各 25 筆；`seed_i = rng_seed + i` 的聯集正好等於單一 N=100 run 的同一組種子，統計內容不變。

**實際可用 N = 97**，三筆缺失，原因全部是基礎設施而非結果篩選：

| 缺失 | seed | 原因 |
|---|---|---|
| shard 25, i=24 | 20260956 | 容器重啟時仍在取樣，未完成 |
| shard 75, i=24 | 20261006 | 同上 |
| shard 25, i=23 | 20260955 | `run_injections` 的失敗 placeholder（單列、位在真值上），依 §C.8 的 fail-closed 原則排除，不計分 |

容器在 14:08 之後開始每 ~20 分鐘重啟一次，短於單筆注入所需的 20–45 分鐘。缺失是由固定的
index 位置決定（各 shard 的最後一筆），與該筆的推論結果無關，因此對 rank 統計而言是
missing-at-random。N 從 100 降到 97 讓二項式標準誤從 0.0332 變成 0.0337，對下面的結論
沒有影響——所有 p 值都在 10⁻⁵ 以下。

保留下來的 97 筆 posterior 樣本數 min/median/max = **736 / 1084 / 1875**，皆遠高於 L=100。

## D.3 SBC rank statistics（N=97, L=100）

| 參數 | rank 平均（期望 50.0） | rank 標準差（期望 29.15） | frac rank=0 | frac rank=L | **kstest p** | χ² p |
|---|---|---|---|---|---|---|
| `M` | 22.81 | 32.14 | **0.495** | 0.010 | **2.2e-25** | 4.5e-97 |
| `a_star` | 26.44 | 34.35 | **0.464** | 0.062 | **3.3e-19** | 3.6e-69 |
| `eps_f` | 65.14 | 39.47 | 0.031 | **0.464** | **3.3e-19** | 2.1e-74 |
| `eps_Q` | 30.27 | 36.87 | **0.505** | 0.041 | **4.8e-23** | 1.3e-88 |
| `log10_A_bounce` | 25.34 | 33.78 | **0.485** | 0.010 | **4.1e-21** | 2.4e-88 |
| `log10_dt_bounce_s` | 26.66 | 32.25 | **0.495** | 0.010 | **5.5e-22** | 2.0e-88 |
| `D_L` | 24.05 | 30.04 | **0.495** | 0.010 | **5.0e-22** | 3.7e-80 |
| `i` | 48.38 | 40.65 | 0.258 | 0.216 | **5.1e-06** | 3.6e-36 |

Rank 直方圖（20 bins，N=97，均勻期望每格 4.85）：

```
M                  53  7  2  1  0  4  3  4  2  0  1  0  3  3  3  2  3  1  1  4
a_star             46  5  7  3  1  1  2  3  3  0  2  4  2  2  2  3  2  1  1  7
eps_f              10  4  5  4  0  5  3  1  3  2  3  2  1  1  0  1  1  1  3 47
eps_Q              51  0  0  1  4  4  2  1  1  2  2  5  1  2  3  1  3  4  3  7
log10_A_bounce     51  5  4  1  3  2  0  1  2  0  6  4  0  3  1  2  2  4  3  3
log10_dt_bounce_s  51  2  0  1  0  5  2  2  1  6  4  5  4  1  3  2  3  1  1  3
D_L                49  2  3  3  0  5  5  2  1  4  5  4  1  0  4  2  3  3  0  1
i                  29  2  2  2  2  3  3  3  1  3  2  2  2  3  3  4  4  2  1 24
```

## D.4 Coverage（N=97）

括號內是偏離名目值的倍數（以二項式標準誤計）。

| 參數 | 名目 50% | 名目 68% | 名目 90% |
|---|---|---|---|
| `M` | 0.216 (−6.8σ) | 0.268 (−9.2σ) | 0.423 (−9.5σ) |
| `a_star` | 0.216 (−6.8σ) | 0.309 (−7.9σ) | 0.454 (−8.8σ) |
| `eps_f` | 0.206 (−7.2σ) | 0.268 (−9.2σ) | 0.423 (−9.5σ) |
| `eps_Q` | 0.206 (−7.2σ) | 0.330 (−7.3σ) | 0.392 (−10.3σ) |
| `log10_A_bounce` | 0.196 (−7.6σ) | 0.289 (−8.5σ) | 0.454 (−8.8σ) |
| `log10_dt_bounce_s` | 0.309 (−4.1σ) | 0.402 (−5.6σ) | 0.443 (−9.1σ) |
| `D_L` | 0.330 (−3.6σ) | 0.381 (−6.1σ) | 0.485 (−8.2σ) |
| `i` | 0.258 (−5.5σ) | 0.361 (−6.6σ) | 0.474 (−8.4σ) |

**24 個數字全部低於名目值，最小偏離 3.6σ，最大 10.3σ。** 這是單向的、系統性的
under-coverage：posterior 一致地太窄。

## D.5 一句話結論

**B3-2 重新參數化之後，`bounce` 模型 GW 通道的 8 個取樣參數全部嚴重不校準
（kstest p 全部 ≤ 5×10⁻⁶，90% coverage 只有 0.39–0.48 對名目 0.90），
偏差方向一致是 posterior 系統性過窄；這次新加回來的 `log10_A_bounce` 與
`log10_dt_bounce_s` 也在其中，所以這次重新參數化沒有通過驗證。**

作為對照：同一條 pipeline 在上一輪（Part C，6 個參數、沒有爆發成分）是全部校準良好的
（kstest p 0.058–0.580，90% coverage 0.855–0.905）。差別只有這次加回的爆發成分與
隨之而來的自由時序參數。

## D.6 診斷數據（供判斷成因，未據此動手修正）

### posterior 寬度：`log10_dt_bounce_s` 被釘到物理上不可能的精度

posterior 標準差相對於先驗寬度的中位數：

| 參數 | 相對寬度中位數 |
|---|---|
| `eps_f` | 1.67e-01 |
| `i` | 1.23e-01 |
| `D_L` | 8.81e-02 |
| `eps_Q` | 7.88e-02 |
| `a_star` | 1.81e-02 |
| `M` | 1.28e-02 |
| `log10_A_bounce` | **1.40e-05** |
| `log10_dt_bounce_s` | **7.11e-08** |

`log10_dt_bounce_s` 的先驗跨度是 3.4 個十進位量級，7.11e-08 的相對寬度代表延遲被定到
約 **2.4×10⁻⁷ 秒**——比取樣間隔（1/4096 = 2.44×10⁻⁴ 秒）還小三個數量級，而爆發本身的
衰減時間中位數是 ~5 ms。在有限訊噪比下把到達時間定到取樣間隔的千分之一是物理上不可能的。

### 真值離 posterior 有多遠

以 posterior 標準差為單位（校準良好時中位數應約 0.67、90 百分位約 1.6）：

| 參數 | \|z\| 中位數 | \|z\| 90 百分位 | \|z\| 最大 | 真值完全落在 posterior 範圍外的比例 |
|---|---|---|---|---|
| `M` | 2.13 | 5.7e+08 | 4.4e+09 | 0.474 |
| `a_star` | 2.51 | 1.8e+11 | 5.0e+11 | 0.505 |
| `eps_f` | 2.27 | 3.7e+08 | 3.5e+09 | 0.464 |
| `eps_Q` | 3.15 | 5.1e+08 | 1.4e+09 | 0.505 |
| `log10_A_bounce` | 2.50 | 3.1e+05 | 4.9e+05 | 0.485 |
| `log10_dt_bounce_s` | 2.60 | 1.3e+11 | 3.1e+11 | 0.474 |
| `D_L` | 2.19 | 9.9e+02 | 1.7e+03 | 0.454 |
| `i` | 1.88 | 6.3e+01 | 9.2e+01 | 0.443 |

10¹¹ 個標準差的偏離不可能出自一個「只是有點窄」的 posterior；它代表對那些注入而言
posterior 幾乎沒有寬度。約 **27%** 的注入其 `M` 的 posterior 寬度小於先驗寬度的 10⁻⁶
（但仍有中位數 1570 個相異值，所以不是字面上的點質量，而是收斂到一個極薄的區域）。
每個參數有大約一半的真值完全落在 posterior 範圍外，落在哪一側則近乎隨機——這正好產生
rank 在 0 與 L 兩端堆積的觀測樣態。

### 排除掉的分析面錯誤

為了確認上述不是合併或配對造成的假象，做過以下檢查：

- 真值重建（從 seed 用 `model.sample_prior(default_rng(seed))`）與兩個有存檔的 shard 的
  `theta_true.csv` 逐一比對通過（rtol 1e-12；差異來自 CSV round-trip，實測最差 1.1e-15）。
- rank=0 的比例在四個 shard 之間一致（0.440 / 0.478 / 0.480 / 0.458），不是某一個 shard
  配對錯位。
- 抽查 shard 0 injection 0：8 個參數的真值全部落在 posterior 範圍內，且 posterior 寬度正常。
- 失敗 placeholder（單列、13 欄）已被排除，不是它拉低了 coverage。

### 一個可能成因（**假說，未驗證，未據此修改任何程式碼**）

發現 (C)（第五輪 taper 在 mock 路徑上讓 per-bin ⟨d|d⟩ 膨脹約 10⁷ 倍）至今未處理。
本輪 median `ln_Z` = **−1.23×10¹¹**、median `ln_Z_err` = **626.9**。likelihood 尺度被
膨脹等同於把有效雜訊變異數縮小同樣的倍數，posterior 會系統性過窄。

這個假說能解釋 under-coverage 的方向，但**不能單獨解釋為什麼上一輪沒事**——Part C 用的是
同一個被膨脹的 likelihood，卻校準良好。兩輪之間的差別是這次多了一個自由的**時序**參數：
在膨脹的 likelihood 尺度下，一個尖銳暫態的到達時間會被定到荒謬的精度，而爆發頻率
`0.8×f_rd` 又把 `M`、`a_star`、`eps_f` 一起拖進同一個極薄的解。也就是說，(C) 的尺度問題
在所有參數都是緩變的時候是無害的，一旦引入尖銳的時序參數就會浮現。

**這只是假說。** 要確認需要用 `TAPER_ALPHA=0.0` 跑一次對照 campaign（比照第一輪
`BH_RINGDOWN_SBC_COVERAGE_REPORT.md` §3.2 Campaign C 的做法），本輪沒有做，
也沒有據此修改任何東西。

## D.7 產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `docs/calibration/bounce_b32_sbc_rank_statistics.csv` | 8 個參數的 rank 統計、kstest／χ² p 值、三個層級的 coverage 與偏離 σ | 是 |
| `docs/calibration/bounce_b32_posterior_widths.csv` | 每筆注入、每個參數的 posterior 相對寬度 | 是 |

---

# Part E：mock likelihood 尺度對校準影響的診斷（2026-09-07）

> **這是診斷性工作，不代表任何 production 行為的改變。** 沒有修改任何 production 程式碼；
> `TAPER_ALPHA` 只在診斷腳本自己的行程內被覆寫（比照第一輪
> `BH_RINGDOWN_SBC_COVERAGE_REPORT.md` §3.2 Campaign C 的做法），repo 內的常數維持 0.1。
> 本節不含任何天文物理宣稱。

## E.1 要檢驗的假說

§D.6 提出的假說是：發現 (C)（taper 在 mock 路徑上讓 per-bin ⟨d|d⟩ 膨脹約 10⁷ 倍）
疊加 B3-2 新引入的自由連續時序參數 `log10_dt_bounce_s`，是 Part D 中 posterior
系統性崩潰過窄（10¹¹ σ 偏離、時序精度到 ~2.4×10⁻⁷ 秒）的根本原因。

假說的機制是：likelihood 尺度被膨脹 ≈ 有效雜訊變異數被縮小 ≈ posterior 被壓窄。

## E.2 結論：**假說被否證**

**關掉 taper 確實把 likelihood 的尺度完全修好了，但它完全沒有改變 likelihood 峰的寬度。**
因此 (C) 不可能是 posterior 過窄的成因。

### 尺度確實被修好（taper 覆寫有生效）

8 筆配對注入（種子與 Part D 相同，資料逐位元相同，只有 likelihood 的 taper 不同）：

| 量 | `TAPER_ALPHA=0.1` | `TAPER_ALPHA=0.0` |
|---|---|---|
| per-bin ⟨d\|d⟩（理論值 2.0） | **3.21×10⁷** | **2.18** |
| null lnL | −1.08×10¹¹ | −7.34×10³ |

這重現了發現 (C) 的既有量測，也確認診斷腳本的覆寫確實生效。

### 但 likelihood 峰的寬度沒有變

對每筆注入，把其他參數固定在真值、沿單一參數掃描 lnL，量測 lnL 下降 0.5 的半寬
（局部高斯下即 1σ 寬度）。**taper 0.0 相對 0.1 的寬度比值：**

| 參數 | 半寬比值（0.0 / 0.1）中位數 |
|---|---|
| `log10_dt_bounce_s` | **0.9956** |
| `M` | **0.9986** |
| `log10_A_bounce` | **1.0000** |
| `eps_f` | **1.0000** |

換句話說，taper 讓 |lnL| 差了 7 個數量級，卻幾乎完全不改變峰的形狀。

### 直接的機制證據

固定同一組參數、在 `log10_dt_bounce_s` 上走一個 10⁻⁷ 的小步：

| | lnL 絕對值 | 該小步造成的 Δ(lnL) |
|---|---|---|
| `TAPER_ALPHA=0.1` | 1.207×10¹¹ | **−8.4076×10⁻³** |
| `TAPER_ALPHA=0.0` | 6.744×10³ | **−8.4144×10⁻³** |

**曲率一致到 0.08%。** taper 對 lnL 的貢獻在參數空間中（到 0.1% 以內）是一個
**與參數無關的加法常數**——它平移 lnL，不縮放殘差。既然不縮放，就不會壓窄 posterior。

數值可解析性已確認，不是被浮點誤差蓋掉：在 |lnL| = 1.21×10¹¹ 時 float64 的 ulp 是
1.53×10⁻⁵ nats，0.5 nat 的落差相當於 3.28×10⁴ 個 ulp；同一點重複計算的離散度為 0
（完全確定性）。

## E.3 那麼窄峰從哪裡來

**峰本來就很窄，兩個 taper 設定下都是。** `log10_dt_bounce_s` 的半寬中位數是 0.0164 dex，
相對於 3.4 dex 的先驗跨度是 **4.8×10⁻³**；最窄的一筆是 1.5×10⁻⁴ dex，即先驗跨度的
**4.4×10⁻⁵**。

這是 likelihood 本身的性質，不是 taper 造成的。一個合理的來源是
`utils/math_utils.ringdown_waveform()` 的波形在 `t0` 是**不連續的**（`t < t0` 恆為 0，
到 `t0` 突然跳到振幅 A）。階梯不連續點的位置可以被 matched filter 定位到遠比衰減時間
（~5 ms）精細的程度。**本節沒有做進一步實驗去確認這一點，僅記錄為觀察。**

如果 likelihood 真的有一個這麼窄的針狀峰，那麼「posterior 很窄」本身並不是錯誤——
錯的是 posterior 沒有涵蓋真值。這把問題的性質從「likelihood 尺度錯誤」改寫成
**「取樣器解析度不足」**：`nlive=250` 在 8 維空間裡要找到一個在單一維度上就只佔先驗
4×10⁻⁵ 的針，本來就極可能失敗、收斂到附近一個假的窄模態上——這正好產生 Part D 觀測到的
「posterior 極窄且不含真值、rank 在 0 與 L 兩端堆積、|z| 達 10¹¹」的樣態。

**這是本節證據指向的方向，不是已驗證的結論，也沒有據此修改任何東西。**

## E.4 taper=0.0 的 SBC 對照 campaign

依要求也啟動了 N=24（4 shard × 6，種子與 Part D 前 6 筆配對）、`nlive=250`、
`likelihood_mode=full`、8 維、`TAPER_ALPHA=0.0` 的 SBC/coverage 對照 campaign。

**該 campaign 未能完成。** 執行期間容器每 10–20 分鐘被回收一次，每次啟動後背景行程只存活
約 3.5–4.5 分鐘。bilby 預設的 `check_point_delta_t` 是 600 秒，行程活不到第一個 checkpoint，
`resume=True` 形同虛設；改成每 90 秒 checkpoint（`check_point_delta_t=90`，
`check_point_plot=False`——只改狀態寫入頻率，`nlive`、sampler、資料、taper 覆寫皆未變）之後
checkpoint 確實開始產生，但累積速度仍遠低於需求。截至記錄時完成 **2/24**。

§E.2 的曲率量測不依賴這個 campaign，而且它對 (C) 的機制已經給出決定性答案，
所以本節的結論不受 campaign 未完成影響。若之後仍要取得 taper=0.0 的 SBC 數字，
需要一個不會每幾分鐘回收一次的執行環境。

## E.5 一句話結論

**假說不成立：關掉 taper 把 mock 的 likelihood 尺度從 per-bin ⟨d|d⟩ = 3.21×10⁷ 修回
2.18（理論值 2.0）、null lnL 從 −1.08×10¹¹ 修回 −7.3×10³，但 likelihood 峰的寬度比值是
0.9956–1.0000、曲率一致到 0.08%，所以發現 (C) 對 posterior 寬度沒有影響，不是 Part D
校準崩潰的成因；證據反而指向「burst 時序方向本來就存在的極窄 likelihood 峰
（先驗跨度的 4×10⁻⁵ 到 5×10⁻³）超出 `nlive=250` 在 8 維下的解析能力」，但這一點尚未驗證，
本輪未據此做任何修正。**

## E.6 產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `docs/calibration/bounce_taper_curvature_diagnostic.csv` | 8 筆配對注入在兩個 taper 設定下的 lnL、null lnL、per-bin ⟨d\|d⟩，以及四個參數的 lnL 峰半寬 | 是 |

---

# Part F：ringdown 起始不連續是否造成窄峰的診斷（2026-09-08）

> **這是診斷性驗證，不是 production 修正。** 沒有修改 `likelihoods/gw_likelihood.py`、
> `simulators/grav_wave.py`、`utils/math_utils.py` 或任何其他 production 檔案；
> `ringdown_waveform` 只在診斷腳本自己的行程內被重新綁定（同時綁到
> `simulators.grav_wave` 與 `likelihoods.gw_likelihood` 兩個 namespace，
> 讓注入資料與擬合樣板用同一個波形）。本節不含任何天文物理宣稱。

## F.1 要檢驗的機制

§E.3 記錄的觀察：`utils/math_utils.ringdown_waveform()` 的波形在 `t0` 是**階躍不連續**的
（`t < t0` 恆為 0，到 `t0` 瞬間跳到振幅 A）。假說是：這個不連續讓 matched filter 能把
爆發起始時刻定位到遠比衰減時間（~5 ms）精細的程度，因而在 `log10_dt_bounce_s` 方向
產生針狀 likelihood 峰，超出 `nlive=250` 在 8 維下的解析能力。

檢驗方式：把起始邊緣換成平滑開關

```
env    = exp(-pi f max(t - t0, 0) / Q)        # 封包在 t < t0 夾住不成長
switch = 0.5 * (1 + tanh((t - t0) / w))
h      = A * env * cos(2 pi f (t - t0) + phase) * switch
```

過渡寬度 `w ∈ {1, 3, 10, 30} × dt`（`dt = 1/4096 s = 0.244 ms`），與原本的階躍（`w = 0`）比較。
量測方法與 §E.2 完全相同：固定其他 7 個參數在真值，沿 `log10_dt_bounce_s` 掃描 lnL，
取 lnL 下降 0.5 的半寬；外加固定 1e-7 步長的 Δ(lnL)。8 筆注入，種子與 Part D 配對。

**掃描視窗**用 ±1.7 dex（`log10_dt_bounce_s` 先驗跨度 3.4 dex 的一半），並明確記錄
右設限（峰在視窗內從未下降 0.5）的次數，避免用窄視窗把寬峰誤記成 NaN 後再取中位數。

## F.2 結論：**機制不成立**

平滑化起始邊緣**沒有系統性地把峰變寬**。

| `w` | 半寬中位數（dex） | 佔先驗跨度 | 相對階躍的比值 |
|---|---|---|---|
| 0（階躍） | 0.003618 | 1.06×10⁻³ | 1.000 |
| 1 dt（0.244 ms） | 0.003215 | 9.45×10⁻⁴ | 0.889 |
| 3 dt（0.732 ms） | 0.003154 | 9.28×10⁻⁴ | 0.872 |
| 10 dt（2.44 ms） | 0.002656 | 7.81×10⁻⁴ | 0.734 |
| 30 dt（7.32 ms） | 0.004336 | 1.28×10⁻³ | 1.199 |

比值在 0.73–1.20 之間來回，沒有隨 `w` 單調變化——這正是要求 3 想檢驗的「峰寬是否與過渡寬度
有清楚對應關係」，答案是**沒有**。逐筆看方向也是混亂的：

| `w` | 變寬（比值>1.2） | 變窄（<0.83） | 大致不變 |
|---|---|---|---|
| 1 dt | 2 | 3 | 3 |
| 3 dt | 2 | 4 | 2 |
| 10 dt | 1 | 4 | 2 |
| 30 dt | 4 | 3 | 1 |

**最直接的反證**：整組裡最尖的那個峰（seed 20260907）在每一個平滑寬度下都還是針——
半寬 6.91×10⁻⁵ dex（階躍）→ 3.81×10⁻⁵（1 dt）→ 5.27×10⁻⁵（3 dt）→ 4.66×10⁻⁵（10 dt）
→ 2.83×10⁻⁵（30 dt），佔先驗跨度 2.0×10⁻⁵ 降到 8.3×10⁻⁶。**拿掉不連續之後它反而更窄。**

固定 1e-7 步長的 Δ(lnL) 也沒有給出支持：8 筆中只有 seed 20260907 有明顯梯度
（−8.41×10⁻³ 於階躍，−1.62×10⁻³ 於 30 dt），其餘 7 筆在所有寬度下都是 0 或 ±1.5×10⁻⁵ 量級。
後者在 |lnL| ~ 10¹¹ 時的 float64 ulp 正好是 1.53×10⁻⁵，所以那些數字是數值底噪、不是訊號——
1e-7 dex 的步長對半寬 ~10⁻³ dex 的峰本來就太小，只有對針狀峰才有鑑別力。

## F.3 附帶更正 §E.3 的兩個數字

§E.3 的半寬是用 ±0.15 dex 視窗、2001 點的等距網格量的，兩端都有問題，本節用更寬的視窗與
bisection 重量：

| §E.3 記載 | 本節重量 | 原因 |
|---|---|---|
| 半寬中位數 0.0164 dex | **0.0036 dex** | 舊視窗把寬峰右設限成 NaN，取中位數時被排除，中位數被往上拉 |
| 最窄 1.5×10⁻⁴ dex（先驗的 4.4×10⁻⁵） | **6.9×10⁻⁵ dex（先驗的 2.0×10⁻⁵）** | 舊網格解析度就是 1.5×10⁻⁴ dex，最窄的峰是被解析度卡住而非真的那麼寬 |

§E 的結論不受影響（那裡的重點是 taper 0.0/0.1 的**比值**≈1，兩邊用同一個量測方法，
偏差同向抵消），但這兩個絕對數字以本節為準。

## F.4 平滑化對波形頻域內容的影響（要求 4）

平滑化**確實明顯改變**爆發波形本身的頻譜，不是免費的：

| `w` | band 內能量（相對階躍） | 1700 Hz 以上的功率佔比 | 2×f_burst 以上的功率佔比 |
|---|---|---|---|
| 0（階躍） | 1.0000 | 8.67×10⁻³ | **0.180** |
| 1 dt | 0.9896 | 4.41×10⁻⁵ | 0.0856 |
| 3 dt | 0.9157 | 8.54×10⁻⁷ | 0.0384 |
| 10 dt | 0.8140 | 1.23×10⁻⁶ | 7.06×10⁻³ |
| 30 dt | 0.8142 | 1.19×10⁻⁶ | 6.64×10⁻⁴ |

也就是說：階躍起始確實把爆發約 **18% 的功率**放到 2 倍標稱頻率以上、**0.87%** 放到分析頻帶
（1700 Hz）以上；平滑化把這些寬頻內容拿掉，代價是 `w ≥ 10 dt` 時 band 內能量掉約 19%。
**所以「階躍在頻域上是顯著的」這件事成立——但它不是決定時序峰寬的因素。**

## F.5 一句話結論

**機制不成立：把 ringdown 起始從階躍換成 1–30 個取樣點寬的平滑過渡，`log10_dt_bounce_s`
的 lnL 峰半寬中位數在 0.0027–0.0043 dex 之間來回（相對階躍 0.73–1.20 倍），沒有隨過渡寬度
單調變化，逐筆方向也混亂；整組最尖的那個峰在平滑化後反而更窄（2.0×10⁻⁵ → 8.3×10⁻⁶ 的
先驗佔比）。階躍不連續在頻域上確實顯著（18% 的爆發功率在 2×f_burst 以上），但它不是
Part D 崩潰的成因。因為機制未成立，本節不給出任何「合理過渡寬度」的建議，
也沒有據此修改任何 production 程式碼。**

窄峰的真正來源仍未確定。目前已被量測排除的有兩項：發現 (C) 的 likelihood 尺度膨脹（§E.2）、
以及本節的起始不連續。

## F.6 產出檔案

| 路徑 | 內容 | 是否進版控 |
|---|---|---|
| `docs/calibration/bounce_onset_smoothing_halfwidths.csv` | 8 筆注入 × 5 個過渡寬度的峰半寬、Δ(lnL)、設限記錄 | 是 |
| `docs/calibration/bounce_onset_smoothing_spectra.csv` | 同一組的爆發波形頻域能量分布 | 是 |
