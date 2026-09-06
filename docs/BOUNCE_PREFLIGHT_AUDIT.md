# `discrete_uniform` 先驗映射修正，與 `bounce` 模型校準前置稽核

## 範圍聲明

**這份文件記錄的是 (Part A) 一個先驗映射 bug 的修正，與 (Part B) 對 `bounce` 模型執行
SBC/coverage 之前的前置稽核結果。不構成任何白洞訊號偵測或未偵測的科學宣稱，也不包含任何
天文物理結論。**

WhiteSearch 是 candidate ranking engine（候選訊號排序引擎），不是白洞證明器。所有資料皆為
mock 模擬資料，未使用任何真實 GWOSC 觀測資料。

**Part B 的 SBC/coverage campaign 尚未執行**——前置稽核發現了與
`docs/BH_RINGDOWN_SBC_COVERAGE_REPORT.md` 發現 (B)/(D) 同等級的阻斷性問題，依 fail-closed
原則先如實回報現況與修法選項，等決策後另開一輪處理，不自行選一個方向就動手改。

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
