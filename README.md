# WhiteSearch — 白洞候選訊號搜尋與證據排序引擎

## 專案定位

**WhiteSearch 不是「證明白洞存在的程式」，而是「以多模型前向模擬、多資料通道、貝氏模型比較與嚴格驗證，搜尋並量化白洞候選訊號的證據引擎」。**

核心工作流是 **`compare` / `rank`（Bayes factor 模型比較）**，不是單看一個模型的 ln Z。

## 支援的白洞模型

| 模型 | 主要可觀測通道 | CLI `--model` |
|------|--------------|---------------|
| GR 永恆白洞 | 影像/陰影 | `gr_eternal` |
| Black-to-white bounce | 重力波 | `bounce` |
| PBH 量子穿隧白洞 | 電磁爆發 (FRB/gamma) | `pbh_tunneling` |
| 標準 BH ringdown | GW（對照） | `bh_ringdown` |
| Magnetar flare | 電磁（對照） | `magnetar` |

## 資料接入（誠實說明）

| 資料源 | 狀態 | 說明 |
|--------|------|------|
| **mock** | 完整支援 | 明確標記 `MOCK_EXPLICIT` |
| **GWOSC** | 精選事件白名單 | 目前驗證：`GW150914`, `GW151226`, `GW170814`, `GW200105`（非完整 O1–O4a 全庫） |
| **CHIME/FRB** | catalog 讀取 | 本地/下載 catalog |
| HEASARC / EHT | 開發中 | 預設 **fail-closed**；需 `--allow-mock-fallback` 才會替換 mock |

**預設行為：真資料載入失敗會直接報錯退出，不會悄悄改用 mock。**

## 安裝（分層依賴）

### 最小安裝（核心 + CLI）

```bash
pip install -e .
```

僅含：numpy, scipy, matplotlib, click, pyyaml, tqdm

### 科學工作流（建議）

```bash
pip install -e ".[science]"
```

含：astropy, astroquery, pandas, gwpy, bilby, dynesty, corner, scikit-learn, pytest

### 選用 extras

| extra | 內容 |
|-------|------|
| `astro` | astropy, astroquery, pandas, h5py |
| `gw` | gwpy |
| `inference` | bilby, dynesty |
| `gw-full` | gw + inference + pycbc |
| `viz` | corner, scikit-learn |
| `tracking` | mlflow, dvc |
| `all` | 全部 |

### Docker（Linux / 完整 GW）

```bash
docker build -t whitesearch:latest containers/
```

## 快速開始

```bash
# 安裝後直接使用（入口：whitesearch.cli）
whitesearch --help

# 核心：模型比較（Bayes factor）
whitesearch compare --model bounce --null null --alt bh_ringdown --channel gw --data mock

# 單模型 fit（會印出資料來源、sampler、是否近似證據）
whitesearch fit --model bounce --channel gw --data mock

# mock 測試時，注入模型預設等於 fit 模型
whitesearch fit --model bh_ringdown --channel gw --data mock
# inject_model=bh_ringdown（一致）

# 模型歧視測試：顯式指定不同注入模型
whitesearch fit --model bh_ringdown --channel gw --data mock --inject-model bounce

# 真實 GWOSC（需 gwpy + 網路；失敗則報錯）
whitesearch compare --model bounce --data gwosc --event GW150914 --channel gw

# 僅在明知後果時允許 mock 替換
whitesearch fit --model bounce --data gwosc --event GW150914 --allow-mock-fallback

# 多模型排名
whitesearch rank --models bounce,bh_ringdown,magnetar,null --channel gw --data mock

# 注入/回收驗證
whitesearch inject --model bounce --channel gw --n-injections 50

# 產生報告（含 provenance / fallback 警告）
whitesearch report --run-dir artifacts/compare --output artifacts/report.md
```

亦可使用：

```bash
python -m whitesearch --help
```

## CLI 輸出說明

`fit` / `compare` 會印出：

- `Fit model` / `Inject model`（mock 時）
- `Actual source used`：`GWOSC` / `MOCK_EXPLICIT` / `MOCK_FALLBACK`
- `Sampler`：`dynesty` 或 `toy_importance_sampling`
- `Approximate evidence: YES/NO`

若未安裝 bilby/dynesty，會顯示 **WARNING: 證據為近似值，不可當發表結論**。

## 目錄結構

```
src/whitesearch/
├── cli.py           # 正式 CLI 入口（whitesearch 指令）
├── models/
├── simulators/
├── dataio/          # loader.py + provenance（fail-closed）
├── likelihoods/
├── inference/
└── validation/
configs/
workflows/           # Snakemake；cli.py 為相容 shim
tests/
```

## GitHub

https://github.com/dragonheart8787/whitehole

```powershell
.\scripts\sync-github.ps1 -Message "更新說明"
```

## 授權

MIT License
