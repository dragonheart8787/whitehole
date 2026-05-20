# WhiteSearch — 白洞候選訊號搜尋與證據排序引擎

## 專案定位

**WhiteSearch 不是「證明白洞存在的程式」，而是「以多模型前向模擬、多資料通道、貝氏模型比較與嚴格驗證，搜尋並量化白洞候選訊號的證據引擎」。**

白洞在廣義相對論與量子重力文獻中有多種理論化身，但截至目前並沒有被觀測確認。本專案採用統計上可審計的方法，以 Bayes factor 作為核心 KPI，對白洞候選事件進行可重現的排名。

## 支援的白洞模型

| 模型 | 主要可觀測通道 | 核心參數 |
|------|--------------|---------|
| GR 永恆白洞 | 影像/陰影 | M, a*, Q, D_L, i |
| Black-to-white bounce | 重力波 | M, a*, τ_bounce, ℓ_q, Δ, p |
| PBH 量子穿隧白洞 | 電磁爆發 (FRB/gamma) | M, f_PBH, k, η_r, η_γ |
| 零假設 + 替代模型 | 三通道 | magnetar, GRB, BH 吸積 |

## 三通道資料接入

| 資料源 | 通道 | 接入工具 |
|--------|------|---------|
| GWOSC (O1–O4a) | 重力波 | GWPy |
| CHIME/FRB Catalog | 電磁爆發 | cfod / astropy |
| HEASARC | X-ray/gamma | astroquery |
| Chandra Archive | X-ray | CIAO |
| XMM-Newton Archive | X-ray | XMM-SAS |
| EHT 2017 L1 | 影像/VLBI | ehtim (Phase 2) |

## 核心統計框架

```
Z_m = ∫ p(d|θ,m) p(θ|m) dθ         (模型證據)
BF_10 = Z_1 / Z_0                    (Bayes factor)
```

使用 Bilby + dynesty nested sampling 同時估計 posterior 與 evidence。

### 驗收門檻

| 指標 | 內部升級 | 可投稿 |
|------|---------|-------|
| ln BF (vs 零假設) | > 3 | > 5 |
| ln BF (vs 最佳替代) | > 1 | > 3 |
| 假陽性率 | < 5% | < 1e-3 |
| 靈敏度 (50% 回收率) | 目標區 | — |
| 靈敏度 (90% 回收率) | — | 目標區 |
| SBC 覆蓋率 | 80–100% | 85–95% |
| 可重現性 | < 0.5 nat | < 0.2 nat |

## 安裝

### 基礎安裝（純 numpy/scipy/astropy）

```bash
pip install -e ".[dev]"
```

### 含 Bayesian 推論（建議）

```bash
pip install -e ".[inference,dev]"
```

### 含 GW 工具（Linux 限定）

```bash
pip install -e ".[inference,gw,dev]"
```

### 完整安裝（建議使用 Docker）

```bash
docker build -t whitesearch:latest containers/
docker run --rm -it whitesearch:latest bash
```

> **注意**：PyCBC、LALSuite、ehtim 等套件在 Windows 上需透過 Docker 或 WSL2 使用。

## GitHub 同步

遠端儲存庫：[https://github.com/dragonheart8787/whitehole](https://github.com/dragonheart8787/whitehole)

```powershell
# 手動同步（add + commit + push）
.\scripts\sync-github.ps1 -Message "描述此次更新"

# 安裝「每次 commit 後自動 push」（只需一次）
.\scripts\install-git-hooks.ps1
```

一般流程：`git add` → `git commit` →（若已安裝 hook 會自動 push）或執行 `sync-github.ps1`。

## 快速開始

```bash
# 執行 mock injection/recovery 測試
whitesearch inject --model bounce --channel gw --n-injections 100

# 對 GWOSC 公開資料計算 Bayes factor
whitesearch fit --model bounce --data gwosc --event GW150914

# 生成靈敏度曲線
whitesearch sensitivity --model pbh_tunneling --channel radio

# 一鍵重跑完整 MVP 管線
snakemake --snakefile workflows/Snakefile --cores 4
```

## 目錄結構

```
白洞程式/
├── src/whitesearch/
│   ├── models/          # 白洞模型 + 替代模型定義
│   ├── simulators/      # 三通道前向模擬器（影像/GW/EM）
│   ├── dataio/          # 六大資料源接口
│   ├── preprocess/      # 品質旗標、濾波、校準
│   ├── likelihoods/     # 通道別 + 聯合 likelihood
│   ├── inference/       # Bilby/dynesty/PyCBC wrappers
│   ├── validation/      # SBC、PPC、注入/回收
│   ├── surrogates/      # Surrogate emulator
│   └── utils/           # 物理常數與數學工具
├── configs/             # YAML priors、run config
├── workflows/           # Snakemake + CLI 管線
├── tests/               # pytest 單元 + 整合測試
├── containers/          # Dockerfile + Apptainer .def
├── notebooks/           # EDA 與報告圖表
└── artifacts/           # 版本化輸出（DVC 管理）
```

## 可重現性保證

- 所有結果以固定隨機種子 + 固定容器雜湊重跑
- 資料版本以 DVC 追蹤；實驗以 MLflow 記錄
- 同容器重跑證據差異應 < 0.2 nat（可投稿門檻）

## 授權

MIT License
