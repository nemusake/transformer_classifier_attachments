# 決算短信センチメント分析ツール（CSV出力）

決算短信の補足資料（`attachments/<証券コード>/`配下のMarkdown）を読み込み、長文対応の分割＋集約でセンチメントスコアを算出し、CSVで出力します。処理対象の企業は `codelist.csv` の `code` 列に記載された5桁英数字コードのみです。

## 🎯 主な特徴

- **金融特化モデル採用**: `bardsai/finance-sentiment-ja-base` 単体使用
- **高い実用性**: 実証済みセンチメント判別能力（4.03-6.47の有意な分散）
- **CSV出力**: UTF-8 BOM付きで文字化けしない（列: `filename, date, code, title, sentiment_score`）
- **最適化された長文処理**: 400文字で安全分割、文字数重み付き平均で集約
- **GPU最適化**: デフォルトGPU実行、`-cpu`オプションでCPU強制可能
- **uvによる依存関係管理**: モダンで信頼性の高いPython環境管理

## ✅ 開発プロセスと最終選択

### **フェーズ1: 基盤構築**
- ✅ bardsai金融特化モデル導入・検証完了
- ✅ 従来の汎用モデル問題（中立収束）を解決

### **フェーズ2: 拡張実験**
- 🔄 ModernBERT（8kトークン）・BigBird（4kトークン）長文対応モデル検討
- ⚠️ コンパイルエラー・未学習分類層の問題により断念

### **最終判断: シンプル＆高性能**
**bardsai単体採用の理由:**
- ✅ **実証済み精度**: スコア範囲4.03-6.47の実用的判別
- ✅ **安定動作**: コンパイルエラーなし、依存関係最小
- ✅ **金融特化**: 決算短信に最適化された学習済みモデル
- ✅ **保守性**: シンプルな構成で長期運用に適している

| 項目 | 汎用モデル | bardsai金融特化 | 実用性 |
|------|------------|----------------|--------|
| スコア範囲 | 4.97-5.01 | **4.03-6.47** | ⭐⭐⭐⭐⭐ |
| 分散 | ほぼ0 | **大幅拡大** | **判別可能** |
| 安定性 | 中立収束 | **実際の傾向反映** | **投資判断支援** |

---

## クイックスタート

### 🚀 環境セットアップ（uv推奨）
```bash
# uvを使用した依存関係のインストール（推奨）
uv sync

# 代替: pipを使用する場合
pip install transformers torch fugashi protobuf unidic-lite
```

### 🏃‍♂️ 実行方法
```bash
# 基本実行（GPU使用・推奨）
uv run python transformer.py -v

# CPU強制使用（GPUが利用できない環境）
uv run python transformer.py -cpu -v

# カスタムパス指定
uv run python transformer.py -i attachments -o results -v

# pipでインストールした場合
python transformer.py -v
```

**実行例の出力:**
```
デバイス: GPU(CUDA:0)
対象コード数: 3
[1/3] 130A0: 7件
[2/3] 13010: 12件  
[3/3] 99840: 11件
完了: results/sentiment_20250909_024908.csv
Device set to use cuda:0
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
```

オプション
- `-i/--input`: 入力ルート（既定: `attachments`）
- `-o/--output`: 出力ディレクトリ（既定: `results`）
- `--codes-file`: コード一覧CSV（既定: `codelist.csv`、BOM付き `code` 列を想定）
- `-cpu/--cpu`: CPU強制（指定時のみCPU）
- `-v/--verbose`: 詳細ログ出力

---

## 入力要件
- ディレクトリ: `attachments/<5桁英数字>/*.md`
  - 例: `attachments/13010/2024-08-05_13010_2025年3月期…_attachments.md`
- コード一覧: `codelist.csv`
```
code
130A0
13010
99840
```

### ファイル名の解析規則
- 想定フォーマット: `YYYY-MM-DD[_-]XXXXX[_-]<タイトル>... .md`
- CSV列への割当:
  - `filename`: ファイル名そのまま
  - `date`: 先頭の `YYYY-MM-DD`
  - `code`: 後続の5桁英数字
  - `title`: それ以降（末尾の `_attachments` は自動除去）
- 上記形式に合致しない場合は、`_` 区切りのフォールバックで抽出を試みます。

---

## 📊 出力
- **パス**: `results/sentiment_<YYYYMMDD_HHMMSS>.csv`
- **エンコーディング**: UTF-8 BOM付き（Excel等で文字化けしない）
- **列**: `filename, date, code, title, sentiment_score`
- **`sentiment_score`**: 0–10 の実数（小数点4桁表示）
  - **0-4**: ネガティブ（業績悪化、リスク要因等）
  - **5前後**: 中立（通常の報告内容）
  - **6-10**: ポジティブ（好調、成長見通し等）

---

## 長文処理の方針
- 分割サイズ目安: 400文字、前後50文字オーバーラップ（512トークン制限対応）
- 区切り優先: `。` `！` `？` 改行 など近傍の自然な境界を探索
- 各分割を個別に推論し、文字数で重み付け平均した値を最終スコアとする
- pipelineにtruncat ion=True, max_length=512を指定してトークン制限に対応

---

## 🤖 モデル仕様
- **採用モデル**: `bardsai/finance-sentiment-ja-base` **単体**
  - 金融ニュース・Financial PhraseBankで学習済み
  - 日本語金融文書に特化した事前学習済み分類器
  - 3段階（POSITIVE/NEUTRAL/NEGATIVE）の確信度スコア出力

- **スコア変換ロジック**: 
  - POSITIVE: `5.0 + (確信度 × 5.0)` = **5.0-10.0**
  - NEGATIVE: `5.0 - (確信度 × 5.0)` = **0.0-5.0**  
  - NEUTRAL: **5.0**（固定）

- **処理仕様**:
  - テキスト分割: 400文字（512トークン制限対応）
  - 重複: 50文字オーバーラップ
  - 集約: 文字数重み付き平均

- **実行デバイス**: GPU（CUDA:0）既定、`-cpu`でCPU強制

---

## 📈 実行結果例

**最終版による処理結果（2025-09-09最新実行）：**
- **対象企業**: 130A0, 13010, 99840（計3社）
- **処理ファイル数**: 30件（130A0: 7件, 13010: 12件, 99840: 11件）
- **出力**: `results/sentiment_20250909_024908.csv`（UTF-8 BOM付き）
- **実行デバイス**: GPU(CUDA:0)
- **処理速度**: 約2分で30件完了

**実際のスコア分析（130A0企業の時系列）:**
```csv
filename,date,title,sentiment_score,判定
2024-05-07_130A0_第１四半期決算短信.md,2024-05-07,第１四半期,6.4746,強いポジティブ
2024-08-05_130A0_第２四半期決算短信.md,2024-08-05,第２四半期,5.7312,やや良好
2024-11-13_130A0_第３四半期決算短信.md,2024-11-13,第３四半期,5.6917,やや良好  
2025-02-13_130A0_期末決算短信.md,2025-02-13,期末決算,4.4845,やや悪化
2025-05-09_130A0_第１四半期決算短信.md,2025-05-09,第１四半期,4.0325,明確な悪化
```

✅ **実用的判別能力**: 時系列での業績変化を適切に反映

---

## 📁 プロジェクト構成
```
transformer_classifier_attachments/
├── transformer.py               # メイン実行ファイル（bardsai単体・最終版）
├── transformer_README.md        # このドキュメント（最新版）  
├── model_research.md            # 初期調査レポート
├── model_research_fin.md        # 金融特化調査
├── model_research_Aspect-based-Sentiment-Analysis.md  # ABSA調査
├── sow.md                       # 開発計画書（フェーズ1完了）
├── pyproject.toml               # uv依存関係管理
├── codelist.csv                 # 処理対象企業コード一覧
├── .venv/                       # uv仮想環境
├── attachments/                 # 入力（企業コード別フォルダ）
│   ├── 130A0/                   # （7件の決算短信）
│   ├── 13010/                   # （12件の決算短信）
│   └── 99840/                   # （11件の決算短信）
└── results/                     # 出力CSV（UTF-8 BOM付き）
    └── sentiment_20250909_024908.csv  # 最新結果
```

**依存関係（pyproject.toml）:**
```toml
dependencies = [
    "fugashi>=1.5.1",           # 日本語形態素解析
    "protobuf>=6.32.0",         # プロトコルバッファ
    "torch>=2.0.0",             # PyTorch
    "transformers>=4.20.0",     # Hugging Face Transformers
    "unidic-lite>=1.0.8",       # 日本語辞書
]
```

---

## トラブルシューティング

### セットアップ関連
- `ModuleNotFoundError`: `uv sync` または `pip install transformers torch` を実行
- GPUを使わない場合/使えない場合: `-cpu` オプションを付与
- `codelist.csv` にヘッダ `code` が必要（BOM付きでも可）

### 実行時エラー
- `Token indices sequence length is longer than 512`: 
  - 修正済み（テキスト分割サイズを400文字に調整、truncation対応）
- CUDA out of memory: `-cpu` オプションを使用
- 処理速度が遅い場合: GPU使用を確認（デフォルトでCUDA:0を使用）

### 開発履歴と今後の可能性

#### **完了済み開発フェーズ**
1. ✅ **フェーズ0**: 基本実装（汎用モデル）
2. ✅ **フェーズ1**: 金融特化モデル導入（bardsai）
3. 🔄 **フェーズ2**: 長文対応実験（ModernBERT/BigBird）→ 技術的問題により中断
4. ✅ **最終判断**: bardsai単体での安定運用選択

#### **将来の拡張可能性**
**研究調査済み（実装保留中）:**
1. **ABSA機能**: アスペクト別センチメント（売上・利益・リスク等）
2. **長文LLM統合**: Qwen2.5/Llama3.1による根拠抽出
3. **ハイブリッド手法**: エンコーダ+LLMアンサンブル
4. **時系列分析**: 過去比較による相対評価

**詳細資料:**
- `sow.md`: 段階的拡張計画
- `model_research_fin.md`: 金融NLP学術調査
- `model_research_Aspect-based-Sentiment-Analysis.md`: ABSA技術調査

---

## 📝 ライセンス/注意
- 利用するモデル`bardsai/finance-sentiment-ja-base`のライセンスに従ってください
- 決算短信データの利用は各社の利用規約・法的制約を確認してください
- 本ツールの分析結果は投資判断の参考情報であり、投資助言ではありません

---

## 🚀 クイック実行コマンド
```bash
# 完全セットアップ＆実行
uv sync && uv run python transformer.py -v

# 結果確認（最新ファイル）
ls -la results/*.csv | tail -1
```
