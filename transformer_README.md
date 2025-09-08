# Transformer決算短信分析ツール

決算短信補足資料のセンチメント分析とゼロショット分類を行うPythonツール。

## 実装状況

### ✅ 実装完了
#### Phase 1: プロジェクト基盤構築
- **プロジェクト基盤**: uvプロジェクト初期化、依存関係管理
- **パッケージ管理**: pyproject.toml設定、transformers/torch等インストール
- **開発環境**: Black/Flake8等の開発ツール設定

#### Phase 2: コア機能実装・テスト
- **11段階スコア解釈**: 0-10の統一スコアから詳細解釈生成
- **直接フォルダ指定**: `attachments/13010/` のような個別企業フォルダ指定対応
- **ファイル処理**: 決算短信の自動検出・抽出、メタデータ抽出
- **出力機能**: CSV/JSON形式対応、完璧なデータ構造
- **GPU/CPU選択**: 自動デバイス検出・選択
- **小規模動作確認**: 13010企業（12ファイル）での処理確認完了

#### モデル対応状況
- **tabularisai**: 多言語対応、正常ロード・処理確認済み ✅
- **daigo**: 認証エラーで利用不可 ❌
- **moritzlaurer**: ゼロショット分類、未テスト ⏸️

### 🔄 現在の課題
- **テキスト長制限**: 決算短信が512トークンを超過してエラー発生
- **実際の感情分析**: すべて「分析エラー」で実際のスコアが取得できない状態

### ⏸️ 次の実装予定
- **Phase 2A**: テキスト分割機能（512トークン以下にチャンク分割）
- **他モデル検証**: moritzlauerモデルでの動作確認
- **代替モデル**: daigoの代替となる日本語モデルの検討

## システム要件

- Python 3.8.1+
- GPU: 任意（自動検出、CPU fallback対応）
- メモリ: 8GB以上推奨
- uv: Pythonパッケージ管理ツール

## セットアップ

### 1. 依存関係のインストール
```bash
# uv使用（推奨）
uv sync

# または手動インストール
pip install transformers torch datasets accelerate tokenizers
```

### 2. 開発環境のセットアップ
```bash
# 開発ツールのインストール
uv sync --dev

# コード品質チェック
uv run black transformer.py
uv run flake8 transformer.py
```

## 使用方法

### 基本的な使用方法
```bash
# uv環境での実行（推奨）
uv run python transformer.py sentiment daigo -i attachments/ -o results/

# または直接実行
python transformer.py sentiment daigo -i attachments/ -o results/
```

### 各モデルでの実行
```bash
# 日本語特化軽量モデル
uv run python transformer.py sentiment daigo -i attachments/ -o results/

# 多言語対応5段階評価モデル
uv run python transformer.py sentiment tabularisai -i attachments/ -o results/

# ゼロショット分類モデル
uv run python transformer.py sentiment moritzlaurer -i attachments/ -o results/
```

### オプション指定
```bash
# CPU強制使用
uv run python transformer.py sentiment daigo --cpu -i attachments/ -o results/

# 出力形式指定
uv run python transformer.py sentiment daigo -f csv -i attachments/ -o results/

# 詳細ログ付き
uv run python transformer.py sentiment daigo -v -i attachments/ -o results/
```

## オプション

- `-i`: 入力ディレクトリ（デフォルト: attachments/）
- `-o`: 出力ディレクトリ（デフォルト: results/）
- `-f`: 出力形式（csv/json/both）
- `--cpu`: CPU強制使用
- `-v`: 詳細ログ

## 出力データ

**CSV**: file_path, company_code, date, period, content_length, sentiment_score, sentiment_interpretation, model_used
**JSON**: 上記 + raw_scores, error（詳細メタデータ付き）

## モデル特徴

- **daigo**: 軽量、日本語特化、2値分類
- **tabularisai**: 多言語対応、3値分類
- **moritzlaurer**: ゼロショット、7段階分類

## テスト実行

### 動作確認済みのテスト
```bash
# 小規模データでのテスト（推奨・動作確認済み）
uv run python transformer.py sentiment tabularisai -i attachments/13010/ -o results/ -v

# 全データでのテスト（大量処理・時間がかかります）
uv run python transformer.py sentiment tabularisai -i attachments/ -o results/ -v

# CPU強制使用でのテスト
uv run python transformer.py sentiment tabularisai -i attachments/13010/ -o results/ --cpu -v
```

### その他のコマンド
```bash
# ヘルプ表示
uv run python transformer.py --help
uv run python transformer.py sentiment --help

# 出力形式指定
uv run python transformer.py sentiment tabularisai -i attachments/13010/ -o results/ -f csv
```

### ⚠️ 現在動作しないテスト
```bash
# daigoモデル（認証エラー）
uv run python transformer.py sentiment daigo -i attachments/13010/ -v

# moritzlauerモデル（未テスト）
uv run python transformer.py sentiment moritzlaurer -i attachments/13010/ -v
```

## プロジェクト構成

```
transformer_classifier_attachments/
├── pyproject.toml          # プロジェクト設定・依存関係
├── transformer.py          # メイン実行ファイル
├── transformer_README.md   # このファイル
├── sow.md                  # 実装計画書
├── pipeline_research.md    # モデル調査結果
├── CLAUDE.md              # プロジェクト方針
├── codelist.csv           # 企業証券コード一覧
└── attachments/           # 決算短信データ（入力）
    ├── 13010/             # 各企業フォルダ
    ├── 130A0/
    └── ...
```

## 次のステップ

### 🚧 現在進行中
**Phase 2A: テキスト分割機能の実装**
- 決算短信の長いテキストを512トークン以下に分割
- 各チャンクの感情分析結果を統合して最終スコアを算出
- 実際の感情分析結果取得を目指す

### 📋 今後の予定
1. **Phase 2A完了**: テキスト分割による感情分析の実現
2. **Phase 3**: moritzlauerモデルでのゼロショット分類テスト
3. **Phase 4**: daigoの代替日本語モデル検討・実装
4. **Phase 5**: 大規模データでの性能検証・最適化

### 📊 現在の成果
- **処理可能ファイル数**: 37,327ファイル（全データ）
- **動作確認済み範囲**: 単一企業（13010）の12ファイル
- **出力データ品質**: 完璧（メタデータ抽出・CSV/JSON対応）
- **課題**: テキスト長制限により実際の感情分析未実現