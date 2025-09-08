# Statement of Work (SOW) - Transformer決算短信分析ツール

## プロジェクト概要
決算短信補足資料に対してTransformerモデルを用いた感情分析を実行するPythonツールの実装

## 実装スコープ

### Phase 1: プロジェクト基盤構築 🔧
- [ ] uvプロジェクト初期化（pyproject.toml作成）
- [ ] 依存関係定義（transformers, torch, etc.）
- [ ] 開発環境セットアップ

### Phase 2: コア感情分析機能 🧠
- [ ] SentimentAnalyzerクラスの実装
  - [ ] daigoモデル対応（日本語特化）
  - [ ] tabularisaiモデル対応（多言語・5段階）
  - [ ] moritzlauerモデル対応（ゼロショット分類）
- [ ] デバイス管理機能（GPU/CPU自動選択）
- [ ] スコア正規化機能（0-10統一スケール）

### Phase 3: ファイル処理システム 📁
- [ ] FileProcessorクラスの実装
  - [ ] 決算短信ファイル自動検出
  - [ ] 証券コード別フォルダ処理
  - [ ] ファイル名からメタデータ抽出
- [ ] マークダウンファイル読み込み

### Phase 4: 出力・結果管理 📊
- [ ] OutputManagerクラスの実装
  - [ ] CSV形式出力
  - [ ] JSON形式出力
  - [ ] 統一結果フォーマット
- [ ] 進捗表示とログ機能

### Phase 5: 統合・CLI機能 ⚙️
- [ ] メイン処理の統合
- [ ] コマンドライン引数処理
- [ ] エラーハンドリング
- [ ] ヘルプ機能

### Phase 6: テスト・検証 🧪
- [ ] 小規模データでの動作テスト
- [ ] GPU/CPU動作確認
- [ ] 3モデル全体の動作確認
- [ ] 出力データの検証

## 技術仕様

### 対応モデル
1. **daigo/bert-base-japanese-sentiment** - 日本語特化、軽量、2値分類
2. **tabularisai/multilingual-sentiment-analysis** - 多言語対応、5段階評価
3. **MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7** - ゼロショット分類

### 入出力仕様
- **入力**: attachments/フォルダ内の決算短信マークダウンファイル
- **出力**: CSV/JSON形式の感情分析結果
- **統一スコア**: 0-10の11段階評価

### システム要件
- Python 3.8+
- uv（パッケージ管理）
- GPU: 任意（自動検出・選択）

## 成果物
- [ ] `transformer.py` - メインプログラム
- [ ] `pyproject.toml` - 依存関係定義
- [ ] `sow.md` - 本文書
- [ ] テスト結果レポート

## 進捗追跡
- ✅ 完了
- 🔄 進行中  
- ⏸️ 待機中
- ❌ 未着手

## 現在のステータス: Phase 1 開始準備完了

次のアクション: uvプロジェクト初期化とpyproject.toml作成