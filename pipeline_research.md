# Hugging Face Transformers Pipeline調査結果

## 1. Pipelineで行えることの一覧

### テキストタスク
- `sentiment-analysis` - 感情分析（ポジティブ/ネガティブ）
- `text-classification` - 汎用テキスト分類
- `zero-shot-classification` - ゼロショット分類
- `ner` - 固有表現認識
- `question-answering` - 質問応答
- `fill-mask` - マスク埋め
- `summarization` - 要約
- `text-generation` - テキスト生成
- `text2text-generation` - テキスト変換
- `translation` - 翻訳
- `feature-extraction` - 特徴抽出

### 音声タスク
- `automatic-speech-recognition` - 音声認識
- `text-to-audio` - テキスト音声変換

### 画像タスク
- `image-classification` - 画像分類
- `image-segmentation` - 画像セグメンテーション
- `object-detection` - 物体検出

## 2. Sentiment Analysis（日本語対応モデル）

### 主要な日本語センチメント分析モデル

#### 1. daigo/bert-base-japanese-sentiment
```python
from transformers import pipeline
jp_sentiment = pipeline(
    "sentiment-analysis",
    model="daigo/bert-base-japanese-sentiment",
    tokenizer="daigo/bert-base-japanese-sentiment"
)
result = jp_sentiment("私は幸福である。")
# [{'label': 'ポジティブ', 'score': 0.98430425}]
```

#### 2. tabularisai/multilingual-sentiment-analysis (2024年12月)
- 22以上の言語対応（日本語含む）
- 5段階評価（Very Negative ～ Very Positive）
- 精度: 約93%
```python
pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
```

#### 3. 特殊用途モデル
- `christian-phu/bert-finetuned-japanese-sentiment` - Amazon レビュー特化
- `kit-nlp/bert-base-japanese-sentiment-irony` - 皮肉・皮肉検出対応
- `jarvisx17/japanese-sentiment-analysis` - chABSAデータセット利用

## 3. Zero-Shot Classification（日本語対応モデル）

### 主要な多言語ゼロショット分類モデル

#### 1. MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
- 27言語対応（日本語含む）
- 日本語精度: 75.9%（XNLI）
- 270万以上の学習データ

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", 
                     model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")

# 日本語テキストの分類
text = "最新のスマートフォンは高性能なカメラを搭載しています"
labels = ["技術", "政治", "スポーツ", "エンターテインメント"]

result = classifier(text, labels, multi_label=False)
```

#### 2. joeddav/xlm-roberta-large-xnli
- 100言語対応
- XLNIデータセット利用
- 多言語対応のRoBERTaベース

```python
classifier = pipeline("zero-shot-classification", 
                     model="joeddav/xlm-roberta-large-xnli")
```

### 決算短信分析への応用例

```python
# ゼロショット分類の例
text = "当期の業績は予想を上回り、売上高は前年同期比20%増となりました"
labels = ["業績好調", "業績不調", "業績横ばい", "将来見通し不透明"]

# センチメント分析と組み合わせた使用
sentiment_result = jp_sentiment(text)
classification_result = classifier(text, labels)
```

## 4. 重要モデル詳細分析

### 日本語センチメント分析
- **daigo/bert-base-japanese-sentiment**: 最も人気、日本語特化、軽量（422MB）、高精度
- **tabularisai/multilingual-sentiment-analysis**: 2024年12月の最新、5段階評価、21言語対応

### ゼロショット分類  
- **MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7**: 日本語75.9%の精度、100言語対応
- 決算短信向けに「業績好調」「業績不調」等のカスタムラベル設定が可能

### 各モデルの決算短信適用評価

#### 1. daigo/bert-base-japanese-sentiment
- **強み**: 日本語特化、軽量（422MB）、高精度
- **弱み**: バイナリ分類のみ、金融専門用語への対応限定的
- **決算短信適用**: 日本企業の決算短信に適用可能だが、金融データでの追加学習推奨

#### 2. tabularisai/multilingual-sentiment-analysis
- **強み**: 21言語対応、5段階評価、2024年12月の最新モデル
- **弱み**: 商用利用に制限、合成データ学習のため実世界との乖離可能性
- **決算短信適用**: 多言語企業や細かい感情分析に最適、ライセンス確認必要

#### 3. MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
- **強み**: 100言語対応、ゼロショット分類、MIT ライセンス、カスタムラベル可能
- **弱み**: 大きなモデルサイズ（276M）、計算リソース要求高め
- **決算短信適用**: 最も柔軟性が高く、「業績好調」「業績不調」等の独自分類が可能

## 5. 2024年の最新動向

1. **多言語モデルの改善**: アジア言語（日本語含む）のサポート強化
2. **ドメイン特化**: Amazon レビュー、SNS、ニュース等の特定分野向けモデル増加
3. **統合モデル**: センチメント分析とゼロショット分類を組み合わせた高度な分析が可能に