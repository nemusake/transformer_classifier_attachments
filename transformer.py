#!/usr/bin/env python3
"""
Transformer決算短信分析ツール

決算短信補足資料のセンチメント分析とゼロショット分類を行うPythonツール。

使用方法:
python transformer.py sentiment <model> -i <input_dir> -o <output_dir> [options]

モデル選択:
- daigo: 日本語特化センチメント分析
- tabularisai: 多言語センチメント分析（5段階）
- moritzlaurer: ゼロショット分類
"""

import argparse
import sys
import os
import json
import csv
import glob
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    from transformers import pipeline
    import torch
except ImportError as e:
    print(f"エラー: 必要なライブラリがインストールされていません: {e}")
    print("pip install transformers torch を実行してください")
    sys.exit(1)


class SentimentAnalyzer:
    """センチメント分析クラス"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.models = {
            'daigo': {
                'name': 'daigo/bert-base-japanese-sentiment',
                'task': 'sentiment-analysis',
                'description': '軽量、日本語特化、2値分類'
            },
            'tabularisai': {
                'name': 'tabularisai/multilingual-sentiment-analysis',
                'task': 'text-classification',
                'description': '多言語対応、3値分類'
            },
            'moritzlaurer': {
                'name': 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
                'task': 'zero-shot-classification',
                'description': 'ゼロショット、7段階分類'
            }
        }
        self.pipeline = None
        self.current_model = None
        self.device = None
    
    def log(self, message: str):
        """詳細ログ出力"""
        if self.verbose:
            print(f"[LOG] {message}")
    
    def setup_device(self, force_cpu: bool = False) -> int:
        """デバイス設定"""
        if force_cpu:
            device = -1
            self.log("CPU使用を強制")
        else:
            if torch.cuda.is_available():
                device = 0
                gpu_name = torch.cuda.get_device_name()
                self.log(f"GPU使用: {gpu_name}")
            else:
                device = -1
                self.log("GPUが利用できないため、CPUを使用")
        
        self.device = device
        return device
    
    def load_model(self, model_name: str):
        """指定されたモデルをロード"""
        if model_name not in self.models:
            raise ValueError(f"サポートされていないモデルです: {model_name}")
        
        model_info = self.models[model_name]
        self.log(f"モデルロード中: {model_info['name']} ({model_info['description']})")
        
        try:
            if model_name == 'moritzlaurer':  # ゼロショット分類
                self.pipeline = pipeline(
                    model_info['task'],
                    model=model_info['name'],
                    device=self.device
                )
            else:
                self.pipeline = pipeline(
                    model_info['task'],
                    model=model_info['name'],
                    device=self.device
                )
            
            self.current_model = model_name
            self.log("モデルロード完了")
            
        except Exception as e:
            print(f"エラー: モデルのロードに失敗しました: {e}")
            sys.exit(1)
    
    def normalize_score(self, raw_result: Any, model_name: str) -> float:
        """モデル出力を0-10の統一スコアに変換"""
        try:
            if model_name == 'daigo':
                # 2値分類 (POSITIVE/NEGATIVE) -> 0-10
                if isinstance(raw_result, list) and len(raw_result) > 0:
                    result = raw_result[0]
                    if result['label'] in ['POSITIVE', 'ポジティブ']:
                        return 5.0 + (result['score'] * 5.0)  # 5.0-10.0
                    else:
                        return 5.0 - (result['score'] * 5.0)  # 0.0-5.0
                        
            elif model_name == 'tabularisai':
                # 5段階分類 -> 0-10
                if isinstance(raw_result, list) and len(raw_result) > 0:
                    result = raw_result[0]
                    label_map = {
                        'Very Negative': 0.0,
                        'Negative': 2.5,
                        'Neutral': 5.0,
                        'Positive': 7.5,
                        'Very Positive': 10.0
                    }
                    return label_map.get(result['label'], 5.0)
                    
            elif model_name == 'moritzlaurer':
                # ゼロショット分類 -> 0-10
                if 'scores' in raw_result and len(raw_result['scores']) > 0:
                    # 最高スコアのラベルに基づく
                    top_label = raw_result['labels'][0]
                    top_score = raw_result['scores'][0]
                    
                    label_map = {
                        '業績好調': 8.0,
                        '業績横ばい': 5.0,
                        '業績不調': 2.0,
                        '将来見通し不透明': 4.0
                    }
                    base_score = label_map.get(top_label, 5.0)
                    # 信頼度で調整
                    return base_score + (top_score - 0.5) * 2.0
                    
        except Exception as e:
            self.log(f"スコア正規化エラー: {e}")
        
        return 5.0  # デフォルト
    
    def get_sentiment_interpretation(self, score: float) -> str:
        """統一スコア（0-10）から11段階解釈を生成"""
        if score >= 10.0:
            return "極めてポジティブ"
        elif score >= 9.0:
            return "非常にポジティブ"
        elif score >= 8.0:
            return "かなりポジティブ"
        elif score >= 7.0:
            return "ポジティブ"
        elif score >= 6.0:
            return "やや ポジティブ"
        elif score >= 5.0:
            return "中立"
        elif score >= 4.0:
            return "やや ネガティブ"
        elif score >= 3.0:
            return "ネガティブ"
        elif score >= 2.0:
            return "かなりネガティブ"
        elif score >= 1.0:
            return "非常にネガティブ"
        else:
            return "極めてネガティブ"
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """テキストのセンチメント分析"""
        if self.pipeline is None:
            raise ValueError("モデルがロードされていません")
        
        try:
            if self.current_model == 'moritzlaurer':
                # 決算短信向けラベル
                labels = ["業績好調", "業績不調", "業績横ばい", "将来見通し不透明"]
                raw_result = self.pipeline(text, labels)
            else:
                raw_result = self.pipeline(text)
            
            # 統一スコアに変換
            unified_score = self.normalize_score(raw_result, self.current_model)
            interpretation = self.get_sentiment_interpretation(unified_score)
            
            return {
                'raw_scores': raw_result,
                'sentiment_score': unified_score,
                'sentiment_interpretation': interpretation,
                'model_used': self.models[self.current_model]['name'],
                'error': None
            }
            
        except Exception as e:
            self.log(f"分析エラー: {e}")
            return {
                'raw_scores': None,
                'sentiment_score': 5.0,
                'sentiment_interpretation': "分析エラー",
                'model_used': self.models[self.current_model]['name'],
                'error': str(e)
            }


class FileProcessor:
    """ファイル処理クラス"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def log(self, message: str):
        """詳細ログ出力"""
        if self.verbose:
            print(f"[FILE] {message}")
    
    def find_markdown_files(self, input_dir: str) -> List[Dict[str, str]]:
        """マークダウンファイルを検出・抽出"""
        files = []
        input_path = Path(input_dir)
        
        if not input_path.exists():
            self.log(f"入力ディレクトリが存在しません: {input_dir}")
            return files
        
        # 直接証券コードフォルダが指定された場合をチェック
        if re.match(r'^[A-Za-z0-9]{5}$', input_path.name):
            # 直接証券コードフォルダが指定された場合
            company_code = input_path.name
            self.log(f"証券コード {company_code} を直接処理中")
            
            # マークダウンファイルを検索
            for md_file in input_path.glob("*.md"):
                # ファイル名から日付と期間を抽出
                file_info = self.extract_file_info(md_file.name, company_code)
                file_info['file_path'] = str(md_file)
                files.append(file_info)
        else:
            # 通常のattachmentsディレクトリが指定された場合
            # 証券コードフォルダ（5桁）を検索
            for company_dir in input_path.iterdir():
                if company_dir.is_dir() and re.match(r'^[A-Za-z0-9]{5}$', company_dir.name):
                    company_code = company_dir.name
                    self.log(f"証券コード {company_code} を処理中")
                    
                    # マークダウンファイルを検索
                    for md_file in company_dir.glob("*.md"):
                        # ファイル名から日付と期間を抽出
                        file_info = self.extract_file_info(md_file.name, company_code)
                        file_info['file_path'] = str(md_file)
                        files.append(file_info)
        
        self.log(f"検出されたマークダウンファイル数: {len(files)}")
        return files
    
    def extract_file_info(self, filename: str, company_code: str) -> Dict[str, str]:
        """ファイル名から情報を抽出"""
        # ファイル名パターン: YYYY-MM-DD_証券コード_期間情報_attachments.md
        parts = filename.split('_')
        
        info = {
            'company_code': company_code,
            'date': 'unknown',
            'period': 'unknown'
        }
        
        if len(parts) >= 3:
            # 日付部分
            if re.match(r'^\d{4}-\d{2}-\d{2}$', parts[0]):
                info['date'] = parts[0]
            
            # 期間情報（第X四半期など）
            for part in parts[2:]:
                if '四半期' in part or '期' in part:
                    info['period'] = part.replace('_attachments.md', '')
                    break
        
        return info
    
    def read_markdown_content(self, file_path: str) -> str:
        """マークダウンファイルの内容を読み込み"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.log(f"ファイル読み込み完了: {file_path} ({len(content)} 文字)")
            return content
        except Exception as e:
            self.log(f"ファイル読み込みエラー: {file_path} - {e}")
            return ""


class OutputManager:
    """出力管理クラス"""
    
    def __init__(self, output_dir: str, format_type: str = "both", verbose: bool = False):
        self.output_dir = Path(output_dir)
        self.format_type = format_type
        self.verbose = verbose
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str):
        """詳細ログ出力"""
        if self.verbose:
            print(f"[OUTPUT] {message}")
    
    def save_results(self, results: List[Dict[str, Any]], model_name: str):
        """結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.format_type in ["csv", "both"]:
            self.save_csv(results, f"{model_name}_{timestamp}.csv")
        
        if self.format_type in ["json", "both"]:
            self.save_json(results, f"{model_name}_{timestamp}.json")
    
    def save_csv(self, results: List[Dict[str, Any]], filename: str):
        """CSV形式で保存"""
        csv_path = self.output_dir / filename
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'file_path', 'company_code', 'date', 'period', 
                    'content_length', 'sentiment_score', 'sentiment_interpretation', 'model_used'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = {
                        'file_path': result['file_path'],
                        'company_code': result['company_code'],
                        'date': result['date'],
                        'period': result['period'],
                        'content_length': result['content_length'],
                        'sentiment_score': result['sentiment_score'],
                        'sentiment_interpretation': result['sentiment_interpretation'],
                        'model_used': result['model_used']
                    }
                    writer.writerow(row)
            
            self.log(f"CSV保存完了: {csv_path}")
            
        except Exception as e:
            self.log(f"CSV保存エラー: {e}")
    
    def save_json(self, results: List[Dict[str, Any]], filename: str):
        """JSON形式で保存"""
        json_path = self.output_dir / filename
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.log(f"JSON保存完了: {json_path}")
            
        except Exception as e:
            self.log(f"JSON保存エラー: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='決算短信補足資料のセンチメント分析ツール',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='使用可能なコマンド')
    
    # sentimentサブコマンド
    sentiment_parser = subparsers.add_parser('sentiment', help='センチメント分析')
    sentiment_parser.add_argument(
        'model',
        choices=['daigo', 'tabularisai', 'moritzlaurer'],
        help='使用するモデル'
    )
    sentiment_parser.add_argument(
        '-i', '--input',
        default='attachments/',
        help='入力ディレクトリ（デフォルト: attachments/）'
    )
    sentiment_parser.add_argument(
        '-o', '--output',
        default='results/',
        help='出力ディレクトリ（デフォルト: results/）'
    )
    sentiment_parser.add_argument(
        '-f', '--format',
        choices=['csv', 'json', 'both'],
        default='both',
        help='出力形式（デフォルト: both）'
    )
    sentiment_parser.add_argument(
        '--cpu',
        action='store_true',
        help='CPU強制使用'
    )
    sentiment_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='詳細ログ'
    )
    
    args = parser.parse_args()
    
    if args.command != 'sentiment':
        parser.print_help()
        sys.exit(1)
    
    # 各コンポーネント初期化
    analyzer = SentimentAnalyzer(verbose=args.verbose)
    processor = FileProcessor(verbose=args.verbose)
    output_manager = OutputManager(args.output, args.format, verbose=args.verbose)
    
    print(f"Transformer決算短信分析ツール")
    print(f"モデル: {args.model}")
    print(f"入力: {args.input}")
    print(f"出力: {args.output}")
    
    # デバイス設定とモデルロード
    analyzer.setup_device(force_cpu=args.cpu)
    analyzer.load_model(args.model)
    
    # ファイル検出
    files = processor.find_markdown_files(args.input)
    if not files:
        print("処理対象のマークダウンファイルが見つかりませんでした。")
        sys.exit(1)
    
    print(f"処理対象ファイル数: {len(files)}")
    
    # 分析実行
    results = []
    for i, file_info in enumerate(files):
        print(f"[{i+1}/{len(files)}] {file_info['file_path']}")
        
        # ファイル読み込み
        content = processor.read_markdown_content(file_info['file_path'])
        if not content:
            continue
        
        # センチメント分析
        analysis_result = analyzer.analyze_text(content)
        
        # 結果をまとめる
        result = {
            'file_path': file_info['file_path'],
            'company_code': file_info['company_code'],
            'date': file_info['date'],
            'period': file_info['period'],
            'content_length': len(content),
            'sentiment_score': analysis_result['sentiment_score'],
            'sentiment_interpretation': analysis_result['sentiment_interpretation'],
            'model_used': analysis_result['model_used'],
            'raw_scores': analysis_result['raw_scores'],
            'error': analysis_result['error']
        }
        results.append(result)
        
        if args.verbose:
            print(f"  スコア: {result['sentiment_score']:.2f} ({result['sentiment_interpretation']})")
    
    # 結果保存
    output_manager.save_results(results, args.model)
    
    print(f"\n分析完了！")
    print(f"処理済みファイル数: {len(results)}")
    print(f"結果保存先: {args.output}")


if __name__ == "__main__":
    main()