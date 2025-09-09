#!/usr/bin/env python3
"""
決算短信（attachments/<5桁コード>/*.md）を読み込み、長文に対応した分割＋集約で
センチメント分析を実行し、CSVを出力します。

要件:
- 実行ファイル名: transformer.py
- 出力CSVの列: filename, date, code, title, sentiment_score
- codelist.csv を読み、記載コードのフォルダのみ処理
- デフォルトGPU実行（CUDAがあればdevice=0）。-cpu 指定時のみCPU
- bardsai/finance-sentiment-ja-base（金融特化モデル）を使用

使用例:
uv run python transformer.py -v
uv run python transformer.py --all -v
python transformer.py -i attachments -o results -cpu
"""

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

try:
    import torch
    from transformers import pipeline
except ImportError as e:
    print(f"ライブラリ読み込みエラー: {e}")
    print("pip install transformers torch を実行してください")
    sys.exit(1)


# bardsai金融特化モデル用のラベル定義（3段階）


def detect_device(force_cpu: bool = False) -> int:
    if force_cpu:
        return -1
    return 0 if torch.cuda.is_available() else -1


def build_pipeline(device: int):
    # bardsai金融特化モデルでパイプラインを構築
    clf = pipeline(
        "text-classification",
        model="bardsai/finance-sentiment-ja-base",
        device=device,
    )
    return clf, "bardsai/finance-sentiment-ja-base"


def split_text(text: str, max_chars: int = 400, overlap: int = 50) -> List[str]:
    # bardsai用の分割設定（512トークン対応）
    # 句点・改行を優先して安全に分割（長文対応）
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    n = len(text)
    seps = "。！？\n\r"
    while start < n:
        end = min(start + max_chars, n)
        cut = end
        if end < n:
            # 近傍の区切り位置を探す
            window = text[start:end]
            back = max((window.rfind(s) for s in seps), default=-1)
            # 区切りが見つかり、かつ前進できる場合のみcutを更新
            if back != -1 and (start + back + 1) > start:
                cut = start + back + 1
        # 念のため最低1文字は前進する（無限ループ防止）
        if cut <= start:
            cut = min(start + 1, n)
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        if cut >= n:
            break
        # 通常はオーバーラップ分だけ戻すが、同一位置に戻らないよう最低1文字は前進
        start = max(cut - overlap, start + 1)
    return chunks


def score_chunk_bardsai(pred) -> float:
    # bardsai/finance-sentiment-ja-base の出力処理
    # 出力は [{'label': 'POSITIVE'|'NEGATIVE'|'NEUTRAL', 'score': p}]
    if not isinstance(pred, list) or not pred:
        return 5.0  # デフォルト中立
    
    item = pred[0]  # 最高スコアの結果を使用
    label = item["label"].lower()
    score = float(item["score"])
    
    # ラベルに基づいてスコアを0-10に変換
    if "positive" in label:
        return 5.0 + (score * 5.0)  # 5.0-10.0
    elif "negative" in label:
        return 5.0 - (score * 5.0)  # 0.0-5.0
    else:  # neutral
        return 5.0


def aggregate_scores(parts: List[Tuple[float, int]]) -> float:
    # 長さ重み付き平均
    total_len = sum(l for _, l in parts) or 1
    return sum(s * l for s, l in parts) / total_len


def parse_filename(fname: str) -> Tuple[str, str, str, str]:
    # 期待: YYYY-MM-DD[_-]XXXXX[_-]<title>... .md
    base = os.path.basename(fname)
    stem = re.sub(r"\.[^.]+$", "", base)
    # 末尾のattachmentsなどを削除
    stem = re.sub(r"[_-]attachments$", "", stem, flags=re.IGNORECASE)
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[_-]([A-Za-z0-9]{5})[_-](.+)$", stem)
    if m:
        date, code, title = m.group(1), m.group(2), m.group(3)
        return base, date, code, title
    # フォールバック：アンダースコア区切りを優先
    parts = stem.split("_")
    date = parts[0] if parts and re.match(r"^\d{4}-\d{2}-\d{2}$", parts[0]) else "unknown"
    code = parts[1] if len(parts) > 1 and re.match(r"^[A-Za-z0-9]{5}$", parts[1]) else "unknown"
    title = "_".join(parts[2:]) if len(parts) > 2 else stem
    return base, date, code, title


def read_codelist(path: Path) -> List[str]:
    codes: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("code") or "").strip()
            if code and re.match(r"^[A-Za-z0-9]{5}$", code):
                codes.append(code)
    return codes


def get_all_codes(attachments_dir: Path) -> List[str]:
    """attachmentsディレクトリ内の全フォルダから5桁コードを取得"""
    codes: List[str] = []
    if not attachments_dir.exists():
        return codes
    
    for item in attachments_dir.iterdir():
        if item.is_dir() and re.match(r"^[A-Za-z0-9]{5}$", item.name):
            codes.append(item.name)
    
    return sorted(codes)


@dataclass
class Args:
    input: Path
    output: Path
    codes_file: Path
    cpu: bool
    verbose: bool
    all: bool


def run(args: Args) -> Path:
    args.output.mkdir(parents=True, exist_ok=True)
    device = detect_device(force_cpu=args.cpu)
    clf, model_name = build_pipeline(device)

    # 出力CSVの準備
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.output / f"sentiment_{ts}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "date", "code", "title", "sentiment_score"])  # 指定の列

        if args.all:
            codes = get_all_codes(args.input)
            if args.verbose:
                print(f"対象コード数: {len(codes)} (attachments内全フォルダ)")
        else:
            codes = read_codelist(args.codes_file)
            if args.verbose:
                print(f"対象コード数: {len(codes)} (codelist.csv)")
        
        if not codes:
            print("処理対象のコードが見つかりません")
            return out_csv

        for i, code in enumerate(codes, 1):
            company_dir = args.input / code
            if not company_dir.exists():
                if args.verbose:
                    print(f"[{i}/{len(codes)}] スキップ（フォルダ無）: {company_dir}")
                continue

            md_files = sorted(company_dir.glob("*.md"))
            if args.verbose:
                print(f"[{i}/{len(codes)}] {code}: {len(md_files)}件")

            for j, md in enumerate(md_files):
                if args.verbose:
                    print(f"  ファイル[{j+1}/{len(md_files)}]: {md.name}")
                try:
                    text = md.read_text(encoding="utf-8")
                except Exception as e:
                    if args.verbose:
                        print(f"  読込失敗: {md} - {e}")
                    continue

                filename, date, code_in_name, title = parse_filename(md.name)
                chunks = split_text(text)
                scored_parts: List[Tuple[float, int]] = []

                if args.verbose:
                    print(f"  チャンク数: {len(chunks)}")

                # 推論時は明示的に勾配追跡を無効化
                try:
                    INFERENCE_CTX = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
                except Exception:
                    # フォールバック（通常は到達しない）
                    INFERENCE_CTX = torch.no_grad

                for k, ch in enumerate(chunks):
                    if args.verbose and k % 10 == 0:
                        print(f"    チャンク[{k+1}/{len(chunks)}]処理中...")
                    try:
                        with INFERENCE_CTX():
                            pred = clf(ch, truncation=True, max_length=512)
                        score = score_chunk_bardsai(pred)
                        scored_parts.append((score, len(ch)))
                    except Exception as e:
                        print(f"    エラー発生: ファイル={md.name}, チャンク={k+1}/{len(chunks)}")
                        print(f"    エラー詳細: {e}")
                        print(f"    チャンク文字数: {len(ch)}")
                        print(f"    チャンク内容（先頭200文字）: {ch[:200]}")
                        print(f"    エラー種類: {type(e).__name__}")
                        import traceback
                        print(f"    スタックトレース:")
                        traceback.print_exc()
                        # エラーの詳細情報を出力してから停止
                        raise e

                final_score = aggregate_scores(scored_parts)
                writer.writerow([filename, date, code_in_name, title, f"{final_score:.4f}"])
                if args.verbose:
                    print(f"  完了: {filename} -> {final_score:.4f}")

    return out_csv


def main():
    p = argparse.ArgumentParser(description="決算短信センチメント分析（CSV出力）")
    p.add_argument("-i", "--input", type=Path, default=Path("attachments"), help="入力ルート（attachments）")
    p.add_argument("-o", "--output", type=Path, default=Path("results"), help="出力ディレクトリ（results）")
    p.add_argument("--codes-file", type=Path, default=Path("codelist.csv"), help="処理対象コード一覧CSV")
    p.add_argument("--all", action="store_true", help="attachments内の全フォルダを処理（codelist.csvを無視）")
    p.add_argument("-cpu", "--cpu", action="store_true", help="CPU強制（指定時のみCPU）")
    p.add_argument("-v", "--verbose", action="store_true", help="詳細ログ")
    args_ns = p.parse_args()

    args = Args(input=args_ns.input, output=args_ns.output, codes_file=args_ns.codes_file, 
                cpu=args_ns.cpu, verbose=args_ns.verbose, all=args_ns.all)

    if not args.input.exists():
        print(f"入力ディレクトリが見つかりません: {args.input}")
        sys.exit(1)
    if not args.all and not args.codes_file.exists():
        print(f"コードリストが見つかりません: {args.codes_file}")
        sys.exit(1)

    device = "GPU(CUDA:0)" if (not args.cpu and torch.cuda.is_available()) else "CPU"
    print(f"デバイス: {device}")
    out_csv = run(args)
    print(f"完了: {out_csv}")


if __name__ == "__main__":
    main()
