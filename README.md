# 音声・動画 ファイルの文字起こし

OpenAI Whisperを使用して音声・動画ファイルを高精度で文字起こしするPythonツールです。日本語に最適化されており、複数の出力形式に対応しています。

## 機能

- 🎯 **高精度文字起こし**: OpenAI Whisperモデルを使用した高精度な音声認識
- 🇯🇵 **日本語最適化**: 日本語音声に特化した設定
- 📁 **複数出力形式**: タイムスタンプ付きテキスト、SRT字幕、全文テキスト
- ⚡ **高速処理**: GPU対応で高速な文字起こし
- 🎛️ **柔軟な設定**: モデルサイズ、ビームサイズ、品質設定のカスタマイズ
- 🔄 **自動セグメント分割**: 長時間音声ファイルを自動的に30分単位で分割して処理（精度向上）
- 📂 **バッチ処理**: inputフォルダ内のファイルを連続処理し、処理済みファイルを自動整理

## 出力ファイル形式

各音声ファイルに対して、以下の形式で出力されます：

1. **`<filename>.time.txt`** - タイムスタンプ付きテキスト
2. **`<filename>.srt`** - SRT形式の字幕ファイル
3. **`<filename>.txt`** - 全文テキスト

出力ファイルは、入力ファイル名と同じ名前のフォルダに保存されます。

## 自動セグメント分割機能

長時間の音声ファイル（30分以上）は、自動的に以下の方法でセグメントに分割されて処理されます：

- **無音区間検出**: 音声の空白域を自動検出して分割点を決定
- **精度向上**: 各セグメントを個別に処理することで、長時間音声での精度低下を防止
- **自動統合**: 分割された結果を自動的に統合して、元のファイルと同じ形式で出力

この機能により、1時間以上の音声ファイルでも最後まで高精度な文字起こしが可能になります。

## バッチ処理機能

`input`フォルダ内のファイルを連続処理し、処理済みファイルを自動的に`output`フォルダに整理する機能です：

- **連続処理**: inputフォルダ内の全ての音声・動画ファイルを順次処理
- **自動整理**: 処理が完了したファイルは`output/<ファイル名>/`フォルダに移動
- **進捗表示**: 処理状況をリアルタイムで表示
- **エラーハンドリング**: 個別ファイルの処理失敗が全体に影響しない

### 出力構造

```
output/
├── ファイル名1/
│   ├── ファイル名1.mp3          # 元の音声ファイル
│   ├── ファイル名1.time.txt     # タイムスタンプ付きテキスト
│   ├── ファイル名1.srt          # SRT字幕ファイル
│   └── ファイル名1.txt          # 全文テキスト
└── ファイル名2/
    ├── ファイル名2.wav
    ├── ファイル名2.time.txt
    ├── ファイル名2.srt
    └── ファイル名2.txt
```

## インストール

### 前提条件

- Python 3.10以上
- [uv](https://github.com/astral-sh/uv) - 高速なPythonパッケージマネージャー
- CUDA対応GPU（推奨、高速処理のため）

### インストール手順

1. リポジトリをクローン

```bash
git clone <repository-url>
cd audiowhisper
```

2. 依存関係をインストール

```bash
uv sync
```

3. 仮想環境をアクティベート

```bash
uv shell
```

## 使用方法

### 基本的な使用方法

```bash
python main.py -i "音声ファイル.mp3"
```

### 複数ファイルの処理

```bash
python main.py -i "ファイル1.mp3" "ファイル2.wav" "ファイル3.m4a"
```

### 高度なオプション

```bash
python main.py \
  -i "音声ファイル.mp3" \
  --model large \
  --beam_size 8 \
  --high_quality \
  --output_mask 7
```

## コマンドラインオプション

| オプション | 説明 | デフォルト値 |
|-----------|------|-------------|
| `-i, --input` | 入力音声ファイルのパス（複数指定可） | - |
| `--batch` | inputフォルダ内のファイルを連続処理 | False |
| `--input_dir` | inputフォルダのパス | input |
| `--output_dir` | 出力先フォルダ | output |
| `--beam_size` | 文字起こし時のビームサイズ | 5 |
| `-m, --model` | Whisperモデルのサイズ | turbo |
| `--output_mask` | 出力ファイルのマスク | 7 |
| `--high_quality` | 高精度モードを有効化 | False |
| `--no_auto_segment` | 長時間音声ファイルの自動セグメント分割を無効化 | False |
| `--segment_duration` | セグメント分割時の目標長（秒） | 1800（30分） |
| `--setup` | 必要なフォルダ（input, output）を作成 | False |
| `--shutdown` | 処理終了後にシャットダウン | False |

### モデルサイズ

| モデル | パラメータ数 | 必要VRAM | 相対速度 | 用途 |
|--------|-------------|----------|----------|------|
| tiny | 39M | ~1GB | ~10x | 軽量処理 |
| base | 74M | ~1GB | ~7x | 標準処理 |
| small | 244M | ~2GB | ~4x | バランス型 |
| medium | 769M | ~5GB | ~2x | 高精度 |
| large | 1550M | ~10GB | 1x | 最高精度 |
| turbo | 809M | ~6GB | ~8x | **推奨** |

### 出力マスク

出力マスクは以下の値の組み合わせで指定します：

- `1`: タイムスタンプ付きテキスト（`.time.txt`）
- `2`: SRT字幕ファイル（`.srt`）
- `4`: 全文テキスト（`.txt`）

例：

- `7` (1+2+4): 全形式出力（デフォルト）
- `3` (1+2): タイムスタンプ付きテキストとSRTのみ
- `6` (2+4): SRTと全文テキストのみ

## 使用例

### 基本的な文字起こし

```bash
python main.py -i "会議録音.mp3"
```

### 高精度モードでの処理

```bash
python main.py -i "重要な会議.mp3" --high_quality --model large
```

### 長時間音声ファイルの処理（自動セグメント分割）

```bash
python main.py -i "1時間の講演.mp3"  # 自動的に30分単位で分割
```

### カスタムセグメント長での処理

```bash
python main.py -i "長時間音声.mp3" --segment_duration 900  # 15分単位で分割
```

### 自動セグメント分割を無効化

```bash
python main.py -i "音声ファイル.mp3" --no_auto_segment  # 通常の処理
```

### バッチ処理（inputフォルダ）

```bash
# セットアップ（必要なフォルダを作成）
python main.py --setup

# inputフォルダ内のファイルを連続処理
python main.py --batch

# 高精度モードでバッチ処理
python main.py --batch --high_quality --model large

# カスタムフォルダでバッチ処理
python main.py --batch --input_dir "my_audio" --output_dir "results"
```

### 特定の出力形式のみ

```bash
python main.py -i "動画.mp4" --output_mask 2  # SRTファイルのみ
```

### 複数ファイルの処理

```bash
python main.py -i *.mp3 --model small --beam_size 3
```

## 高精度モード

`--high_quality`オプションを使用すると、以下の設定が自動的に適用されます：

- モデルが`turbo`の場合、`large`に自動変更
- ビームサイズが6以上に設定
- `condition_on_previous_text: True`
- `temperature: 0`
- `compression_ratio_threshold: 2.4`
- `no_speech_threshold: 0.6`

これにより、より高精度な文字起こしが可能になりますが、処理時間は長くなります。

## 対応ファイル形式

- **音声ファイル**: MP3, WAV, M4A, FLAC, OGG
- **動画ファイル**: MP4, AVI, MOV, MKV（音声部分を抽出）

## パフォーマンス

- **CPU処理**: すべての環境で動作
- **GPU処理**: CUDA対応GPUがある場合、大幅な速度向上
- **メモリ使用量**: モデルサイズに応じて1GB〜10GB

## トラブルシューティング

### よくある問題

1. **CUDAエラー**: GPUメモリ不足の場合、より小さいモデルを使用
2. **ファイル形式エラー**: 対応していない形式の場合は、事前に変換
3. **メモリ不足**: モデルサイズを小さくするか、ビームサイズを減らす
4. **長時間音声の精度低下**: 自動セグメント分割機能が有効になっているか確認
5. **セグメント分割が動作しない**: `--no_auto_segment`オプションが指定されていないか確認
6. **inputフォルダが見つからない**: `--setup`オプションでフォルダを作成
7. **処理済みファイルの移動失敗**: ファイルが他のプロセスで使用中でないか確認

### 推奨設定

- **一般的な用途**: `--model turbo`（デフォルト）
- **高精度が必要**: `--model large --high_quality`
- **高速処理**: `--model small --beam_size 3`
- **メモリ制限環境**: `--model tiny --beam_size 1`
- **長時間音声（1時間以上）**: デフォルトで自動セグメント分割が有効
- **短時間音声**: `--no_auto_segment`で自動分割を無効化可能
- **バッチ処理**: `--batch`オプションで連続処理
- **大量ファイル処理**: `--model small --beam_size 3 --batch`で効率的に処理

### 自動セグメント分割の詳細

- **動作条件**: 音声ファイルが30分（1800秒）以上の場合
- **分割方法**: 無音区間を検出して自然な区切りで分割
- **精度向上**: 各セグメントを個別に処理することで、長時間音声での精度低下を防止
- **カスタマイズ**: `--segment_duration`で分割長を調整可能
- **無効化**: `--no_auto_segment`で自動分割を無効化

### バッチ処理の詳細

- **対応ファイル形式**: MP3, WAV, M4A, FLAC, OGG, AAC, MP4, AVI, MOV, MKV, WMV, FLV
- **処理順序**: ファイル名のアルファベット順
- **エラーハンドリング**: 個別ファイルの失敗が全体に影響しない
- **進捗表示**: 現在処理中のファイルと全体の進捗を表示
- **自動整理**: 処理完了後、元ファイルと出力ファイルを`output/<ファイル名>/`フォルダに整理

## 開発環境

このプロジェクトは[uv](https://github.com/astral-sh/uv)を使用して依存関係を管理しています。

### 開発用コマンド

```bash
# 依存関係のインストール
uv sync

# 仮想環境のアクティベート
uv shell

# 開発用依存関係の追加
uv add --dev pytest

# 依存関係の更新
uv lock --upgrade
```

## 貢献

バグ報告や機能要望は、GitHubのIssuesページでお知らせください。

## 謝辞

- [OpenAI Whisper](https://github.com/openai/whisper) - 音声認識エンジン
- [PyTorch](https://pytorch.org/) - 機械学習フレームワーク
- [uv](https://github.com/astral-sh/uv) - 高速なPythonパッケージマネージャー
