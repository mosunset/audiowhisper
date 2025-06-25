# 音声・動画 ファイルの文字起こし

OpenAI Whisperを使用して音声・動画ファイルを高精度で文字起こしするPythonツールです。日本語に最適化されており、複数の出力形式に対応しています。

## 機能

- 🎯 **高精度文字起こし**: OpenAI Whisperモデルを使用した高精度な音声認識
- 🇯🇵 **日本語最適化**: 日本語音声に特化した設定
- 📁 **複数出力形式**: タイムスタンプ付きテキスト、SRT字幕、全文テキスト
- ⚡ **高速処理**: GPU対応で高速な文字起こし
- 🎛️ **柔軟な設定**: モデルサイズ、ビームサイズ、品質設定のカスタマイズ

## 出力ファイル形式

各音声ファイルに対して、以下の形式で出力されます：

1. **`<filename>.time.txt`** - タイムスタンプ付きテキスト
2. **`<filename>.srt`** - SRT形式の字幕ファイル
3. **`<filename>.txt`** - 全文テキスト

出力ファイルは、入力ファイル名と同じ名前のフォルダに保存されます。

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
| `-i, --input` | 入力音声ファイルのパス（複数指定可） | 必須 |
| `--beam_size` | 文字起こし時のビームサイズ | 5 |
| `-m, --model` | Whisperモデルのサイズ | turbo |
| `--output_mask` | 出力ファイルのマスク | 7 |
| `--high_quality` | 高精度モードを有効化 | False |
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

### 特定の出力形式のみ

```bash
python main.py -i "動画.mp4" --output_mask 2  # SRTファイルのみ
```

### バッチ処理

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

### 推奨設定

- **一般的な用途**: `--model turbo`（デフォルト）
- **高精度が必要**: `--model large --high_quality`
- **高速処理**: `--model small --beam_size 3`
- **メモリ制限環境**: `--model tiny --beam_size 1`

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

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

バグ報告や機能要望は、GitHubのIssuesページでお知らせください。

## 謝辞

- [OpenAI Whisper](https://github.com/openai/whisper) - 音声認識エンジン
- [PyTorch](https://pytorch.org/) - 機械学習フレームワーク
- [uv](https://github.com/astral-sh/uv) - 高速なPythonパッケージマネージャー
