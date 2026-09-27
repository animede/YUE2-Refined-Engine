# YuE2 使用ガイドまとめ

情報源: [GitHubリポジトリ](https://github.com/multimodal-art-projection/YuE) / [デモページ](https://map-yue2.github.io/) / skillガイド(`skills/yue2-music/`) / [GIGAZINE記事](https://gigazine.net/gsc_news/en/20260911-yue2-music-generation-ai/) ほか(2026-09-24時点)

## 1. YuE2とは

Multimodal Art Projection (m-a-p) が2026年9月9日に公開したオープンな楽曲生成モデル。歌詞とスタイルプロンプトから、ボーカル+伴奏つきの完全な楽曲(48kHzステレオ)を生成する。

最大の特徴は **「ホワイトボックス作曲」**: 音声を生成する前に、まずABC記譜法によるシンボリックな楽譜プラン(メロディ+コード)を書き出し、人間やエージェントがそれを読み・演奏し・編集してから音声化できる。

3つの利用モード:
- **CREATE**: 歌詞+スタイル → 楽曲生成
- **COVER**: 既存音源をSheetSage2で採譜 → メロディを保ったままゼロショットでスタイル変換
- **EDIT**: エージェントとの対話による楽譜の部分編集 → 再生成

日本語ボーカルにも対応(公式デモに日本語曲あり)。

## 2. アーキテクチャ

- **YuE2-3B** (約3.6Bパラメータ): AR–NAR Mixture-of-Transformers バックボーン
  - 自己回帰(AR)で楽譜トークンとセマンティックトークンを生成
  - フローマッチング(NAR)で音響潜在表現を生成
- **YuE2-Vae**: 音響潜在表現を量子化なしで48kHzステレオ音声にデコード
- Python APIは段階実行可能: `plan()` → `generate_semantic()` → `synthesize()` → `decode()`

## 3. モデル一覧 (Hugging Face)

| モデル | HF ID | 役割 |
|---|---|---|
| YuE2-3B | `m-a-p/YuE2-3B` | 本体(生成・プラン・カバー・編集) |
| YuE2-Vae | `m-a-p/YuE2-Vae` | 標準デコーダ(通常の生成・試聴用) |
| YuE2-Vae-legacy | `m-a-p/YuE2-Vae-legacy` | ベンチマーク再現用デコーダ |
| SheetSage2 | `m-a-p/SheetSage2` | 音源→楽譜(メロディ/コード/拍/キー/ABC/MIDI)採譜 |
| MERT-v2-FullSong | `m-a-p/MERT-v2-FullSong` | SheetSage2のエンコーダ親(全曲特徴量) |
| MERT-v2-30s | `m-a-p/MERT-v2-30s` | 短尺クリップ用特徴量(任意) |

初回実行時にHugging Faceから自動ダウンロード。

## 4. 動作要件・インストール

- Linux / Python 3.12 / BF16対応NVIDIA GPU / **VRAM 24GB**(1リクエストずつ)
- 速度目安: RTX 4090で約215秒の曲を約71秒で生成

```bash
git clone https://github.com/multimodal-art-projection/YuE.git
cd YuE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

注意: **SheetSage2は依存が衝突するため別仮想環境が必要**(YuE2: PyTorch 2.10/Transformers 4.57/NumPy 2.2、SheetSage2: PyTorch 2.8/Transformers 4.45/NumPy 1.24、Python 3.10–3.11、FFmpeg 6.1必須)。

## 5. 基本の使い方

### CLI
```bash
yue2 generate --request examples/song.json --output outputs/song-cli
# または
python examples/generate.py --output outputs/first-song
```

### Python API
```python
import json
from pathlib import Path
from yue2 import YuE2Pipeline

request = json.loads(Path("examples/song.json").read_text(encoding="utf-8"))
with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda") as pipe:
    song = pipe(**request)
    song.save_artifacts("outputs/my-song")
```

### リクエストJSONの主要フィールド
- `style`: ジャンル・楽器・ボーカル特性・言語・テンポの記述
- `lyrics`: `[Verse]` `[Chorus]` などのセクションタグ付き歌詞
- `cot` (プランニングモード):
  - `"full"`(既定): メロディ+コードの編集可能なプランを生成
  - `"melody"`: メロディのみプラン、伴奏は自由(カバー向き)
  - `"off"`: プランなしで直接生成
- `abc`: 自作/編集済みのABC楽譜を直接指定(`full`/`melody`モードで有効)
- `cfg_scale`: テキストガイダンス強度(既定値は調整済み。むやみに変えない)

### プラン(楽譜)と音声合成の分離実行
```python
plan = pipe.plan(**request)
plan.save("outputs/plan")
restored = SymbolicPlan.load("outputs/plan")
semantic = pipe.generate_semantic(restored)
audio = pipe.decode(pipe.synthesize(semantic))
```
編集は保存済みプランを直接書き換えず、ABCのコピーを作って修正する。

### 出力アーティファクト (`save_artifacts()`)
音声(FLAC)、楽譜(ABC)、プランJSON、セマンティックトークン、音響潜在配列、設定、タイミング、整合性記録。潜在をキャッシュしておけば、曲を再生成せずにデコーダだけ差し替えて再デコード可能。

### 再現性
`from_pretrained(..., revision=, vae_revision=, cache_dir=, local_files_only=True)` でモデルをピン留め。シードを固定して比較する。

## 6. カバー(ゼロショットスタイル変換)ワークフロー

1. SheetSage2で元音源を採譜(MERT-v2-FullSongを自動ロード)
2. メロディとタイミングを確認・修正、`melody_only=True` でメロディのみABCをエクスポート(コード記号は除去: `python scripts/abc_tools.py strip-chords input.abc output.abc`)
3. YuE2-3Bに `cot="melody"` + 新しいスタイル + 歌詞を渡して生成
4. YuE2-Vaeでデコード

性能: 948曲のカバー評価で、楽譜条件付きは CLEWS mAP **0.647**、楽譜なしは 0.006(カバー専用ファインチューニングなしで達成)。

注意: MERTの連続埋め込みをYuE2に直接渡してはいけない(YuE2は離散セマンティックトークンを使う)。

## 7. エージェント編集(EDIT)ワークフロー

skillガイド(`references/editing-workflows.md`)の要点:

1. **ベースライン確保**: 元曲を完全生成し、リクエスト・プラン・ABC・セマンティックトークン・潜在・デコーダ名をすべて保存
2. **不変条件(コントラクト)を定義**: 「音高と音価を完全維持」「音高列は維持しリズムは変更可」「テーマが認識できればよい」「自由改変」のどのレベルか明示。テンポ・拍子・歌詞・間奏の変更可否も指定(`--allow-tempo-change` フラグあり)
3. **編集エージェントに委任**: 未変更のABC+スタイル/歌詞+比較用音源、変更指示、保存契約を渡す。成果物として「編集済み楽譜+変更箇所マニフェスト(小節/拍範囲、コード境界、音符・歌詞変更)+音楽的根拠」を要求
4. **独立検証**: 別のレビュアーが前後のABCと制約を突き合わせて違反を検出
5. **再生成して試聴**: 新しい出力ディレクトリでレンダリングし、全曲+編集箇所周辺を確認

リハーモナイズ時は小節単位でなく、保続音・強拍・カデンツ・フレーズ境界をマークして音の持続範囲全体で検討する。歌詞の翻訳・差し替えは音節と音符の対応を文書化する。

## 8. skillガイド(`skills/yue2-music/`)の構成

Claude等のエージェント向けポータブルskillパッケージとして同梱:

```
skills/yue2-music/
├── SKILL.md                 # ワークフロー全体の指示
├── agents/                  # 編集用エージェント定義
├── assets/                  # プロンプト例など
├── scripts/                 # run_yue2.py, abc_tools.py, listen.py
└── references/
    ├── models-and-setup.md      # モデル・環境構築
    ├── generation-and-covers.md # 生成・カバー
    ├── abc-editing.md           # ABC記譜編集
    ├── editing-workflows.md     # エージェント編集
    └── listening-and-evaluation.md # 試聴・評価
```

主要スクリプト:
```bash
python scripts/run_yue2.py generate --request assets/prompt.json --output outputs/pop
python scripts/run_yue2.py plan --request assets/prompt.json --output outputs/plan
python scripts/abc_tools.py strip-chords input.abc output.abc
python scripts/listen.py outputs/pop outputs/jazz --output outputs/comparison
```

skillの品質基準: 楽譜上のチェック・ASR解析・実際の試聴を区別して報告し、楽譜チェックだけで「音が完全に保存された」と主張しない。プラン・潜在・ABCの原本は編集前に必ず保存する。

## 9. ベンチマーク (WildSongBench, 192プロンプト, 2026年9月)

| システム | SongBench Avg | AudioBox PQ | MuLan | PER |
|---|---|---|---|---|
| **YuE2 (best-of-8)** | **6.9632** | 8.2714 | 0.5051 | 9.79% |
| Mureka 9 | 6.9377 | 8.0226 | 0.4394 | 11.69% |
| Suno v5 | 6.8721 | 8.1698 | 0.5428 | 8.10% |
| **YuE2 (単発)** | 6.7316 | 8.2598 | 0.5068 | 8.44% |
| Suno v6 | 6.5562 | — | — | — |

17システム中、best-of-8で最高平均。ベンチマーク再現には YuE2-Vae-legacy を使用。

## 10. ライセンス

- コード/ドキュメント/エージェントskill: Apache 2.0
- モデル重み: **CC BY-NC 4.0 + クリエイター許諾条項**
  - 個人・商用クリエイター: 生成物の利用・収益化は無料で可
  - 学術研究: 非商用で無料
  - 企業: 商用ライセンスは別途相談

## 11. 論文・ブログ・関連リンク

- **技術レポート**: [YuE2 Technical Report](https://github.com/multimodal-art-projection/YuE/blob/main/docs/technical_report.pdf)
- 公式デモ: https://map-yue2.github.io/ (旧: https://map-yue.github.io/)
- オンライン試用: https://yue.noizai.net/
- Discord: https://discord.gg/ssAyWMnMzu
- リリース: [yue2-v0.1.6](https://github.com/multimodal-art-projection/YuE/releases/tag/yue2-v0.1.6) (2026-09-09)
- ComfyUI版チュートリアル: https://docs.comfy.org/ja/tutorials/audio/yue2/yue2
- 日本語解説記事: [GIGAZINE](https://gigazine.net/gsc_news/en/20260911-yue2-music-generation-ai/) / [Innovatopia](https://innovatopia.jp/ai/ai-news/117767/) / [note(導入ガイド)](https://note.com/sepiablue/n/n0d79de507613)
