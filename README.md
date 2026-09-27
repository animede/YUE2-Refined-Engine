# YuE2 API Server

> [!IMPORTANT]
> 本リポジトリはYuE2の非公式・第三者実装です。
> YuE2の開発チームによる公式APIサーバではありません。

[YuE2](https://github.com/multimodal-art-projection/YuE)(オープンな楽曲生成モデル)を、
REST API(`/release_task`・`/query_result`)経由で利用する常駐サーバです。
[Momo Song v5](https://github.com/animede/momo-song-v5) のバックエンドとして開発しました。

## YuE2要約

- 歌詞とスタイルプロンプトから、ABC記譜ベースのプランを経由して楽曲を生成するモデルです
- 生成は `CREATE` / `COVER` / `EDIT` の3系統で、楽譜の編集や再生成を挟めます
- 本READMEはサーバ利用向けの要点をまとめたもので、詳細メモは [YuE2まとめ.md](YuE2まとめ.md) に置いています

## 特徴

- **タスクAPI**: `/release_task` で生成を開始 → `/query_result` でポーリング → `/v1/audio` から生成物をダウンロード
- **楽譜(ABC)ワークフロー**: `/plan`(楽譜のみ生成)・`/edit_task`(編集済み楽譜で再生成)・`/regenerate_task`(同じ楽譜で別テイク)・`/redecode_task`(潜在から音声のみ再デコード、約0.4秒)
- **ABC事前検証**: YuE2ネイティブ方言パーサ(skill同梱の`abc_tools.py`)で、音価・タイ・コード語彙・小節グリッドをGPU実行前に検証し、小節位置つきエラーを返します
- **実行プロファイル** (`YUE2_PROFILE`):
  | プロファイル | 簡単な説明 | 構成 | 3分曲の生成時間* | VRAMピーク* |
  |---|---|---|---:|---:|
  | `original` | アップストリーム準拠。再現性確認や標準構成との比較向け | torch/BF16 | 約61秒 | 約9GB |
  | `fast` (既定) | 速度優先。十分なVRAMがあり、連続生成する通常運用向け | vLLMエンジン常駐 + FP8融合カーネル | **約33秒** | 約15GB |
  | `fast-lowvram` | VRAM節約優先。12GB級GPU向けだが、曲ごとのエンジン再起動分だけ遅い | FP8 + エンジン非常駐 | 約55秒 | 約9GB |

  *RTX PRO 5000 Blackwell (sm_120) 実測。`fast`系はFP8パッチ(下記)が必要です

### `fast` / `fast-lowvram` で行っていること

どちらも、楽譜プランとセマンティックトークンを生成するAR部分をtorch/BF16からvLLMへ切り替え、
このリポジトリの4行のパッチでvLLMの重みFP8量子化と融合カーネルを有効にしています。
BlackwellではAR部分が約36%高速化しました。NAR音響生成とVAEデコードにはFP8を適用していません。

- `fast`: vLLMワーカープロセスをNAR/VAE処理中も終了させず、次の曲までGPUに常駐させます。曲ごとのエンジン再起動を省くため最速ですが、vLLMとNAR/VAEのメモリが重なる分、約15GBのVRAMを使います。
- `fast-lowvram`: 同じvLLM＋FP8を使いながら、AR生成後、NAR処理の前にvLLMを解放します。次の曲ではエンジンを再起動するため遅くなりますが、メモリの重なりを避けてピークを約9GBに抑えます。

### RTX PRO 4000 Blackwell実測

`fast`プロファイルのウォーム状態では、221.999秒の曲を79.737秒で生成しました。
これは約2.78倍リアルタイム、3分曲換算で約64.6秒です。別seedを含む2回の測定から、
**3分曲で約63〜65秒、VRAMピーク約14.6GiB**を代表値とします。

| GPU | 音声長 | YuE2内部 | API外形 | 3分換算 | 最大VRAM |
|---|---:|---:|---:|---:|---:|
| RTX PRO 4000 Blackwell 24GB | 221.999秒 | 79.737秒 | 81.148秒 | 64.6秒 | 14,955MiB |

測定条件、フェーズ別時間、コールドスタート値、再現用リクエストは
[docs/performance.md](docs/performance.md)を参照してください。

## セットアップ

```bash
git clone --branch yue2-v0.1.6 --depth 1 https://github.com/multimodal-art-projection/YuE.git
python3.12 -m venv YuE/.venv
YuE/.venv/bin/pip install --upgrade pip
YuE/.venv/bin/pip install ./YuE
YuE/.venv/bin/pip install -r requirements-server.txt

# fast/fast-lowvramプロファイルを使う場合 (vLLM + FP8パッチ)
YuE/.venv/bin/pip install -r requirements-fast.txt
./patches/apply_fp8_patch.sh          # 詳細は patches/README.md
```

`vllm==0.19.0` は現在のFP8パッチで検証済みの組み合わせです。このサーバは
vLLMのOpenAI互換HTTPサーバを起動せず、内部エンジンとしてのみ使用します。
依存関係には定期的にセキュリティ更新が入るため、更新版はFP8パッチと
生成品質の両方を検証してから採用してください。

モデル(m-a-p/YuE2-3B 等)は初回リクエスト時にHugging Faceから自動取得されます。
要件: Linux / Python 3.12 / BF16対応NVIDIA GPU (VRAM 12GB以上、`original`+低予算設定なら12GB級で動作確認済み)。

## 起動

```bash
./run_yue2_api_server.sh              # ポート8002
YUE2_PROFILE=original ./run_yue2_api_server.sh   # プロファイル指定
```

既定では `127.0.0.1:8002` だけで待ち受けます。このサーバはローカルの
1ユーザー利用を想定しており、レート制限や生成物の自動削除は行いません。

主な環境変数: `YUE2_PROFILE` / `YUE2_DEVICE` / `YUE2_MEMORY_BUDGET_GIB` /
`YUE2_MODEL_REVISION` / `YUE2_VAE_REVISION` /
`YUE2_API_PORT` / `YUE2_API_KEY`(認証) / `YUE2_OUTPUT_DIR`。
詳細は [yue2_api_server.py](yue2_api_server.py) 冒頭のコメントを参照してください。
モデルは検証済みのHugging Face commit SHAに固定しており、別リビジョンはこれらの
環境変数で明示的に指定できます。

### LAN/外部ネットワークで使う場合

全インターフェースで待ち受ける場合は、強力なAPIキーを必ず併用してください。

```bash
YUE2_API_HOST=0.0.0.0 \
YUE2_API_KEY='replace-with-a-long-random-value' \
./run_yue2_api_server.sh
```

認証有効時はタスクAPIと `/v1/audio` の両方でBearerヘッダーが必要です。

```bash
curl -H 'Authorization: Bearer replace-with-a-long-random-value' \
  'http://127.0.0.1:8002/v1/audio?path=TASK_ID/audio.flac'
```

インターネットへの直接公開は想定していません。必要な場合はTLS、追加認証、
リクエストサイズ制限、レート制限を備えたリバースプロキシの内側で運用してください。

## 使用例

```bash
# 生成タスク投入
curl -X POST localhost:8002/release_task -H "Content-Type: application/json" -d '{
  "prompt": "Japanese, upbeat city pop, female vocal, electric piano, 104 BPM",
  "lyrics": "[Verse]\n街の灯りが揺れて\n[Chorus]\n走り出せ 今すぐに",
  "cot": "full"
}'
# → {"data": {"task_id": "..."}} を /query_result でポーリング
```

## FP8パッチについて

vLLMエンジンに重みFP8量子化オプションを追加する4行のパッチです(Blackwell実測でAR +36%)。
diffパッチ([patches/fast-fp8.diff](patches/fast-fp8.diff))と改変済みファイル同梱
([vendor/yue2-fp8/fast.py](vendor/yue2-fp8/fast.py)、Apache-2.0改変表示付き)の両方式に対応し、
[patches/apply_fp8_patch.sh](patches/apply_fp8_patch.sh) が自動選択します。
詳細は [patches/README.md](patches/README.md)。

## ライセンス

- 本リポジトリのコード: [Apache License 2.0](LICENSE)
- `vendor/yue2-fp8/fast.py` は YuE2 (Apache-2.0) の改変版(改変表示はファイル冒頭)
- YuE2のモデル重みは本リポジトリに含まれません(CC BY-NC 4.0+クリエイター許諾条項。
  [MODEL_LICENSE](https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE) を参照)

## テスト

```bash
YuE/.venv/bin/pip install -r requirements-dev.txt
YUE2_INIT_AT_STARTUP=0 YuE/.venv/bin/python -m unittest discover -s tests -v
```

このテストはモデルやGPUを読み込まず、認証、ファイル配信の境界、APIの基本応答を確認します。

## 関連ドキュメント

- [YuE2まとめ.md](YuE2まとめ.md) — 詳細な調査メモ(日本語)
