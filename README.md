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

結論からいうと、両者は**同じAR高速化経路**を使い、AR終了後にvLLMを
残すか捨てるかが主な違いです。`fast`は速度のためにARエンジンを残し、
`fast-lowvram`はVRAMのためにARエンジンを捨てます。NARとVAEの計算内容、
生成ステップ数、サンプリング設定を省略して速くしているわけではありません。

#### 実際に設定される値

| 設定 | `fast` | `fast-lowvram` | 意味 |
|---|---:|---:|---|
| `YUE2_BACKEND` | `vllm` | `vllm` | ABCとセマンティックトークンのAR生成をvLLMで実行 |
| `YUE2_VLLM_QUANT` | `fp8` | `fp8` | vLLMワーカー内のAR重みをFP8量子化 |
| `YUE2_KEEP_VLLM` | `1` | `0` | NAR開始時にvLLMワーカーを終了させるかどうか |
| `YUE2_OFFLOAD_AR` | `0` | `0` | NAR中のAR層CPU退避は行わない |
| `YUE2_MEMORY_BUDGET_GIB` | `32` | `12` | GPU物理容量で上限を切る前のメモリ予算 |
| VAEの既定タイル長 | 1024 frames | 512 frames | 12GiB以下の予算ではデコード時の一時メモリを削減 |

表はプロファイルの既定値です。同名の個別環境変数を明示すれば、その項目だけを
上書きできます。たとえば`YUE2_KEEP_VLLM=0`を指定した`fast`はエンジンを常駐させません。

`YUE2_MEMORY_BUDGET_GIB=32`は32GiBを必ず確保する指定ではありません。
物理VRAM容量を上限として扱う「使用可能予算」で、24GB GPUなら実容量側で制限されます。
反対に`fast-lowvram`の12GiB設定は、親プロセスのPyTorchアロケータ、vLLMの
GPU使用率、およびVAEのタイル長を低メモリ向けに制限します。数値はハード上限の
目安であり、CUDAコンテキストや別プロセスを含む`nvidia-smi`の表示値と完全には一致しません。

#### 1曲を生成する処理の流れ

YuE2の生成は、大きく次の4段階です。

1. **ABCプラン生成（AR）**: スタイルと歌詞から、メロディーやコードを表すABC記譜を生成します。
2. **セマンティックトークン生成（AR）**: 条件文とABCから、音楽の粗い時系列表現であるcodec token列を生成します。
3. **NAR音響生成**: セマンティックトークンから64次元の音響latentを生成します。32ステップのmidpoint法によるflow matchingです。
4. **VAEデコード**: latentを48kHzの波形へ変換し、FLACとして保存します。

`fast`系がvLLM＋FP8へ置き換えるのは、1と2の**AR部分だけ**です。3のNARは
YuE2本体をBF16で、4のVAEはFP32で実行します。したがって「モデル全体をFP8化」
しているわけではなく、FP8による速度差と数値差が直接入る範囲はAR生成です。

処理は内部で次のように進みます。

1. APIサーバ起動時に`YuE2Pipeline`を作成し、モデルの所在とハッシュを確認します。
   この時点ではvLLMエンジン自体はまだ起動しません。
2. 最初のAR生成時に、親のAPIプロセスとは別に
   `python -m yue2.fast --worker`を起動します。プロセスを分けることで、
   ワーカー終了時にvLLMのCUDAコンテキストとキャッシュをまとめて確実に解放できます。
3. 元のYuE2 MoTチェックポイントから、ARに必要な埋め込み、Qwen3層、正規化層、
   LM headだけを抽出し、vLLMが読めるQwen3形式の派生チェックポイントを作ります。
   派生物は入力モデルのidentityをキーに`$YUE2_CACHE/yue2-ar`、または
   `$HF_HOME/yue2-ar`以下へ保存されます。ロック、manifest、SHA-256検証があるため、
   2回目以降は同じ派生物を安全に再利用します。
4. ワーカー起動前に、親プロセスにロード済みのYuE2本体やVAEがあればCPUへ移し、
   `torch.cuda.empty_cache()`を実行して、vLLMを載せる空きを作ります。
5. vLLM 0.19.0の`AsyncLLM`を、BF16演算dtype、最大コンテキスト24,576、
   同時系列数1、chunked prefill、prefix cache有効で構築します。KV cacheは
   BF16相当の必要量を系列数1として明示計算します。
6. 本リポジトリのFP8パッチが`quantization="fp8"`をvLLMへ渡します。
   これによりAR重みのFP8量子化と対応する融合カーネルが選ばれます。
   パッチはこのオプションを環境変数から渡すだけで、プロンプト、乱数seed、
   temperature、top-p、top-k、生成可能token集合などは変更しません。
7. ABC生成とセマンティック生成は同じワーカーを続けて使用します。フェーズごとに
   許可するtoken範囲と終了tokenを制限し、YuE2固有の直近window反復ペナルティは
   カスタムTriton logits processorで再現します。API親プロセスとの通信は
   stdin/stdout上のJSON Linesで行い、tokenと計測値を受け取ります。
8. セマンティック生成が終わるとNARへ進み、ここでプロファイルごとの違いが現れます。

#### `fast`: vLLMを曲間でも常駐させる

アップストリームの`YuE2Pipeline.synthesize()`は、NAR開始直前に`close_vllm()`を
呼びます。`fast`ではAPIサーバ側がこの呼び出しだけを無操作化し、実際の終了関数は
サーバ停止時用に保存します。その結果、次の状態になります。

```text
1曲目: vLLM起動 → ABC → semantic → [vLLMを残したまま] NAR → VAE
2曲目:             ABC → semantic → [vLLMを残したまま] NAR → VAE
                              ↑ エンジンの再ロードなし
```

- vLLMワーカーはNAR/VAE中も生存し、AR重みとKV cache用領域を保持します。
- NAR時には親プロセスのBF16 YuE2本体もGPUへ載るため、別プロセスのvLLMと
  メモリ使用期間が重なります。これが約15GBのピークになる主因です。
- VAEデコード前には親プロセス内のYuE2本体をCPUへ戻しますが、vLLMワーカーは残ります。
- 2曲目以降はvLLMのプロセス生成、派生チェックポイント読込、FP8重み準備、
  エンジンとカーネルの初期化などの固定費を再度払わず、すぐAR生成へ入れます。
- APIサーバ終了時は保存しておいた本来の終了関数を呼び、ワーカープロセスを停止します。
  プロセスグループへSIGTERMを送り、10秒で止まらなければSIGKILLまで行って孤児化を防ぎます。

つまり`fast`の速さは、ARをvLLM＋FP8で速くする効果に加え、**常駐サーバで
1曲ごとのエンジン再起動を消す効果**によるものです。性能値を比較するときは、
初回のコールド生成と、エンジンロード済みのウォーム生成を分けて考える必要があります。

#### `fast-lowvram`: ARとNAR/VAEを同時に載せない

`fast-lowvram`はアップストリーム本来の`close_vllm()`をそのまま使います。
セマンティック生成を終えて`synthesize()`へ入った直後、NARモデルをGPUへ載せる前に
vLLMワーカープロセスを終了します。

```text
1曲目: vLLM起動 → ABC → semantic → vLLM終了 → NAR → VAE
2曲目: vLLM再起動 → ABC → semantic → vLLM終了 → NAR → VAE
                                      ↑ VRAM使用期間を分離
```

- AR中はvLLMワーカーを使用し、NAR/VAE用モデルはCPU側に置きます。
- AR終了後はワーカーをプロセスごと終了するため、vLLMの重み、KV cache、
  CUDAコンテキストが解放されてからBF16のYuE2本体をGPUへ載せます。
- VAE時にはYuE2本体を再びCPUへ移し、VAEだけをGPUへ載せます。
- さらに12GiB予算ではVAEを1024 framesではなく512 frames単位でタイルデコードし、
  長い曲のデコード一時メモリを減らします。halo 16 framesを付けて境界を処理するため、
  単純に音声を不連続なブロックへ分割しているわけではありません。
- 次の曲ではvLLMを再び起動する必要があり、実測環境では曲ごとに約20秒の固定費が
  加わります。派生ARチェックポイントはキャッシュ済みでも、重みのロードと
  GPU上のエンジン構築は毎回必要です。

このように、約9GBというピークはモデルを小さくした結果ではなく、主に
**AR、NAR、VAEのGPU滞在時間を重ねない**ことで実現しています。生成アルゴリズムを
省略していないため、代償は曲ごとの再起動時間です。

#### FP8の範囲、品質、再現性

- FP8になるのはvLLMワーカー内のAR重みです。KV cache、NAR、VAEまでFP8になるわけではありません。
- FP8パッチによる変更は[`patches/fast-fp8.diff`](patches/fast-fp8.diff)の実質4行です。
  `YUE2_VLLM_QUANT`を読み、値をvLLMの`AsyncEngineArgs.quantization`へ渡します。
- Blackwellで、FP8なしのvLLM ARと比較してAR部分が約36%高速化しました。
  これは曲全体が36%高速になるという意味ではありません。NARとVAEの時間は残ります。
- FP8はBF16と丸め方が異なるため、同じseedでもBF16経路とtoken選択が分岐し、
  最終的な曲が変わる可能性があります。聴感品質は同等であることを確認していますが、
  BF16の同一出力再現やアップストリームとの厳密比較には`original`を使用してください。
- `fast`と`fast-lowvram`同士はAR設定が同じです。ただし、実行環境、ライブラリ、
  GPU、並列実行状況まで含むbit単位の同一性を保証するものではありません。

#### vLLMを使わずtorchへフォールバックする場合

`YUE2_PROFILE=fast`系を選んでも、すべての入力が必ずvLLMを通るわけではありません。
次の場合は互換性を優先してAR生成をtorchへフォールバックします。

- `cfg_scale != 1.0`で、positive/negativeの2枝を使うCFGが必要な場合
- `cot=off`の歴史的サンプリング経路を使う場合（既定guidanceも1.01）
- CUDA以外のdeviceを指定した場合
- パイプライン本体の実験的`quantization`を別途有効にした場合

通常の`cot=full`または`cot=melody`、`cfg_scale`未指定、CUDA実行ではvLLM経路になります。
実際にどちらを通ったかは生成物の`result.json`にある
`timing.abc.backend_actual`と`timing.semantic.backend_actual`で確認できます。
同じ箇所の`engine_load_seconds`が0なら既存エンジンを再利用しており、0より大きければ
そのフェーズで新しくロードしたことを示します。`output_tps`、`ttft_seconds`、
`kv_cache_memory_bytes`もAR側の詳細確認に利用できます。

#### 選び方

- **24GB級以上で、APIサーバから連続して曲を作る**: `fast`。ウォーム状態の速度を優先します。
- **12GB級、または他のGPU処理へVRAMを空けたい**: `fast-lowvram`。1曲ごとの待ち時間と引換えにピークを抑えます。
- **アップストリームBF16経路との比較、FP8を含まない再現性が必要**: `original`。

なお、`/plan`はABC生成だけでNARへ進まないため、`fast-lowvram`でもその呼び出し直後には
「NAR直前の解放点」へ到達しません。通常の`/release_task`、`/edit_task`、
`/regenerate_task`ではsemantic生成後にNARへ進むため、上記のライフサイクルになります。

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
