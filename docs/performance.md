# ローカル生成性能

この文書は、YuE2 APIサーバーのローカル実測値と測定条件を記録します。
音質ベンチマークではなく、1タスクを単独実行した場合の生成速度とGPUメモリ使用量です。

## RTX PRO 4000 Blackwell 24GB

測定日: 2026-09-26

### 実行環境

| 項目 | 値 |
|---|---|
| GPU | NVIDIA RTX PRO 4000 Blackwell, 24,467MiB |
| GPU割り当て | 物理GPU 1のみ公開 (`CUDA_VISIBLE_DEVICES=1`)、プロセス内では`cuda:0` |
| NVIDIAドライバ | 580.173.02 |
| Python | 3.12.3 |
| PyTorch / CUDA | 2.10.0+cu128 / CUDA 12.8 |
| vLLM | 0.19.0 |
| APIサーバー | 2026-09-26測定時版(生成経路は初回公開版と同一) |
| YuE2 | `09a1e8a85bf35a93b8c01b3f12b139b558b49852` |
| モデル | `m-a-p/YuE2-3B` |
| VAE | `m-a-p/YuE2-Vae` |
| プロファイル | `fast` (`vllm`, FP8, エンジン常駐, ARオフロードなし) |
| 生成モード | `cot=full`, 48kHz FLAC |

### 測定方法

- 再現用入力は[`benchmarks/requests/pro4000-blackwell-full.json`](../benchmarks/requests/pro4000-blackwell-full.json)。
- サーバーとvLLMエンジンをロード済みにした後、同時リクエストなしで1タスクずつ実行。
- 内部時間は生成物`result.json`の`timing.e2e_seconds`。
- API外形時間は`POST /release_task`直前から、`/query_result`が成功を返すまでの経過時間。
- VRAMは生成中に`nvidia-smi`を1秒間隔で取得した`memory.used`の最大値。
- 3分換算は `内部時間 / 音声秒数 * 180`。曲長に対する単純換算値であり、固定費用があるため実際の3分出力と完全には一致しない。
- LLMによる歌詞生成、背景画像生成、HTTPでの音声ダウンロード時間は含まない。

### ウォーム測定結果

| seed | 音声長 | 内部時間 | API外形 | RTF | 倍リアルタイム | 3分換算 | VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2860647060 | 221.999秒 | 79.737秒 | 81.148秒 | 0.359 | 2.78倍 | 64.6秒 | 14,955MiB |
| 別seed | 252.439秒 | 87.697秒 | 88.902秒 | 0.347 | 2.88倍 | 62.5秒 | 15,055MiB（観測値） |

代表値は、**3分曲で約63〜65秒、VRAMピーク約14.6GiB**。

### 固定seed測定のフェーズ内訳

| フェーズ | 時間 | 補足 |
|---|---:|---|
| ABC | 10.474秒 | 2,030 tokens、193.8 tokens/s、TTFT 0.096秒 |
| Semantic | 32.904秒 | 5,551 tokens、168.7 tokens/s、TTFT 0.027秒 |
| NAR | 30.710秒 |  |
| VAE | 5.641秒 |  |
| E2E | 79.737秒 |  |

### コールドスタート

同じ固定seed・同じ221.999秒出力の初回生成は109.862秒でした。このときvLLMエンジンの
ロードに17.426秒を要しています。常駐APIとしての通常性能にはウォーム測定値を使用します。

### 実行例

```bash
curl -X POST http://127.0.0.1:8002/release_task \
  -H 'Content-Type: application/json' \
  --data-binary @benchmarks/requests/pro4000-blackwell-full.json
```

返却された`task_id`を`POST /query_result`でポーリングし、完了後の生成ディレクトリにある
`result.json`で内部時間と出力音声長を確認します。
