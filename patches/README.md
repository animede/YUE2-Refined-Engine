# YuE2 FP8パッチ (vLLMエンジンの重みFP8量子化)

YuE2のvLLMバックエンドに、環境変数 `YUE2_VLLM_QUANT=fp8` でエンジン側の
重みFP8量子化(融合カーネル)を有効化するオプションを追加します。
NVIDIA Blackwell (sm_120) 実測でAR生成が約36%高速化します。既定動作は不変です。

- 対象: `yue2-infer` の `yue2/fast.py`(変更は実質4行、[fast-fp8.diff](fast-fp8.diff) 参照)
- 原著: [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)(コードは Apache-2.0)
- 本パッチもApache-2.0。改変ファイル([../vendor/yue2-fp8/fast.py](../vendor/yue2-fp8/fast.py))には
  Apache-2.0 §4(b) に基づく改変表示ヘッダを付しています

## 適用方法

```bash
./patches/apply_fp8_patch.sh                # YuE/.venv を対象
./patches/apply_fp8_patch.sh /path/to/venv/bin/python   # 任意のvenv
```

スクリプトは次の順で適用します:

1. **diffパッチ方式**: `fast-fp8.diff` を `patch -p1` で適用。
   yue2-inferがバージョンアップしても該当箇所が変わっていなければそのまま当たります
2. **同梱ファイル方式**(フォールバック): diffが当たらない場合、
   インストール済みバージョンが 0.1.6 のときのみ `vendor/yue2-fp8/fast.py` で置き換えます
3. どちらも不可(0.1.6以外でdiff不適合)の場合はエラー終了します。
   その際は diff の4行を手動で適用してください

適用前のファイルは `fast.py.orig-backup` として自動バックアップされます。
再実行は安全です(適用済みを検出してスキップ)。

## 元に戻す

```bash
FAST=$(YuE/.venv/bin/python -c "import yue2.fast as m; print(m.__file__)")
cp "$FAST.orig-backup" "$FAST"
```

## 注意

- FP8有効時はBF16と数値経路が異なるため、同一シードでも生成される曲は変わります
  (聴感品質は同等であることを確認済み。ベンチマーク再現時はFP8を無効にしてください)
- yue2-inferを更新(pip install -U等)すると上書きされるため、更新後に再適用してください
