#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# -----------------------------------------------------------------------------
# YuE2 REST API サーバ
#
# - in-memory キュー / ジョブストアのため workers=1 固定
# - YuE2 は1リクエストずつ処理 (ワーカー1本)
# - 既定ポート 8002
# -----------------------------------------------------------------------------

# Xet CDN が不安定な環境向け (従来HTTPダウンロードを使用)
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

export YUE2_MODEL="${YUE2_MODEL:-m-a-p/YuE2-3B}"
export YUE2_VAE="${YUE2_VAE:-m-a-p/YuE2-Vae}"
# 検証済みモデルリビジョン。更新時は明示的に上書きする。
if [ "$YUE2_MODEL" = "m-a-p/YuE2-3B" ]; then
  export YUE2_MODEL_REVISION="${YUE2_MODEL_REVISION:-14fc6c6f146441b1dd6363fcb2e01e82a6914cb7}"
else
  export YUE2_MODEL_REVISION="${YUE2_MODEL_REVISION:-}"
fi
if [ "$YUE2_VAE" = "m-a-p/YuE2-Vae" ]; then
  export YUE2_VAE_REVISION="${YUE2_VAE_REVISION:-9a94e1d0ea9f8087e98f77fa88df4a4068104d2a}"
else
  export YUE2_VAE_REVISION="${YUE2_VAE_REVISION:-}"
fi
export YUE2_DEVICE="${YUE2_DEVICE:-cuda:0}"

# 実行プロファイル (詳細は yue2_api_server.py 冒頭コメント):
#   original     — torch/BF16 アップストリーム既定 (3分曲 約61秒, VRAM 約9GB)
#   fast         — vLLM常駐+FP8 最速 (3分曲 約33秒, VRAM 約15GB) ※品質確認済み
#   fast-lowvram — vLLM+FP8をNAR前に解放 (曲ごとの再起動コストと引換えにVRAM削減)
export YUE2_PROFILE="${YUE2_PROFILE:-fast}"
export YUE2_OUTPUT_DIR="${YUE2_OUTPUT_DIR:-$ROOT_DIR/outputs/api}"
# 安全のため既定はローカルホストのみ。LAN/外部へ公開する場合は
# YUE2_API_HOST=0.0.0.0 と強力な YUE2_API_KEY を明示的に設定する。
export YUE2_API_HOST="${YUE2_API_HOST:-127.0.0.1}"
export YUE2_API_PORT="${YUE2_API_PORT:-8002}"
# export YUE2_API_KEY=replace-with-a-long-random-value  # Bearer認証を有効にする場合

exec "$ROOT_DIR/YuE/.venv/bin/python" yue2_api_server.py
