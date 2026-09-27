#!/usr/bin/env bash
set -euo pipefail

# YuE2 vLLMエンジンにFP8量子化オプション(YUE2_VLLM_QUANT=fp8)を追加するパッチ適用スクリプト。
#
#   1. まず patches/fast-fp8.diff の適用を試みる(将来バージョンにも当たる可能性が高い)
#   2. 当たらなければ、yue2-infer 0.1.6 と一致する場合のみ vendor/yue2-fp8/fast.py で置き換える
#
# 使い方: patches/apply_fp8_patch.sh [venvのpython]   (省略時 YuE/.venv/bin/python)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${1:-$ROOT_DIR/YuE/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
  echo "エラー: python が見つかりません: $PYTHON" >&2
  exit 1
fi

FAST=$("$PYTHON" -c "import yue2.fast as m; print(m.__file__)")
VERSION=$("$PYTHON" -c "import importlib.metadata as md; print(md.version('yue2-infer'))")
PKG_DIR=$(dirname "$FAST")
echo "対象: $FAST (yue2-infer $VERSION)"

if grep -q "YUE2_VLLM_QUANT" "$FAST"; then
  echo "適用済みです。何もしません。"
  exit 0
fi

cp "$FAST" "$FAST.orig-backup"
echo "バックアップ: $FAST.orig-backup"

# --- 方式1: diffパッチ ---
if patch --dry-run -p1 -d "$(dirname "$PKG_DIR")" < "$ROOT_DIR/patches/fast-fp8.diff" >/dev/null 2>&1; then
  patch -p1 -d "$(dirname "$PKG_DIR")" < "$ROOT_DIR/patches/fast-fp8.diff"
  echo "OK: diffパッチを適用しました。"
else
  # --- 方式2: 同梱ファイル(バージョン一致時のみ) ---
  if [ "$VERSION" = "0.1.6" ]; then
    cp "$ROOT_DIR/vendor/yue2-fp8/fast.py" "$FAST"
    echo "OK: 同梱の改変ファイル(0.1.6ベース)で置き換えました。"
  else
    rm -f "$FAST.orig-backup"
    echo "エラー: パッチが適用できず、バージョンも0.1.6ではありません($VERSION)。" >&2
    echo "patches/fast-fp8.diff の変更(4行)を手動で適用してください。" >&2
    exit 1
  fi
fi

"$PYTHON" -c "import importlib, yue2.fast; importlib.reload(yue2.fast); print('検証OK: importに成功')"
echo "有効化するには YUE2_VLLM_QUANT=fp8 を設定してサーバを起動してください。"
