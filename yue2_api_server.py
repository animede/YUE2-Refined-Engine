"""FastAPI server for YuE2.

Endpoints:
- POST /release_task   生成タスクを投入
- POST /query_result   タスク結果のバッチ照会
- POST /plan           楽譜(ABC)プランのみ生成して返す (YuE2固有)
- POST /edit_task      既存タスクの楽譜(ABC)を編集して新規再生成 (YuE2固有)
- POST /regenerate_task 保存済みプランを再利用し半合成 (YuE2固有)
- POST /redecode_task  保存済みlatentから音声のみ再デコード (YuE2固有)
- GET  /v1/models      利用可能モデル一覧
- GET  /v1/audio       生成物ダウンロード (audio.flac / score.abc など)
- GET  /health         ヘルスチェック

in-memory キュー + ジョブストアのため workers=1 固定。
YuE2 は1リクエストずつしか処理できないので、ワーカーも1本。

起動:
    .venv/bin/python -m uvicorn yue2_api_server:app --host 127.0.0.1 --port 8002 --workers 1

環境変数:
    YUE2_MODEL              (default: m-a-p/YuE2-3B)
    YUE2_VAE                (default: m-a-p/YuE2-Vae)
    YUE2_MODEL_REVISION     (default: 検証済みcommit SHA)
    YUE2_VAE_REVISION       (default: 検証済みcommit SHA)
    YUE2_DEVICE             (default: auto)
    YUE2_MEMORY_BUDGET_GIB  (default: 24)
    YUE2_OUTPUT_DIR         (default: ./outputs/api)
    YUE2_API_KEY            (default: なし = 認証無効。非ループバック待受時は必須)
    YUE2_QUEUE_SIZE         (default: 20)
    YUE2_INIT_AT_STARTUP    (default: 1; 0で初回リクエスト時にロード)
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
import time
import traceback
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------

DEFAULT_MODEL_ID = "m-a-p/YuE2-3B"
DEFAULT_VAE_ID = "m-a-p/YuE2-Vae"
MODEL_ID = os.environ.get("YUE2_MODEL", DEFAULT_MODEL_ID)
VAE_ID = os.environ.get("YUE2_VAE", DEFAULT_VAE_ID)
MODEL_REVISION = os.environ.get(
    "YUE2_MODEL_REVISION",
    "14fc6c6f146441b1dd6363fcb2e01e82a6914cb7" if MODEL_ID == DEFAULT_MODEL_ID else "",
)
VAE_REVISION = os.environ.get(
    "YUE2_VAE_REVISION",
    "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a" if VAE_ID == DEFAULT_VAE_ID else "",
)
DEVICE = os.environ.get("YUE2_DEVICE", "auto")
OUTPUT_ROOT = Path(os.environ.get("YUE2_OUTPUT_DIR", "outputs/api")).resolve()
API_KEY = os.environ.get("YUE2_API_KEY", "")
QUEUE_SIZE = int(os.environ.get("YUE2_QUEUE_SIZE", "20"))
INIT_AT_STARTUP = os.environ.get("YUE2_INIT_AT_STARTUP", "1") == "1"

# 実行プロファイル (YUE2_PROFILE):
#   original     — torch/BF16。アップストリーム既定。シード再現性・ベンチマーク互換
#   fast         — vLLM常駐+FP8融合カーネル。最速 (3分曲 約33秒, VRAM計 約15GB)
#   fast-lowvram — vLLM+FP8をNAR前に解放 + 低メモリ予算。再起動コストと引換えにVRAM削減
# 個別の環境変数 (YUE2_BACKEND / YUE2_KEEP_VLLM / YUE2_VLLM_QUANT /
# YUE2_OFFLOAD_AR / YUE2_MEMORY_BUDGET_GIB) はプロファイル既定を上書きする。
_PROFILES = {
    "original":     {"backend": "torch", "keep_vllm": "0", "vllm_quant": "",
                     "offload_ar": "0", "budget": "24"},
    "fast":         {"backend": "vllm", "keep_vllm": "1", "vllm_quant": "fp8",
                     "offload_ar": "0", "budget": "32"},
    # 非常駐: エンジン6GBとメイン9GBが同時にGPUへ載らない(曲毎に約20秒の再起動コスト)
    "fast-lowvram": {"backend": "vllm", "keep_vllm": "0", "vllm_quant": "fp8",
                     "offload_ar": "0", "budget": "12"},
}
PROFILE = os.environ.get("YUE2_PROFILE", "fast")
if PROFILE not in _PROFILES:
    raise ValueError(f"YUE2_PROFILE must be one of {sorted(_PROFILES)}, got {PROFILE!r}")
_prof = _PROFILES[PROFILE]
BACKEND = os.environ.get("YUE2_BACKEND", _prof["backend"])
KEEP_VLLM = os.environ.get("YUE2_KEEP_VLLM", _prof["keep_vllm"]) == "1"
OFFLOAD_AR = os.environ.get("YUE2_OFFLOAD_AR", _prof["offload_ar"]) == "1"
MEMORY_BUDGET_GIB = float(os.environ.get("YUE2_MEMORY_BUDGET_GIB", _prof["budget"]))
# fast.py(vLLMワーカーサブプロセス)は環境変数で量子化を判定するため、ここで確定させる
if "YUE2_VLLM_QUANT" not in os.environ and _prof["vllm_quant"]:
    os.environ["YUE2_VLLM_QUANT"] = _prof["vllm_quant"]
VLLM_QUANT = os.environ.get("YUE2_VLLM_QUANT", "")

STATUS_MAP = {"queued": 0, "running": 0, "succeeded": 1, "failed": 2}


def wrap_response(data: Any, code: int = 200, error: Optional[str] = None) -> Dict[str, Any]:
    """共通レスポンス封筒."""
    return {"data": data, "code": code, "error": error}


# --------------------------------------------------------------------------
# リクエストモデル
# --------------------------------------------------------------------------

def _load_abc_tools():
    """skill同梱のABC方言パーサ(標準ライブラリのみ)を遅延ロードする。"""
    import importlib.util
    path = os.environ.get(
        "YUE2_ABC_TOOLS",
        str(Path(__file__).resolve().parent / "YuE" / "skills" / "yue2-music"
            / "scripts" / "abc_tools.py"))
    if not Path(path).is_file():
        return None
    spec = importlib.util.spec_from_file_location("yue2_abc_tools", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["yue2_abc_tools"] = module  # dataclass装飾が要求する
    spec.loader.exec_module(module)
    return module


_ABC_TOOLS = None


def _validate_abc_text(abc: str) -> None:
    """ABC検証。skillのネイティブ方言パーサがあれば本格検証、なければ軽量チェック。"""
    if not abc or not abc.strip():
        raise HTTPException(status_code=422, detail="abc: must not be empty")
    global _ABC_TOOLS
    if _ABC_TOOLS is None:
        _ABC_TOOLS = _load_abc_tools() or False
    if _ABC_TOOLS:
        # 音価倍数・タイ・コード語彙・小節グリッドまで検証(YuE2ネイティブ方言)
        try:
            _ABC_TOOLS.parse_abc(abc)
        except _ABC_TOOLS.AbcError as exc:
            raise HTTPException(status_code=422, detail=f"abc: {exc}") from None
        return
    lines = [line for line in abc.splitlines() if line.strip()]
    if not any(line.lstrip().startswith("X:") for line in lines):
        raise HTTPException(status_code=422, detail="abc: missing 'X:' reference number header")
    if not any(line.lstrip().startswith("V:") for line in lines):
        raise HTTPException(status_code=422, detail="abc: missing 'V:' voice line")


def _build_song_request(fields: Dict[str, Any]):
    """SongRequest構築をラップし ValueError/TypeError を422に変換する (release_task/plan/edit_task共通)."""
    from yue2.protocol import SongRequest
    if fields.get("abc"):
        _validate_abc_text(fields["abc"])
    try:
        return SongRequest(**fields)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid request: {exc}") from None


class GenerateMusicRequest(BaseModel):
    prompt: str = Field(default="", description="スタイル記述 (style と同義)")
    lyrics: str = Field(default="", description="歌詞 ([Verse]/[Chorus] タグ付き)")
    thinking: bool = Field(default=True, description="True=シンボリックプラン使用 (cot未指定時 full/off にマップ)")
    model: Optional[str] = None
    seed: Union[int, str] = -1
    use_random_seed: bool = True
    guidance_scale: Optional[float] = Field(default=None, description="YuE2の cfg_scale にマップ (未指定推奨)")
    task_type: str = "text2music"

    # YuE2固有
    style: Optional[str] = Field(default=None, description="スタイル記述 (prompt より優先)")
    cot: Optional[str] = Field(default=None, description="full / melody / off")
    abc: Optional[str] = Field(default=None, description="ABC楽譜を直接指定 (cot=full/melody 必須)")

    def to_song_request(self) -> Dict[str, Any]:
        style = self.style if self.style is not None else self.prompt
        if not style or not self.lyrics:
            raise HTTPException(status_code=422, detail="style (or prompt) and lyrics are required")
        cot = self.cot or ("full" if self.thinking else "off")
        if cot not in {"full", "melody", "off"}:
            raise HTTPException(status_code=422, detail="cot must be full, melody or off")
        req: Dict[str, Any] = {"style": style, "lyrics": self.lyrics, "cot": cot}
        seed = int(self.seed) if str(self.seed).lstrip("-").isdigit() else -1
        if not self.use_random_seed and seed >= 0:
            req["seed"] = seed
        else:
            req["seed"] = int.from_bytes(os.urandom(4), "big")
        if self.abc:
            _validate_abc_text(self.abc)
            req["abc"] = self.abc
        if self.guidance_scale is not None:
            req["cfg_scale"] = self.guidance_scale
        return req


class EditTaskRequest(BaseModel):
    """/edit_task: 既存タスクを土台に abc を差し替えて再生成する."""
    task_id: str
    abc: str
    style: Optional[str] = None
    lyrics: Optional[str] = None
    seed: Optional[int] = None
    cfg_scale: Optional[float] = None


class RegenerateTaskRequest(BaseModel):
    """/regenerate_task: 保存済みプランを流用して半合成する."""
    task_id: str
    seed: Optional[int] = None


class RedecodeTaskRequest(BaseModel):
    """/redecode_task: 保存済みlatentから音声だけ再デコードする."""
    task_id: str


# --------------------------------------------------------------------------
# パイプライン (単一インスタンス・遅延ロード)
# --------------------------------------------------------------------------

class _PipelineHolder:
    def __init__(self) -> None:
        self._pipe = None
        self._lock = threading.Lock()
        self.load_error: Optional[str] = None
        self.loading = False

    def get(self):
        with self._lock:
            if self._pipe is None:
                from yue2 import YuE2Pipeline
                if BACKEND == "vllm" and KEEP_VLLM:
                    # YuE2は24GB GPU向けにNAR前へエンジンを解放するが、VRAM潤沢環境では
                    # 常駐の方が速い(曲毎の再起動14秒を排除)。停止時はshutdown_pipeline()で解放。
                    import yue2.fast as yue2_fast
                    if not hasattr(yue2_fast, "_real_close_vllm"):
                        yue2_fast._real_close_vllm = yue2_fast.close_vllm
                        yue2_fast.close_vllm = lambda pipe: None
                self.loading = True
                try:
                    self._pipe = YuE2Pipeline.from_pretrained(
                        MODEL_ID, vae=VAE_ID, device=DEVICE,
                        revision=MODEL_REVISION or None,
                        vae_revision=VAE_REVISION or None,
                        memory_budget_gib=MEMORY_BUDGET_GIB, progress=False,
                        backend=BACKEND, offload_ar=OFFLOAD_AR,
                    )
                    self.load_error = None
                except Exception as exc:
                    self.load_error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    self.loading = False
            return self._pipe

    @property
    def ready(self) -> bool:
        return self._pipe is not None

    def shutdown(self) -> None:
        """常駐vLLMエンジンを含めて解放する(サーバ停止時用)."""
        with self._lock:
            if self._pipe is None:
                return
            import yue2.fast as yue2_fast
            real_close = getattr(yue2_fast, "_real_close_vllm", None)
            if real_close is not None:
                real_close(self._pipe)
            self._pipe.close()
            self._pipe = None


PIPELINE = _PipelineHolder()


# --------------------------------------------------------------------------
# ジョブストア + ワーカー
# --------------------------------------------------------------------------

class Job:
    def __init__(self, kind: str, request: Dict[str, Any], parent_task_id: Optional[str] = None) -> None:
        self.job_id = uuid.uuid4().hex
        self.kind = kind  # "generate" | "plan" | "regenerate" | "redecode"
        self.request = request
        self.parent_task_id = parent_task_id
        self.status = "queued"
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None


JOBS: Dict[str, Job] = {}


def _audio_url(path: Path) -> str:
    # API応答にサーバの絶対パスを露出しない。
    relative = path.resolve().relative_to(OUTPUT_ROOT)
    return f"/v1/audio?path={urllib.parse.quote(relative.as_posix())}"


def _relative_output_path(path: Path) -> str:
    return path.resolve().relative_to(OUTPUT_ROOT).as_posix()


def _run_generate(job: Job) -> Dict[str, Any]:
    """ブロッキング生成本体 (ワーカースレッドで実行)."""
    pipe = PIPELINE.get()
    out_dir = OUTPUT_ROOT / job.job_id
    song = pipe(**job.request)
    info = song.save_artifacts(out_dir)
    audio_path = out_dir / "audio.flac"
    abc_path = out_dir / "score.abc"
    # /query_result のキー構成
    return {
        "first_audio_path": _audio_url(audio_path),
        "second_audio_path": None,
        "audio_paths": [_audio_url(audio_path)],
        "raw_audio_paths": [_relative_output_path(audio_path)],
        "seed_value": str(job.request.get("seed", "")),
        "prompt": job.request["style"],
        "lyrics": job.request["lyrics"],
        "status_message": "complete",
        "metas": {
            "prompt": job.request["style"],
            "lyrics": job.request["lyrics"],
            "duration": info.get("audio_seconds"),
        },
        "generation_info": {
            "sample_rate": info.get("sample_rate"),
            "audio_seconds": info.get("audio_seconds"),
            "truncated": info.get("truncated"),
            "timing": info.get("timing"),
            "weights": info.get("weights"),
        },
        # YuE2固有: 楽譜と再生成用アーティファクト
        "abc_score": song.abc,
        "abc_path": _audio_url(abc_path) if abc_path.is_file() else None,
        "artifacts_dir": _relative_output_path(out_dir),
    }


def _run_plan(job: Job) -> Dict[str, Any]:
    """プラン(ABC楽譜)のみ生成 (音声合成なし)."""
    pipe = PIPELINE.get()
    out_dir = OUTPUT_ROOT / job.job_id
    plan = pipe.plan(**job.request)
    plan.save(out_dir)
    # /edit_task がプランの元リクエストを引き継げるよう、
    # 通常生成の save_artifacts() と同じ形式で保存する。
    from yue2.storage import write_json
    write_json(out_dir / "request.json", plan.request.to_dict())
    return {
        "abc_score": plan.abc,
        "truncated": plan.truncated,
        "plan_dir": _relative_output_path(out_dir),
        "prompt": job.request["style"],
        "lyrics": job.request["lyrics"],
        "status_message": "plan complete",
    }


def _run_regenerate(job: Job) -> Dict[str, Any]:
    """保存済み SymbolicPlan からABC生成を省略し, generate_semantic以降のみ実行."""
    from yue2.pipeline import SymbolicPlan
    from yue2.protocol import SongRequest
    pipe = PIPELINE.get()
    parent_dir = OUTPUT_ROOT / job.parent_task_id
    plan = SymbolicPlan.load(parent_dir)
    seed = job.request.get("seed")
    if seed is not None:
        plan = SymbolicPlan(SongRequest(**{**plan.request.to_dict(), "seed": seed}),
                            plan.abc, plan.abc_ids, plan.prefix, plan.timing, plan.truncated)
    out_dir = OUTPUT_ROOT / job.job_id
    semantic = pipe.generate_semantic(plan)
    latents = pipe.synthesize(semantic)
    audio = pipe.decode(latents)
    config = pipe.effective_config(plan.request)
    from yue2.pipeline import SongResult
    song = SongResult(audio, 48000, semantic, latents, config, pipe.weights,
                      {"semantic": semantic.timing}, "")
    info = song.save_artifacts(out_dir)
    audio_path, abc_path = out_dir / "audio.flac", out_dir / "score.abc"
    return {
        "first_audio_path": _audio_url(audio_path),
        "second_audio_path": None,
        "audio_paths": [_audio_url(audio_path)],
        "raw_audio_paths": [_relative_output_path(audio_path)],
        "seed_value": str(plan.request.seed),
        "prompt": plan.request.style,
        "lyrics": plan.request.lyrics,
        "status_message": "complete",
        "metas": {"prompt": plan.request.style, "lyrics": plan.request.lyrics,
                  "duration": info.get("audio_seconds")},
        "generation_info": {"sample_rate": info.get("sample_rate"),
                             "audio_seconds": info.get("audio_seconds"),
                             "truncated": info.get("truncated"), "timing": info.get("timing")},
        "abc_score": song.abc,
        "abc_path": _audio_url(abc_path) if abc_path.is_file() else None,
        "artifacts_dir": _relative_output_path(out_dir),
        "parent_task_id": job.parent_task_id,
    }


def _run_redecode(job: Job) -> Dict[str, Any]:
    """保存済み latent.npy から音声のみ再デコード (ARを再実行しない軽量ジョブ)."""
    pipe = PIPELINE.get()
    parent_dir = OUTPUT_ROOT / job.parent_task_id
    latents = np.load(parent_dir / "latent.npy")
    audio = pipe.decode(latents)
    out_dir = OUTPUT_ROOT / job.job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    audio_path = out_dir / "audio.flac"
    sf.write(audio_path, audio, 48000, subtype="PCM_24")
    request_json = parent_dir / "request.json"
    meta = json.loads(request_json.read_text()) if request_json.is_file() else {}
    return {
        "first_audio_path": _audio_url(audio_path),
        "second_audio_path": None,
        "audio_paths": [_audio_url(audio_path)],
        "raw_audio_paths": [_relative_output_path(audio_path)],
        "seed_value": str(meta.get("seed", "")),
        "prompt": meta.get("style", ""),
        "lyrics": meta.get("lyrics", ""),
        "status_message": "complete",
        "generation_info": {"sample_rate": 48000, "audio_seconds": len(audio) / 48000},
        "artifacts_dir": _relative_output_path(out_dir),
        "parent_task_id": job.parent_task_id,
    }


async def _worker(app: FastAPI) -> None:
    queue: asyncio.Queue = app.state.job_queue
    while True:
        job_id = await queue.get()
        job = JOBS.get(job_id)
        if job is None:
            queue.task_done()
            continue
        job.status = "running"
        job.started_at = time.time()
        try:
            runner = {"plan": _run_plan, "regenerate": _run_regenerate,
                     "redecode": _run_redecode}.get(job.kind, _run_generate)
            job.result = await asyncio.to_thread(runner, job)
            job.status = "succeeded"
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            job.finished_at = time.time()
            queue.task_done()


# --------------------------------------------------------------------------
# 認証
# --------------------------------------------------------------------------

def _verify_token(body: Dict[str, Any], authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    token = token or body.get("api_key") or body.get("token")
    if not isinstance(token, str) or not secrets.compare_digest(token, API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


# --------------------------------------------------------------------------
# FastAPI アプリ
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    app.state.job_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    worker = asyncio.create_task(_worker(app))
    if INIT_AT_STARTUP:
        # モデルロードは重い(初回はHFからDL)のでバックグラウンドスレッドで先行実行
        threading.Thread(target=lambda: PIPELINE.get(), daemon=True).start()
    yield
    worker.cancel()
    PIPELINE.shutdown()


app = FastAPI(title="YuE2 API Server", lifespan=lifespan)


async def _read_body(request: Request) -> Dict[str, Any]:
    content_type = (request.headers.get("content-type") or "").lower()
    if "json" in content_type:
        return await request.json()
    form = await request.form()
    return {key: value for key, value in form.items()}


async def _enqueue(app: FastAPI, job: Job) -> Dict[str, Any]:
    queue: asyncio.Queue = app.state.job_queue
    if queue.full():
        raise HTTPException(status_code=429, detail="Server busy: queue is full")
    JOBS[job.job_id] = job
    await queue.put(job.job_id)
    position = queue.qsize()
    return {"task_id": job.job_id, "status": "queued", "queue_position": position}


@app.post("/release_task")
async def release_task(request: Request, authorization: Optional[str] = Header(None)):
    body = await _read_body(request)
    _verify_token(body, authorization)
    req = GenerateMusicRequest(**{k: v for k, v in body.items()
                                  if k in GenerateMusicRequest.model_fields})
    song_req = req.to_song_request()
    _build_song_request(song_req)  # キュー投入前にSongRequest全体を検証
    job = Job("generate", song_req)
    return wrap_response(await _enqueue(app, job))


@app.post("/plan")
async def plan_task(request: Request, authorization: Optional[str] = Header(None)):
    """YuE2固有: ABC楽譜プランのみ生成するタスクを投入する."""
    body = await _read_body(request)
    _verify_token(body, authorization)
    req = GenerateMusicRequest(**{k: v for k, v in body.items()
                                  if k in GenerateMusicRequest.model_fields})
    song_req = req.to_song_request()
    if song_req.get("cot") == "off":
        raise HTTPException(status_code=422, detail="plan requires cot=full or melody")
    _build_song_request(song_req)
    job = Job("plan", song_req)
    return wrap_response(await _enqueue(app, job))


def _require_completed_parent(task_id: str) -> Job:
    job = JOBS.get(task_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"task_id not found: {task_id}")
    if job.status != "succeeded":
        raise HTTPException(status_code=409, detail=f"task_id not completed (status={job.status})")
    return job


@app.post("/edit_task")
async def edit_task(request: Request, authorization: Optional[str] = Header(None)):
    """既存タスクの request.json を土台に abc を差し替えて新規generateジョブを投入する."""
    body = await _read_body(request)
    _verify_token(body, authorization)
    edit = EditTaskRequest(**{k: v for k, v in body.items() if k in EditTaskRequest.model_fields})
    _require_completed_parent(edit.task_id)
    parent_dir = OUTPUT_ROOT / edit.task_id
    request_json = parent_dir / "request.json"
    if not request_json.is_file():
        raise HTTPException(status_code=404, detail=f"request.json missing for task_id: {edit.task_id}")
    base = json.loads(request_json.read_text())
    if base.get("cot") == "off":
        raise HTTPException(status_code=422, detail="parent task has cot=off; ABC editing requires full/melody")
    base.update({"abc": edit.abc})
    for field in ("style", "lyrics", "seed", "cfg_scale"):
        value = getattr(edit, field)
        if value is not None:
            base[field] = value
    base.pop("id", None)  # ジョブごとに job_id で出力先を分けるため id は既定値のまま
    _build_song_request(base)  # キュー投入前に検証 (422はここで送出)
    job = Job("generate", base, parent_task_id=edit.task_id)
    result = await _enqueue(app, job)
    result["parent_task_id"] = edit.task_id
    return wrap_response(result)


@app.post("/regenerate_task")
async def regenerate_task(request: Request, authorization: Optional[str] = Header(None)):
    """保存済みプランを再利用し, ABC生成をスキップして generate_semantic 以降のみ実行する."""
    body = await _read_body(request)
    _verify_token(body, authorization)
    regen = RegenerateTaskRequest(**{k: v for k, v in body.items() if k in RegenerateTaskRequest.model_fields})
    _require_completed_parent(regen.task_id)
    parent_dir = OUTPUT_ROOT / regen.task_id
    if not (parent_dir / "plan_manifest.json").is_file():
        raise HTTPException(status_code=404, detail=f"no saved plan for task_id: {regen.task_id}")
    if regen.seed is not None and not 0 <= regen.seed < 2**63:
        raise HTTPException(status_code=422, detail="seed must be an integer in [0, 2**63)")
    job = Job("regenerate", {"seed": regen.seed}, parent_task_id=regen.task_id)
    result = await _enqueue(app, job)
    result["parent_task_id"] = regen.task_id
    return wrap_response(result)


@app.post("/redecode_task")
async def redecode_task(request: Request, authorization: Optional[str] = Header(None)):
    """保存済み latent.npy から音声のみ再デコードする (AR/NARの再実行なし)."""
    body = await _read_body(request)
    _verify_token(body, authorization)
    redecode = RedecodeTaskRequest(**{k: v for k, v in body.items() if k in RedecodeTaskRequest.model_fields})
    _require_completed_parent(redecode.task_id)
    parent_dir = OUTPUT_ROOT / redecode.task_id
    if not (parent_dir / "latent.npy").is_file():
        raise HTTPException(status_code=404, detail=f"no saved latent for task_id: {redecode.task_id}")
    job = Job("redecode", {}, parent_task_id=redecode.task_id)
    result = await _enqueue(app, job)
    result["parent_task_id"] = redecode.task_id
    return wrap_response(result)


@app.post("/query_result")
async def query_result(request: Request, authorization: Optional[str] = Header(None)):
    body = await _read_body(request)
    _verify_token(body, authorization)
    raw = body.get("task_id_list", "[]")
    task_ids: List[str] = json.loads(raw) if isinstance(raw, str) else list(raw)
    data_list = []
    for task_id in task_ids:
        job = JOBS.get(task_id)
        if job is None:
            data_list.append({"task_id": task_id, "status": 2,
                              "result": None, "message": "not found"})
            continue
        data_list.append({
            "task_id": task_id,
            "status": STATUS_MAP.get(job.status, 2),
            "status_text": job.status,
            "kind": job.kind,
            "result": job.result,
            "message": job.error,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        })
    return wrap_response(data_list)


@app.get("/v1/models")
async def list_models():
    return wrap_response({
        "models": [{"id": MODEL_ID, "revision": MODEL_REVISION or None,
                    "vae": VAE_ID, "vae_revision": VAE_REVISION or None,
                    "ready": PIPELINE.ready,
                    "loading": PIPELINE.loading, "load_error": PIPELINE.load_error,
                    "backend": BACKEND, "keep_vllm": KEEP_VLLM, "device": DEVICE}],
    })


@app.get("/v1/audio")
async def download_audio(path: str, authorization: Optional[str] = Header(None)):
    _verify_token({}, authorization)
    requested = Path(urllib.parse.unquote(path))
    resolved = (requested if requested.is_absolute() else OUTPUT_ROOT / requested).resolve()
    if not resolved.is_relative_to(OUTPUT_ROOT):
        raise HTTPException(status_code=403, detail="Path outside output directory")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media = {"flac": "audio/flac", "wav": "audio/wav",
             "abc": "text/plain", "json": "application/json"}
    return FileResponse(resolved, media_type=media.get(resolved.suffix.lstrip("."),
                                                       "application/octet-stream"))


@app.get("/health")
async def health():
    return wrap_response({
        "status": "ok",
        "model_ready": PIPELINE.ready,
        "model_loading": PIPELINE.loading,
        "load_error": PIPELINE.load_error,
        "profile": PROFILE,
        "backend": BACKEND,
        "vllm_quant": VLLM_QUANT or None,
        "offload_ar": OFFLOAD_AR,
        "queue_size": app.state.job_queue.qsize(),
        "jobs": {status: sum(1 for j in JOBS.values() if j.status == status)
                 for status in ("queued", "running", "succeeded", "failed")},
    })


def main() -> None:
    import uvicorn
    host = os.environ.get("YUE2_API_HOST", "127.0.0.1")
    port = int(os.environ.get("YUE2_API_PORT", "8002"))
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
