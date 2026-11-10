"""FastAPI server for the STE-RLHF comparison app (contract C5)."""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from webapp import model_interface
from webapp.model_interface import GenerationSettings, ModelNotAvailableError

log = logging.getLogger("webapp")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
EXAMPLES_PATH = BASE_DIR / "examples.json"
EXAMPLES_V2_PATH = BASE_DIR / "examples_v2.json"
RESULTS_PATH = BASE_DIR.parent / "results" / "metrics" / "summary.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_interface.load_models()
    except NotImplementedError:
        log.warning("model interface not connected")
    except Exception:
        log.exception("load_models() failed; continuing without models")
    yield


app = FastAPI(title="Mini-RLHF", lifespan=lifespan)


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    max_new_tokens: int = Field(default=1024, ge=8, le=2048)
    greedy: bool = True
    temperature: float = Field(default=0.7, ge=0.1, le=1.5)

    @field_validator("prompt", mode="before")
    @classmethod
    def _strip_prompt(cls, v: Any) -> Any:
        # Strip first so the length limits apply to the stripped text.
        return v.strip() if isinstance(v, str) else v


def _ste(text: str) -> Optional[dict]:
    """STE report for one answer; None if the STE module is not available."""
    try:
        from ste_rlhf.ste.metrics import ste_report
    except ImportError:
        return None
    try:
        return ste_report(text)
    except Exception:
        log.exception("ste_report failed")
        return None


def _run_one(prompt: str, kind: str, settings: GenerationSettings) -> dict:
    try:
        r = model_interface.generate_response(prompt, kind, settings)
    except NotImplementedError:
        return {"ok": False, "error": "not_connected",
                "message": "Model not connected yet"}
    except ModelNotAvailableError as e:
        return {"ok": False, "error": "not_available",
                "message": str(e) or "Model not trained yet"}
    except Exception as e:
        log.exception("generation failed for %s", kind)
        return {"ok": False, "error": "failed",
                "message": str(e) or e.__class__.__name__}
    return {"ok": True, "model_name": r.model_name, "response": r.response,
            "reward": r.reward, "latency_ms": r.latency_ms,
            "num_tokens": r.num_tokens,
            "finished": bool(r.finished), "ste": _ste(r.response)}


@app.get("/api/health")
def health() -> dict:
    try:
        status: Optional[dict] = model_interface.get_status()
    except NotImplementedError:
        return {"connected": False, "status": None}
    except Exception:
        log.exception("get_status() failed")
        return {"connected": False, "status": None}
    return {"connected": True, "status": status}


@app.get("/api/examples")
def examples() -> dict:
    """Plain list of strings (v1 shape). If examples_v2.json exists, its items are also
    returned as "selected": [{"question", "label"?}] and "examples" lists their questions."""
    if EXAMPLES_V2_PATH.exists():
        with open(EXAMPLES_V2_PATH, encoding="utf-8") as f:
            sel = [e for e in json.load(f) if isinstance(e, dict) and e.get("question")]
        if sel:
            return {"examples": [e["question"] for e in sel], "selected": sel}
    with open(EXAMPLES_PATH, encoding="utf-8") as f:
        return {"examples": json.load(f)}


@app.get("/api/results")
def results():
    if not RESULTS_PATH.is_file():
        return JSONResponse({"error": "not_found", "message": "No results yet"}, status_code=404)
    try:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        log.exception("could not read %s", RESULTS_PATH)
        return JSONResponse({"error": "unreadable", "message": "Results file unreadable"}, status_code=404)


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict:
    settings = GenerationSettings(max_new_tokens=req.max_new_tokens,
                                  greedy=req.greedy, temperature=req.temperature)
    return {
        "prompt": req.prompt,
        "settings": {"max_new_tokens": req.max_new_tokens, "greedy": req.greedy,
                     "temperature": req.temperature},
        "base": _run_one(req.prompt, "base", settings),
        "tuned": _run_one(req.prompt, "tuned", settings),
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.middleware("http")
async def revalidate_static(request, call_next):
    """Make browsers re-check the page and static files on every load (cheap via ETag),
    so frontend edits show up without a hard refresh."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
