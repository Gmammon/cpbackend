"""
ConjointPro Unified Backend Service
====================================
Combines OA Generator + ACA Engine into a single FastAPI application.
Deploy to Railway, Render, Fly.io, or any cloud VM.

Startup:
    cd backend
    pip install -r requirements.txt
    python main.py
    # or: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ── OA imports ──
from oa_engine import generate_design

# ── ACA imports ──
from aca_engine import (
    start_session, get_session, start_aca_session,
    record_answer, get_session_result, cleanup_expired,
    compute_choice_hit_rate, validate_mbc, generate_holdout_tasks,
    undo_last_answer,
)

# ── MaxDiff imports ──
from maxdiff_engine import (
    generate_design as md_generate_design,
    insert_duplicate_tasks as md_insert_dupes,
    mle_estimate as md_mle_estimate,
    count_model as md_count_model,
)


# ── Lifespan (replaces deprecated on_event) ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: start ACA session cleanup thread
    def cleanup_loop():
        while True:
            time.sleep(300)
            cleanup_expired()

    t = threading.Thread(target=cleanup_loop, daemon=True)
    t.start()
    yield
    # Shutdown: nothing to clean up (sessions are in-memory)


# ── Unified App ──

app = FastAPI(
    title="ConjointPro Backend",
    description="OA Generator + ACA Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — restrict in production
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ──

@app.get("/")
def health():
    return {"status": "ok", "services": ["oa-generator", "aca-service", "maxdiff"]}


# ══════════════════════════════════════════════
#  OA Generator Endpoints
# ══════════════════════════════════════════════

class AttributeRequest(BaseModel):
    name: str
    levels: list[str]


class DesignRequest(BaseModel):
    attributes: list[AttributeRequest] = Field(..., min_length=1, max_length=8)


class DesignResponse(BaseModel):
    design_matrix: list[list[int]]
    array_name: str
    is_fractional: bool
    total_trials: int
    error: Optional[str] = None


@app.post("/api/generate-oa", response_model=DesignResponse)
def generate_orthogonal_array(request: DesignRequest):
    """
    Generate an orthogonal array or D-optimal design for the given attributes.
    """
    attr_levels = [len(a.levels) for a in request.attributes]
    result = generate_design(attr_levels)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return DesignResponse(
        design_matrix=result["design_matrix"],
        array_name=result["array_name"],
        is_fractional=result["is_fractional"],
        total_trials=result["total_trials"],
    )


# ══════════════════════════════════════════════
#  ACA Service Endpoints
# ══════════════════════════════════════════════

class ACAStartRequest(BaseModel):
    survey_id: str
    respondent_id: str
    attributes: list
    upper_bound: float = 100.0
    max_questions: Optional[int] = None
    min_questions: Optional[int] = None
    convergence_threshold: float = 0.02
    consecutive_count: int = 3
    level_counts: Optional[list] = None
    early_stop: bool = False


class ACAAnswerRequest(BaseModel):
    session_id: str
    respondent_id: str
    rating: float


@app.post("/api/aca/start")
def aca_start(req: ACAStartRequest):
    """Create a new ACA session and return the first question."""
    if not req.attributes:
        raise HTTPException(400, "attributes cannot be empty")

    for attr in req.attributes:
        if 'name' not in attr or 'levels' not in attr:
            raise HTTPException(400, "each attribute must have 'name' and 'levels'")
        if len(attr['levels']) < 2:
            raise HTTPException(400, f"attribute '{attr['name']}' must have >= 2 levels")

    try:
        session_id = start_session(
            req.respondent_id, req.attributes, req.upper_bound,
            max_questions=req.max_questions, min_questions=req.min_questions,
            convergence_threshold=req.convergence_threshold,
            consecutive_count=req.consecutive_count,
            survey_id=req.survey_id, level_counts=req.level_counts,
            early_stop=req.early_stop,
        )
        question = start_aca_session(session_id)
        return {"session_id": session_id, "question": question}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/aca/answer")
def aca_answer(req: ACAAnswerRequest):
    """Submit a rating and get the next question + current estimates."""
    try:
        get_session(req.session_id, req.respondent_id)
    except ValueError as e:
        msg = str(e)
        if "expired" in msg:
            raise HTTPException(401, msg)
        if "does not belong" in msg:
            raise HTTPException(403, msg)
        raise HTTPException(404, msg)

    try:
        result = record_answer(req.session_id, req.rating)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/aca/undo")
def aca_undo(req: ACAAnswerRequest):
    """Undo the last answer and return the restored state."""
    try:
        get_session(req.session_id, req.respondent_id)
    except ValueError as e:
        msg = str(e)
        if "expired" in msg:
            raise HTTPException(401, msg)
        if "does not belong" in msg:
            raise HTTPException(403, msg)
        raise HTTPException(404, msg)

    try:
        result = undo_last_answer(req.session_id)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/aca/result/{session_id}")
def aca_result(session_id: str):
    """Get final utility estimates for a session."""
    from aca_engine import sessions
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    try:
        return get_session_result(session_id)
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════
#  Validation Endpoints
# ══════════════════════════════════════════════

class ChoiceValidationRequest(BaseModel):
    session_id: str
    respondent_id: str
    tasks: list  # [{"task_index": 0, "options": [...], "respondent_choice": "opt1"}]


class MBCValidationRequest(BaseModel):
    session_id: str
    respondent_id: str
    ideal_config: dict       # {"brand": "A", "price": "cheap"}
    real_products: list       # [{"id": "p1", "name": "...", "levels": {...}}]
    final_choice: str         # "p3"


class HoldoutGenerateRequest(BaseModel):
    attributes: list
    n_tasks: int = 3
    n_options: int = 3
    seed: Optional[int] = None


@app.post("/api/aca/validate/choice")
def aca_validate_choice(req: ChoiceValidationRequest):
    """Compute hit rate for choice validation."""
    try:
        sess = get_session(req.session_id, req.respondent_id)
    except ValueError as e:
        msg = str(e)
        if "expired" in msg:
            raise HTTPException(401, msg)
        if "does not belong" in msg:
            raise HTTPException(403, msg)
        raise HTTPException(404, msg)

    result = get_session_result(req.session_id)
    utilities = result["utilities"]

    return compute_choice_hit_rate(utilities, req.tasks)


@app.post("/api/aca/validate/mbc")
def aca_validate_mbc(req: MBCValidationRequest):
    """Validate MBC: predict best product, compare with actual choice."""
    try:
        sess = get_session(req.session_id, req.respondent_id)
    except ValueError as e:
        msg = str(e)
        if "expired" in msg:
            raise HTTPException(401, msg)
        if "does not belong" in msg:
            raise HTTPException(403, msg)
        raise HTTPException(404, msg)

    result = get_session_result(req.session_id)
    utilities = result["utilities"]

    return validate_mbc(utilities, req.real_products, req.ideal_config, req.final_choice)


@app.post("/api/aca/validate/generate-holdouts")
def aca_generate_holdouts(req: HoldoutGenerateRequest):
    """Generate random holdout choice tasks for validation."""
    return generate_holdout_tasks(req.attributes, req.n_tasks, req.n_options, req.seed)


# ══════════════════════════════════════════════
#  MaxDiff Endpoints
# ══════════════════════════════════════════════

class MaxDiffDesignRequest(BaseModel):
    items: list[str] = Field(..., min_length=3)
    set_size: int = Field(default=4, ge=2, le=10)
    appearances: int = Field(default=4, ge=2, le=20)
    seed: Optional[int] = None
    insert_duplicates: bool = True


class MaxDiffDesignResponse(BaseModel):
    tasks: list[list[str]]
    duplicate_pairs: list[dict]
    metrics: dict
    seed: int
    method: str


class MaxDiffAnalyzeRequest(BaseModel):
    responses: list  # [{items: [...], best: "x", worst: "y"}, ...]
    items: list[str]


class MaxDiffAnalyzeResponse(BaseModel):
    utilities: dict
    scores: dict
    standard_errors: dict
    log_likelihood: float
    rlh: float
    random_rlh: float
    rlh_ratio: float
    iterations: int
    converged: bool


class MaxDiffCountRequest(BaseModel):
    responses: list
    items: list[str]


class MaxDiffCountResponse(BaseModel):
    best_count: dict
    worst_count: dict
    diff: dict
    scores: dict


@app.post("/api/maxdiff/generate-design", response_model=MaxDiffDesignResponse)
def maxdiff_generate_design(req: MaxDiffDesignRequest):
    """Generate a BIBD MaxDiff design."""
    if req.set_size > len(req.items):
        raise HTTPException(400, "set_size cannot exceed number of items")

    result = md_generate_design(
        items=req.items,
        set_size=req.set_size,
        appearances=req.appearances,
        seed=req.seed,
    )
    if result is None:
        raise HTTPException(500, "Failed to generate design")

    tasks = result["tasks"]
    duplicate_pairs = result["duplicate_pairs"]

    if req.insert_duplicates:
        tasks, duplicate_pairs = md_insert_dupes(tasks)

    return MaxDiffDesignResponse(
        tasks=tasks,
        duplicate_pairs=duplicate_pairs,
        metrics=result["metrics"],
        seed=result["seed"],
        method=result["method"],
    )


@app.post("/api/maxdiff/analyze", response_model=MaxDiffAnalyzeResponse)
def maxdiff_analyze(req: MaxDiffAnalyzeRequest):
    """Run MLE analysis on MaxDiff responses."""
    if not req.responses:
        raise HTTPException(400, "responses cannot be empty")

    result = md_mle_estimate(req.responses, req.items)
    if result is None:
        raise HTTPException(400, "No valid tasks found in responses")

    return MaxDiffAnalyzeResponse(**result)


@app.post("/api/maxdiff/count-model", response_model=MaxDiffCountResponse)
def maxdiff_count(req: MaxDiffCountRequest):
    """Run count model (B-W scores) on MaxDiff responses."""
    if not req.responses:
        raise HTTPException(400, "responses cannot be empty")

    result = md_count_model(req.responses, req.items)
    return MaxDiffCountResponse(**result)


# ── Entry Point ──

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
