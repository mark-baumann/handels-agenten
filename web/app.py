"""Small HTTP interface for launching and reading TradingAgents analyses."""

from __future__ import annotations

import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

STATIC_DIR = Path(__file__).parent / "static"
TICKER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-^=]{0,19}$")
ANALYSTS = {"market", "social", "news", "fundamentals"}

app = FastAPI(title="TradingAgents", version="0.3.1")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AnalysisRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=20)
    trade_date: date
    analysts: list[str] = Field(default_factory=lambda: sorted(ANALYSTS))
    asset_type: str = "stock"


class AnalysisStore:
    """In-memory job store. A single worker avoids concurrent config mutation."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")

    def start(self, request: AnalysisRequest) -> str:
        job_id = uuid.uuid4().hex
        with self.lock:
            self.jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "ticker": request.ticker.upper(),
                "trade_date": request.trade_date.isoformat(),
                "result": None,
                "error": None,
            }
        self.executor.submit(self._run, job_id, request)
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return job.copy() if job else None

    def _run(self, job_id: str, request: AnalysisRequest) -> None:
        with self.lock:
            self.jobs[job_id]["status"] = "running"
        try:
            graph = TradingAgentsGraph(
                selected_analysts=request.analysts,
                config=DEFAULT_CONFIG.copy(),
            )
            state, signal = graph.propagate(
                request.ticker.upper(), request.trade_date.isoformat(), request.asset_type
            )
            result = {
                "signal": str(signal),
                "final_decision": state.get("final_trade_decision", ""),
                "reports": {
                    name: state.get(name, "")
                    for name in ("market_report", "sentiment_report", "news_report", "fundamentals_report")
                    if state.get(name)
                },
            }
            with self.lock:
                self.jobs[job_id].update(status="completed", result=result)
        except Exception as exc:  # Surface provider/data failures in the UI instead of hanging.
            with self.lock:
                self.jobs[job_id].update(status="failed", error=str(exc))


store = AnalysisStore()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyses", status_code=202)
def create_analysis(request: AnalysisRequest) -> dict[str, str]:
    ticker = request.ticker.strip().upper()
    if not TICKER_PATTERN.fullmatch(ticker):
        raise HTTPException(422, "Bitte gib ein gültiges Börsenkürzel ein, z. B. AAPL oder BTC-USD.")
    if request.trade_date > date.today():
        raise HTTPException(422, "Das Analysedatum darf nicht in der Zukunft liegen.")
    if not request.analysts or not set(request.analysts).issubset(ANALYSTS):
        raise HTTPException(422, "Wähle mindestens einen gültigen Analysten aus.")
    if request.asset_type not in {"stock", "crypto"}:
        raise HTTPException(422, "Asset-Typ muss stock oder crypto sein.")
    request.ticker = ticker
    return {"id": store.start(request)}


@app.get("/api/analyses/{job_id}")
def get_analysis(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Analyse nicht gefunden.")
    return job
