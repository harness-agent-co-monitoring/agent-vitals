"""HTTP API and static frontend for the dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from vitals.dashboard.data import Dashboard, EpisodeBusy

STATIC = Path(__file__).parent / "static"


def create_app(dash: Dashboard) -> FastAPI:
    app = FastAPI(title="agent-vitals dashboard", docs_url=None, redoc_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=4096)

    def store(sid: str) -> str:
        if sid not in dash.stores:
            raise HTTPException(404, f"unknown store {sid}")
        return sid

    @app.get("/api/sources")
    def sources():
        return dash.sources()

    @app.get("/api/changes")
    def changes(since: int = 0):
        return dash.changes(since)

    @app.get("/api/stores/{sid}/episodes")
    def episodes(sid: str):
        return dash.episodes(store(sid))

    @app.get("/api/stores/{sid}/episodes/{eid}/readmap")
    def readmap(sid: str, eid: str):
        try:
            return dash.readmap(store(sid), eid)
        except EpisodeBusy as e:
            raise HTTPException(503, "the episode is being written; try again shortly") from e
        except KeyError as e:
            raise HTTPException(404, f"unknown episode {eid}") from e

    @app.get("/api/stores/{sid}/episodes/{eid}/text")
    def text(sid: str, eid: str, turn: int, section: str):
        try:
            return dash.section_text(store(sid), eid, turn, section)
        except KeyError as e:
            raise HTTPException(404, f"unknown episode {eid}") from e

    @app.get("/api/stores/{sid}/episodes/{eid}/explanations")
    def explanations(sid: str, eid: str):
        try:
            return dash.explanations(store(sid), eid)
        except KeyError as e:
            raise HTTPException(404, f"unknown episode {eid}") from e

    @app.get("/api/stores/{sid}/alarms")
    def alarms(sid: str):
        return dash.alarms(store(sid))

    @app.get("/api/stores/{sid}/episodes/{eid}/flags")
    def flags(sid: str, eid: str):
        try:
            return dash.flag_explanations(store(sid), eid)
        except KeyError as e:
            raise HTTPException(404, f"unknown episode {eid}") from e

    @app.get("/api/stores/{sid}/matrix")
    def matrix(sid: str):
        return dash.matrix(store(sid))

    @app.get("/api/monitor")
    def monitor(target: str, ref: str, threshold: float = Query(3.0, gt=0), harness_only: bool = True):
        return dash.monitor(store(target), store(ref), threshold, harness_only)

    @app.get("/api/runs/{rid}")
    def run(rid: str):
        if rid not in dash.runs:
            raise HTTPException(404, f"unknown run {rid}")
        return dash.evolution(rid)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
