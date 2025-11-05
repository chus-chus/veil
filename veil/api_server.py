from typing import Dict, List, Optional, Set
from uuid import uuid4

from fastapi import FastAPI, Response
from pydantic import BaseModel

from veil.config.api_server import ApiServerConfig
from veil.core.document import Document
from veil.logger import init_logger
from veil.pipeline import Pipeline

logger = init_logger(__name__)


class ApiServer:
    """API server for the pipeline."""

    def __init__(self, pipeline: Pipeline, config: ApiServerConfig):
        self.pipeline = pipeline
        self.config = config
        # readiness flag toggled once app is fully built
        self._ready: bool = False

    def _build_app(self):

        class MaskRequest(BaseModel):
            doc_id: str | None = None
            text: str
            # { entity_type: { id: [aliases...] } }
            entity_cache: Optional[Dict[str, Dict[int, List[str]]]] = None

        app = FastAPI()

        @app.get("/healthz")
        def healthz():
            return {"status": "ok"}

        @app.get("/readyz")
        def readyz():
            if self._ready:
                return {"status": "ready"}
            return Response(
                content='{"status":"starting"}',
                media_type="application/json",
                status_code=503,
            )

        @app.post("/mask")
        def mask(req: MaskRequest):
            doc_id = req.doc_id or f"req-{uuid4()}"
            logger.debug(
                "/mask request received",
                extra={"doc_id": doc_id, "text_len": len(req.text)},
            )
            doc = Document(text=req.text, doc_id=doc_id)
            # Convert list-based cache aliases to sets as expected by pipeline/resolvers
            cache_converted: Optional[Dict[str, Dict[int, Set[str]]]] = None
            if req.entity_cache:
                cache_converted = {}
                for etype, id_map in req.entity_cache.items():
                    inner: Dict[int, Set[str]] = {}
                    for k, v in id_map.items():
                        try:
                            kid = int(k)
                        except Exception:
                            continue
                        inner[kid] = set(v or [])
                    cache_converted[etype] = inner

            res = self.pipeline.process(doc, entity_cache=cache_converted)
            # serialise spans for JSON
            entities = [
                {
                    "start": int(s.start),
                    "end": int(s.end),
                    "entity_type": getattr(
                        getattr(s, "entity_type", None), "name", None
                    ),
                    "id": getattr(s, "id", None),
                    "confidence": getattr(s, "confidence", None),
                    "replacement": getattr(s, "replacement", None),
                }
                for s in res.entities
            ]
            # Log masked output (avoid logging original text to prevent PII leakage)
            try:
                logger.info(
                    f"/mask result for doc_id={res.doc_id}, entities_count={len(entities)}, masked_text_len={len(res.masked_text)}"
                )
            except Exception:
                # Never fail the request because of logging
                pass
            return {
                "doc_id": res.doc_id,
                "masked_text": res.masked_text,
                "entities": entities,
            }

        self.app = app
        # Mark server as ready after building routes and holding a constructed pipeline
        try:
            self._ready = True
        except Exception:
            self._ready = False

    def run(self):
        import uvicorn

        logger.info("Starting API server on %s:%s", self.config.host, self.config.port)
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
            access_log=True,
        )
