import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from veil.api_server import ApiServer
from veil.config.pipeline import PipelineConfig, PipelineMode
from veil.datahandler import DataHandler
from veil.logger import init_logger, set_log_level
from veil.pipeline import Pipeline

logger = init_logger(__name__)


class OfflineRunner:
    """Offline mode runner."""

    def __init__(self, cfg: PipelineConfig):
        self.pipeline = Pipeline(cfg)

        # If metric store is enabled, redirect DataHandler output into the run directory
        dh_cfg = cfg.datahandler
        try:
            if self.pipeline.metric_store is not None:
                run_dir = self.pipeline.metric_store.run_dir
                # Preserve user-provided filename if any; otherwise derive from input
                if getattr(dh_cfg, "output_path", None):
                    out_name = Path(dh_cfg.output_path).name
                else:
                    try:
                        in_stem = Path(dh_cfg.input_path).expanduser().resolve().stem  # type: ignore[arg-type]
                        out_name = f"{in_stem}.masking_results.jsonl"
                    except Exception:
                        out_name = "masked_output.jsonl"
                redirected_out = run_dir / out_name
                if str(getattr(dh_cfg, "output_path", "")) != str(redirected_out):
                    logger.info(
                        "Metric store enabled; redirecting DataHandler output to %s",
                        str(redirected_out),
                    )
                dh_cfg = replace(dh_cfg, output_path=str(redirected_out))
        except Exception:
            logger.exception(
                "Failed to align DataHandler output path with metric store run directory"
            )

        self.datahandler = DataHandler(dh_cfg)

    def run(self):
        concurrency = max(1, int(getattr(self.pipeline.config, "concurrency", 1)))
        if concurrency == 1:
            for doc in self.datahandler:
                result = self.pipeline.process(doc)
                self.datahandler.write_result(doc, result)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {}
                for doc in self.datahandler:
                    fut = executor.submit(self.pipeline.process, doc)
                    futures[fut] = doc
                for fut in as_completed(futures):
                    doc = futures[fut]
                    try:
                        result = fut.result()
                        self.datahandler.write_result(doc, result)
                    except Exception:
                        logger.exception(
                            "Failed processing document %s",
                            getattr(doc, "doc_id", None),
                        )
        # finalize metrics and plots at end of offline run
        try:
            if self.pipeline.metric_store:
                self.pipeline.metric_store.finalize()
        except Exception:
            logger.exception("Failed to finalize metrics.")


class APIRunner:
    """Online mode runner."""

    def __init__(self, cfg: PipelineConfig):
        self.pipeline = Pipeline(cfg)
        self.server = ApiServer(self.pipeline, cfg.api_server)
        self.server._build_app()

    def run(self):
        self.server.run()


def main() -> None:
    """Parse CLI / YAML and execute one or many pipeline runs."""

    # Allow runtime override of log verbosity
    lvl = os.environ.get("VEIL_LOG_LEVEL")
    if lvl:
        try:
            set_log_level(lvl)
        except Exception:
            pass

    configs = PipelineConfig.create_from_cli_args()
    logger.info("Running %d pipeline configuration(s).", len(configs))
    for cfg in configs:
        mode = PipelineMode.from_str(cfg.mode.lower())
        if mode == PipelineMode.ONLINE:
            if len(configs) > 1:
                raise ValueError(
                    "Online mode does not support multiple configurations."
                )
            runner = APIRunner(cfg)
        else:
            runner = OfflineRunner(cfg)
        logger.info(f"Running pipeline in {mode.name} mode with config: {cfg}")
        runner.run()


if __name__ == "__main__":  # pragma: no cover
    main()
