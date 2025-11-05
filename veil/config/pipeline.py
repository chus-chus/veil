from __future__ import annotations

"""Pipeline configuration dataclasses built on Veil’s frozen-dataclass config system.
"""

from dataclasses import field
from typing import List, Optional

from veil.config.api_server import ApiServerConfig
from veil.config.core.flat_dataclass import create_flat_dataclass
from veil.config.core.frozen_dataclass import frozen_dataclass
from veil.config.datahandler import DataHandlerConfig
from veil.config.entity_detectors import BaseEntityDetectorConfig
from veil.config.entity_resolvers import BaseEntityResolverConfig
from veil.config.evaluator import EvaluatorConfig
from veil.config.masker import MaskerConfig
from veil.config.metric_store import MetricStoreConfig
from veil.config.overlap_resolver import OverlapResolverConfig
from veil.core.enums.pipeline_mode import PipelineMode
from veil.logger import init_logger, set_log_level

logger = init_logger(__name__)


@frozen_dataclass(allow_from_file=True)
class PipelineConfig:
    """Root configuration dataclass for a Veil masking run."""

    mode: str = field(
        default="offline",
        metadata={
            "help": "Mode of the pipeline (offline|online). Online exposes an API for real-time masking. 'Offline' is the default and batch processes files with the dataloader."
        },
    )
    datahandler: Optional[DataHandlerConfig] = field(
        default=None,
        metadata={
            "help": "Input/output data handler configuration. Required for offline mode."
        },
    )
    api_server: Optional[ApiServerConfig] = field(
        default=None,
        metadata={"help": "API server configuration. Required for online mode."},
    )
    entity_detectors: List[BaseEntityDetectorConfig] = field(
        default_factory=list,
        metadata={"help": "Entity detector configurations executed in order"},
    )
    entity_resolvers: Optional[List[BaseEntityResolverConfig]] = field(
        default=None,
        metadata={"help": "Optional entity resolvers executed after detection"},
    )
    metric_store: MetricStoreConfig = field(
        default_factory=MetricStoreConfig,
        metadata={"help": "Metric store configuration"},
    )
    masker: MaskerConfig = field(
        default_factory=MaskerConfig,
        metadata={"help": "Masker configuration"},
    )
    overlap_resolver: OverlapResolverConfig = field(
        default_factory=OverlapResolverConfig,
        metadata={"help": "Configuration for the OverlapResolver"},
    )
    evaluator: Optional[EvaluatorConfig] = field(
        default=None,
        metadata={
            "help": "Evaluator configuration. If absent, no evaluation is performed. "
            "Ground truth in input is required."
        },
    )
    concurrency: int = field(
        default=1,
        metadata={
            "help": "Number of documents to process in parallel in offline mode (threads)."
        },
    )
    log_level: str = field(
        default="INFO",
        metadata={
            "help": "Global log level for Veil (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
        },
    )

    def __post_init__(self):
        # Apply logging level as early as possible
        try:
            set_log_level(getattr(self, "log_level", "INFO"))
        except Exception:
            pass

        mode = PipelineMode.from_str(self.mode.lower())
        if mode == PipelineMode.OFFLINE:
            if self.datahandler is None:
                raise ValueError("Offline mode requires a datahandler config.")
        elif mode == PipelineMode.ONLINE:
            if self.api_server is None:
                logger.warning("No API server config provided, using default.")
                self.api_server = ApiServerConfig()
            self.datahandler = None

        # Eagerly load all entity type modules and validate alias map to fail fast on conflicts
        try:
            import pkgutil
            from importlib import import_module

            import veil.entity_detectors as detectors_pkg  # type: ignore[import-not-found]
            import veil.entity_resolvers as resolvers_pkg  # type: ignore[import-not-found]

            for _finder, modname, _ispkg in pkgutil.walk_packages(
                detectors_pkg.__path__, detectors_pkg.__name__ + "."
            ):
                if modname.endswith("_entity_type"):
                    try:
                        import_module(modname)
                    except Exception:
                        # Bubble up; importing an entity type module should not fail silently
                        raise

            # Trigger import to register resolvers at import-time
            for _finder, modname, _ispkg in pkgutil.walk_packages(
                resolvers_pkg.__path__, resolvers_pkg.__name__ + "."
            ):
                try:
                    import_module(modname)
                except Exception:
                    raise

            from veil.core.base_entity_type import EntityTypeBase

            _ = EntityTypeBase.global_alias_map()
        except Exception:
            # Re-raise to fail fast during configuration construction
            raise

    @classmethod
    def create_from_cli_args(cls):
        """Return one or many PipelineConfig instances based on CLI/YAML combos."""
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        instances = []
        for flat_config in flat_configs:
            instance = flat_config.reconstruct_original_dataclass()
            object.__setattr__(instance, "__flat_config__", flat_config)
            instances.append(instance)
        return instances
