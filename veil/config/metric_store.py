from dataclasses import field

from veil.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class MetricStoreConfig:

    enabled: bool = field(
        default=False,
        metadata={"help": "Enable metric collection and plotting."},
    )

    output_dir: str = field(
        default="veil_runs",
        metadata={"help": "Path to the directory where the metrics will be stored."},
    )
    save_config_json: bool = field(
        default=True,
        metadata={
            "help": "Whether to save the config of the pipeline as a JSON file in the output directory."
        },
    )
