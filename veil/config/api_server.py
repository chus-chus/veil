from dataclasses import field

from veil.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class ApiServerConfig:
    host: str = field(
        default="0.0.0.0", metadata={"help": "Host to bind the API server to."}
    )
    port: int = field(
        default=8000, metadata={"help": "Port to bind the API server to."}
    )
    reload: bool = field(
        default=False, metadata={"help": "Reload the API server on code changes."}
    )
