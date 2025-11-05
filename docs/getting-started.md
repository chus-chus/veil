# Getting Started

Veil is a framework for building and running text processing and masking pipelines.

## Installation

### Environment installation (support for all included entity detectors)

```bash
make build
```

Activate the environment with:

```bash
mamba activate ./env
```

### Development packages

You might also need the development requirements (to build documentation, run tests, etc.). Inside the environment:

```bash
python3 -m pip install -r requirements_dev.txt
```

## Documentation

You can build the documentation with:

```bash
make docs/html
```

and serve it locally with:

```bash
make docs/serve
```

which will start a local server at `http://localhost:5500`.

## Running from the CLI

Veil is highly configurable. All configuration classes, defined in `veil/config`, have a 1-1 mapping to CLI parameters.
You can see the available ones with:

```bash
python3 -m veil --help
```

## Running from a file

For example, create a configuration file like `run_configs/example_offline.yml`:

```yaml
mode: offline
dataloader:
  path: data/input/example.jsonl
entity_detectors:
  - type: regex
    min_confidence: 0.3
```

And run:

```bash
python3 -m veil --pipeline-config-from-file run_configs/example_offline.yml
```

The input data must contain at least an `input` field with the text to be processed.

See `docs/architecture.md` for more details.

## Docker

### Running the Veil API with Docker

We provide a Docker image for a reproducible deployment of the API. See the configuration used in the image at `run_configs/prod_pipeline_v1.yml`. You can also build the image yourself:

```bash
# Replace docker-username with your Docker Hub username
make docker/build
```

or download it from Docker Hub:

```bash
docker pull docker-username/veil:gpu-latest
```

Next, run:

```bash
docker run --gpus all -t -e HUGGINGFACE_HUB_TOKEN=hf_your_token -p 8000:8000 docker-username/veil:gpu-latest
```

This will start the API server on port 8000. See the API details in `veil/api_server.py`.
