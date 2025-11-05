# Veil

Veil is a framework for building and running text processing and masking pipelines.

## Getting Started

### Environment setup (supports all included entity detectors)

```bash
make build
```

activate the environment with:

```bash
mamba activate ./env
```

Note that you will need CUDA for GPU model execution. If not available, veil will fall back to CPU execution.

### Development packages

You may also need development requirements (build documentation, run tests, etc.). Inside the environment:

```bash
python3 -m pip install -r requirements_dev.txt
```

## Documentation

You can extract the most out of Veil when you bring your own entity detectors. To learn how, you can read the documentation. To build it, run:

```bash
make docs/html
```

and serve it locally with:

```bash
make docs/serve
```

which will start a local server at `http://localhost:5500`.

## Run from the CLI

Veil is highly configurable. All configuration classes, defined in `veil/config`, have a 1–1 mapping with CLI parameters.
You can see the available options with:

```bash
python3 -m veil --help
```

## Run from a file

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

Input data must contain at least an `input` field with the text to process.

See `docs/architecture.md` for more details.

## Docker

### Run the Veil API with Docker

We provide a Docker image for a reproducible API deployment. See the configuration used in the image at `run_configs/online_multi_detector.yml`. You can also build the image yourself:

```bash
make docker/build
```

or pull it from Docker Hub:

```bash
docker pull docker-username/veil:gpu-latest
```

Then run:

```bash
docker run --gpus all -t -e HUGGINGFACE_HUB_TOKEN=hf_your_token -p 8000:8000 docker-username/veil:gpu-latest
```

This will start the API server on port 8000. See API details in `veil/api_server.py`.
