# Veil

Veil is a framework for building and running Named Entity Recognition & masking pipelines.

## Why Veil?

Named Entity Recognition (NER) is the process of identifying and classifying named entities in text (names, organizations, locations, etc.). Although public models and techniques perform well, a production-ready NER system can be comprised of many components, not just a standalone model. Veil provides a framework to build and run production-tested NER pipelines with many different components. It offers primitives for defining a pipeline that may include:

- Multiple entity detectors that identify entities in the text, extracting the most out of many different NER techniques and models
- Entity resolvers that resolve entities to a single canonical form
- Overlap resolvers that resolve overlaps between entities
- Maskers that mask the entities in the text for privacy-focused use cases
- Evaluators that exhaustively measure the pipeline, both in quality of detection and system performance

Veil can be deployed in `online` mode as an API server, or in `offline` mode as a batch processor.

## Getting Started

First, clone the repository:

```bash
git clone https://github.com/chus-chus/veil.git
cd veil
```

### Environment setup (supports all included entity detectors)

First, install `make` and `mamba` if you don't have them already. Then, build the environment:

```bash
make build
```

activate the environment with:

```bash
mamba activate ./env
```

Note that you will need CUDA for GPU model execution. If not available, Veil will fall back to CPU execution.

### Development packages

You may also need development requirements (build documentation, run tests, etc.). Inside the environment:

```bash
python -m uv pip install -r requirements_dev.txt
```

## Documentation

You can extract the most out of Veil when you bring your own entity detectors. To learn how, you can read the documentation on [Read the Docs](https://veil-project.readthedocs.io/en/latest/). 

You can also build the documentation by running:

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
python -m veil --help
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
python -m veil --pipeline-config-from-file run_configs/example_offline.yml
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
