# Docker

## Build

**GPU** (CUDA 12.x host + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
docker compose build
# or: docker build -t layer-wise-integrated-gradients:gpu .
```

**CPU only**:

```bash
docker compose -f docker-compose.cpu.yml build
# or: docker build -f Dockerfile.cpu -t layer-wise-integrated-gradients:cpu .
```

## Run `lig explain`

```bash
mkdir -p output

# GPU
docker compose run --rm lig explain "The cat sat on the mat." \
  --model bert-base-uncased \
  --steps 16 \
  --granularity layer \
  --layers 0 \
  --target-tokens 1,2 \
  -o /output/attributions.json

# CPU
docker compose -f docker-compose.cpu.yml run --rm lig explain "Hello world" \
  --steps 4 --granularity layer --layers 0 --target-tokens 1 -o /output/out.json
```

Model weights are cached in the Docker volume `hf-cache` (first run downloads from Hugging Face).

## Interactive shell

```bash
docker compose run --rm --entrypoint bash lig
```
