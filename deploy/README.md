# Deployment Skeleton

This folder is a starting point for demo deployments. It intentionally does not copy model weights, checkpoints, generated outputs, or third-party repositories into container images.

## Backend

`Dockerfile.backend` uses a CUDA runtime placeholder image and installs the FastAPI backend from `virtual_tryon/pyproject.toml` and `uv.lock`. For production IDM-VTON inference, adjust the CUDA/PyTorch base image to match the RunPod environment and mount:

- `virtual_tryon/models`
- `virtual_tryon/third_party`
- `virtual_tryon/data/outputs`
- `virtual_tryon/data/eval_set`

## Frontend

`Dockerfile.frontend` builds the Vite app and serves it with Vite preview.

## Compose

```bash
cd deploy
docker compose up --build
```

Ports:

- backend: `8000`
- frontend: `5173`

For RunPod, the native scripts in `virtual_tryon/scripts/` are usually simpler than Docker while iterating.
