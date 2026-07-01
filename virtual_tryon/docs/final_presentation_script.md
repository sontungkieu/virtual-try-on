# Final Presentation Script

## 0:00-0:40 - Problem And Goal

"This project is a production-oriented Virtual Try-On demo. It takes a person image and a reference garment, then generates a new image in which the person wears that garment. The key requirement is controlled change: preserve identity, pose, body shape, and background while transferring the garment's structure, color, and texture."

Show the person and garment inputs in the frontend. Mention that the category and prompt make the request explicit.

## 0:40-1:30 - Architecture

"The default core is IDM-VTON because it is designed for garment-conditioned synthesis. Before inference, the backend validates the uploads and prepares parsing, DensePose, and garment masks. The adapter builds a one-sample VITON-HD-style dataset, runs the isolated IDM-VTON implementation, and captures its command and logs."

"FLUX is optional and comes after the core. It only attempts local refinement. If it is unavailable, fails, or makes the image worse, the quality gate keeps the IDM-VTON output. This avoids making an unrestricted editor responsible for the entire try-on task."

Show the architecture diagram in `docs/architecture.md` or briefly point to the core/refiner flow.

## 1:30-2:30 - Live Workflow

Submit the sample with asynchronous mode and `use_refiner=false`.

"The API immediately returns a job ID. The job system records queued, running, and completed or failed states, while serializing GPU work to avoid overlapping large inference processes. The frontend polls the job and shows this lifecycle as a timeline."

When completed, show `result.png`, the job ID, and the artifact list. Open the quality section.

"Every run is reproducible as an artifact folder. It includes the core output, selected result, masks, quality report, manifest, metadata, and stdout/stderr logs. API failures use a structured error format and do not expose a raw stack trace."

## 2:30-3:20 - Operations And Safety

Open `/health`, `/system`, and `/metrics`.

"The health endpoint checks engine readiness, including checkpoints. The system endpoint records Python, PyTorch, CUDA, GPU, and commit details. Metrics expose job volume, runtime, failures, queue size, GPU memory, and artifact bytes for Prometheus-style monitoring."

"Uploads are limited by size and MIME type and must decode as real images. Artifact serving uses an extension allowlist and a fixed output root. Tokens, checkpoints, third-party repositories, outputs, temporary data, dependencies, and browser traces are never committed."

## 3:20-4:10 - Verification And Limitations

Show the release check summary.

"The release candidate passed 61 backend tests, the frontend production build, Playwright, and a real IDM-VTON API smoke test on an RTX 3090 Ti. The real run completed in 49.104 seconds. GitHub CI is intentionally mock-only because hosted runners do not contain the GPU or checkpoints."

"The main limitations are fine logos and text, difficult poses, and hair or hands crossing the garment. Cancellation is cooperative, FLUX can still over-edit, and the demo does not yet provide user authentication or tenant isolation. These limits are documented rather than hidden."

## 4:10-4:40 - Close

"The result is more than a model notebook: it is an inspectable try-on service with asynchronous execution, artifacts, quality evidence, operational endpoints, a usable frontend, and a repeatable release process. The next priority is a larger, more diverse evaluation set and stronger parser-aware masks before adding broader product features."
