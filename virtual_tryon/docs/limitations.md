# Current Limitations

- Fine garment logos, typography, patterns, and small accessories can be distorted.
- Large pose changes and unusual camera angles reduce garment alignment quality.
- Occluded torso regions are difficult to reconstruct reliably.
- Long hair and hands overlapping the garment can cause mask or boundary artifacts.
- The original garment can remain visible near the lower torso or hemline when the upper-body agnostic mask does not cover the full old garment.
- The optional refiner can over-edit identity, background, or garment details despite mask constraints.
- Klein Try-On LoRA is experimental and may over-edit identity, body shape, hands, pants, or background, especially when prompts are too broad.
- Klein Try-On LoRA depends on three image references: person, top garment, and bottom garment. Upper-body samples without a bottom reference use an auto-crop from the person image, which can bias results toward preserving the original lower garment.
- FLUX.2 Klein local model access depends on Hugging Face license acceptance, local storage, compatible Diffusers dependencies, and GPU memory. The optional fal.ai backend also depends on external credentials and endpoint availability.
- Runtime and output behavior depend on third-party checkpoint availability and model licenses.
- The demo currently has no authentication, tenant isolation, quotas, or per-user artifact access control.
- Cancellation is cooperative: an active GPU process finishes before the job becomes fully cancelled.
- Timeout enforcement marks an overlong attempt as failed after it returns; it does not terminate the model subprocess yet.
