# Tổng quan hệ thống Virtual Try-On

Tài liệu này giải thích cách hệ thống Virtual Try-On hoạt động từ giao diện web, backend, các engine sinh ảnh, ComfyUI workflow, artifacts, history đến hiệu năng. Mục tiêu là để khi xem một kết quả, mình biết nó đi qua pipeline nào, lỗi nên soi ở đâu, và vì sao mode `IDM + Klein hybrid` thường tốt hơn cho các case innerwear.

## 1. Ý tưởng chính

Hệ thống không chỉ gọi một model sinh ảnh. Nó là một pipeline gồm nhiều lớp:

1. Frontend React để upload ảnh người, ảnh trang phục, chọn category, engine, seed, resolution và steps.
2. Backend FastAPI để nhận job, tạo mask động, chọn engine, chạy model, lưu artifacts và trả kết quả.
3. Các engine: IDM-VTON, Klein LoRA local, IDM + Klein hybrid, Flux Redux + CatVTON qua ComfyUI, và một số baseline/experimental mode.
4. ComfyUI custom node để reproduce workflow bằng graph, nhưng node vẫn gọi lại backend project để dùng đúng pipeline đang test.
5. Mỗi job có thư mục riêng trong `data/outputs/{job_id}` để kiểm tra input, mask, prompt, raw output, result, metadata và timing.

Hai mục tiêu chính:

- Giữ đúng người, pose, mặt, tóc, da, background và các vùng không cần thay.
- Đưa được chi tiết trang phục tham chiếu vào đúng vùng cần thay, đặc biệt với innerwear adult non-sexual.

## 2. Các thành phần runtime

### Frontend

Frontend nằm trong:

```text
virtual_tryon/frontend
```

Đây là app React/Vite. Trên RunPod, frontend thường chạy ở port `8080`.

Frontend phụ trách:

- Upload ảnh person.
- Upload ảnh garment theo category.
- Chọn category: `Top`, `Bottom`, `Dress`, `Outfit`, `Men underwear`, `Women underwear`, `Bra`.
- Chọn engine mode.
- Chọn output resolution, steps, seed và deterministic mode.
- Gửi job lên backend.
- Poll job status.
- Hiển thị stage, timing, output, masks, artifacts và history.

Frontend chạy async: sau khi bấm Generate, UI nhận `job_id`, rồi poll backend để cập nhật `queued`, `running`, `generating`, `refining`, `completed`.

### Backend

Backend nằm trong:

```text
virtual_tryon/backend
```

Backend là FastAPI. Trên RunPod, backend thường chạy ở port `8000`.

Endpoint quan trọng:

```text
GET  /health
POST /tryon/model/prepare
POST /tryon
GET  /tryon/{job_id}
GET  /tryon/history
GET  /artifacts/{job_id}/{file_path}
```

Backend làm các việc chính:

- Validate input và tham số.
- Resize/normalize ảnh.
- Tạo prompt theo category và engine mode.
- Tạo mask động theo ảnh person và category.
- Gọi core engine.
- Composite output về ảnh person gốc nếu engine có xu hướng thay đổi ngoài vùng mask.
- Quality check cơ bản.
- Lưu artifacts và metadata.
- Ghi stage timing.
- Trả `result_url` và các debug URL.

### ComfyUI

ComfyUI nằm ở runtime remote:

```text
/workspace/ComfyUI
```

Custom node của project được link vào:

```text
/workspace/ComfyUI/custom_nodes/vton_phase2_nodes
```

Node quan trọng:

```text
VTONPhase2BackendTryOnAPI
```

Node này nhận ảnh trong ComfyUI, gửi request sang backend `/tryon`, đợi job hoàn thành, rồi trả output image và mask preview về graph.

Workflow reproduce nằm trong:

```text
virtual_tryon/comfyui_workflows/omnitry_innerwear_repro
```

Mỗi case thường có 2 file:

```text
*_api.json
*_ui.workflow.json
```

- `*_api.json`: queue qua ComfyUI `/prompt`.
- `*_ui.workflow.json`: load trực tiếp trong ComfyUI editor.

## 3. Luồng chạy của một job

Một job try-on đi qua luồng chính:

```text
person + garment + category + engine config
  -> validate upload và config
  -> normalize person/garment
  -> build prompt
  -> build dynamic mask
  -> build agnostic image
  -> chọn engine
  -> engine run
  -> composite/fuse nếu cần
  -> quality check
  -> optional refine
  -> save result và artifacts
  -> update history
```

Các stage async:

```text
queued
running
generating
refining
completed
```

Ý nghĩa:

- `queued`: backend đã nhận job, chờ GPU lock nếu có job khác đang chạy.
- `running`: preprocess input, mask, prompt, artifact folder.
- `loading_model`: load hoặc reuse model resident. Nếu model đã pin sẵn trong worker thì stage này ghi `skipped`.
- `generating`: core engine đang sinh ảnh.
- `refining`: optional refiner như FLUX refine. Nếu không dùng refine thì stage này ghi `skipped`.
- `completed`: lưu kết quả cuối, manifest, history.

Mỗi stage có `runtime_seconds` trong `job.json`.

## 4. Category và input slots

Category quyết định mask, prompt và garment slot nào được dùng.

| Category UI | Backend category | Garment slot chính | Mục tiêu |
|---|---|---|---|
| Top | `upper_body` | `garment_top` | Thay áo/phần trên |
| Bottom | `lower_body` | `garment_bottom` | Thay quần/phần dưới |
| Dress | `dress` | `garment_dress` | Thay đầm/váy liền |
| Outfit | `full_outfit` | top/bottom/dress | Bộ đồ nhiều món |
| Men underwear | `men_underwear` | `garment_bottom` | Thay quần lót nam adult non-sexual |
| Women underwear | `women_underwear` | `garment_bottom` | Thay quần lót nữ adult non-sexual |
| Bra | `women_bra` | `garment_top` | Thay bra/innerwear top nữ adult non-sexual |

Với innerwear:

- `women_underwear`: cần person + underwear bottom.
- `men_underwear`: cần person + underwear bottom.
- `women_bra`: cần person + bra/top.
- `dress`: cần person + dress.
- `full_outfit`: có thể dùng nhiều slot.

Hệ thống innerwear chỉ dành cho ảnh người lớn, dùng trong bối cảnh product try-on không khiêu dâm.

## 5. Dynamic mask

Mask là phần quyết định vùng nào được phép thay.

Backend tạo mask dựa trên:

- Kích thước ảnh person sau normalize.
- Body estimate hoặc silhouette nếu tách được người khỏi background.
- Category.
- Dilation/blur config.
- Cache key dựa trên ảnh person, category, resolution và mask config.

Artifacts liên quan:

```text
raw_mask.png
agnostic_mask.png
soft_mask.png
mask_preview.png
mask_innerwear_shape.png
mask_body_silhouette.png
mask_metadata.json
```

Ý nghĩa:

- `raw_mask.png`: mask hình học ban đầu.
- `agnostic_mask.png`: mask đưa vào engine.
- `soft_mask.png`: mask mềm dùng khi composite.
- `mask_preview.png`: overlay để xem nhanh trên UI/ComfyUI.
- `mask_innerwear_shape.png`: hình mask riêng cho innerwear.
- `mask_body_silhouette.png`: silhouette người nếu detect được.
- `mask_metadata.json`: bbox, nguồn mask, cache hit/miss, warnings.

Với innerwear, mask preview có thể rộng hơn vùng thay thật để đảm bảo xóa đồ cũ. Riêng mode hybrid còn tạo `hybrid_idm_delta_mask.png`, tức vùng pixel mà IDM thực sự đã thay đổi, rồi dùng vùng này để fuse chi tiết Klein.

## 6. Các engine mode

### IDM-VTON

IDM-VTON là core engine ổn định nhất cho garment-conditioned try-on.

Ưu điểm:

- Giữ pose, identity, da và background tốt.
- Ít thay đổi lung tung ngoài vùng garment.
- Phù hợp làm baseline production.

Nhược điểm:

- Có thể làm mờ logo, texture hoặc đường viền nhỏ.
- Với innerwear, form thường ổn nhưng chất liệu/viền có thể chưa sắc.

Backend stage input cho IDM theo dạng VITON-HD:

```text
idm_vton_dataset/test/image/person_0001.jpg
idm_vton_dataset/test/cloth/garment_0001.jpg
idm_vton_dataset/test/agnostic-mask/person_0001_mask.png
idm_vton_dataset/test/image-densepose/person_0001.jpg
test_pairs.txt
```

Nếu resident worker tắt, backend launch official IDM inference mỗi job. Nếu resident worker bật, backend giữ model trong worker process và gửi request qua JSONL, giúp warm job nhanh hơn.

Khi chọn engine trong UI, frontend gọi `POST /tryon/model/prepare`. Nút Generate bị khóa cho tới khi prepare trả `ready`, nên thời gian load model/compile không còn bị gộp mơ hồ vào chữ `Working` của job sinh ảnh. Nếu worker đã sẵn, prepare trả nhanh.

### IDM resident worker và torch_compile

Resident worker:

- Load IDM một lần.
- Giữ model trong VRAM.
- Job sau không phải launch `accelerate` và reload checkpoint.

`torch_compile` có thể giảm latency warm job, nhưng có cold compile cost.

Cần hiểu đúng:

- Cold compile chỉ xảy ra khi graph chưa được compile trong process hiện tại.
- Nếu worker còn sống, các job sau dùng graph đã compile.
- Nếu đổi resolution/shape, có thể sinh graph mới.
- Nếu kill worker để switch sang model khác, lúc quay lại IDM có thể phải warmup lại.

Con số kiểu `181.8s` là cold compile/warmup sau restart hoặc sau khi unload worker, không phải lần nào bấm Generate cũng mất từng đó. Warm job sau đó có thể chỉ còn vài giây.

Cache có thể giúp một phần:

```text
TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor
```

Nhưng cache disk không thay thế hoàn toàn resident process. Vẫn có overhead trace, guard và load graph khi process mới start. Cách nhanh nhất vẫn là giữ worker đã compile không bị kill.

### TensorRT cho IDM

TensorRT là hướng benchmark/experimental.

Điểm cần nhớ:

- VAE decode path ổn định hơn full UNet.
- Full UNet hoặc UNet encoder có thể crash/hang nếu TensorRT builder gặp partition khó.
- Cần benchmark theo GPU, shape, dtype và profile.
- Engine cache có thể giảm build lại, nhưng không đảm bảo luôn nhanh hơn eager hoặc torch_compile.

Vì vậy TensorRT hiện là option thử nghiệm, không phải mặc định production nhanh nhất.

### Klein LoRA local

Klein LoRA dùng local FLUX.2 Klein 9B + Try-On LoRA.

Ưu điểm:

- Giữ màu, texture, đường viền và chi tiết garment tốt hơn IDM trong nhiều case.
- Hữu ích khi garment reference có logo, pattern hoặc chất liệu rõ.

Nhược điểm:

- Là generative model mạnh nên nếu chạy unrestricted có thể thay đổi vùng ngoài garment.
- Có thể làm đen/mờ vùng da xung quanh nếu mask/composite không kiểm soát tốt.
- Chậm và tốn VRAM hơn warm IDM resident.

Mode local không cần `FAL_KEY`. `FAL_KEY` chỉ liên quan khi config trỏ sang backend/API external.

Klein local cũng dùng resident worker. Worker giữ pipeline theo cache key gồm base model, LoRA, device map, quantization và TensorRT profile. Chọn lại đúng preset thì lần gen sau bỏ qua load model; đổi từ `klein_lora` sang `klein_bnb_4bit` hoặc đổi ngược lại thì worker load lại preset mới. Khi chuyển giữa IDM và Klein, backend unload worker cũ để tránh giữ hai pipeline lớn trên GPU 24 GB.

Riêng hybrid IDM + Klein không thể đảm bảo cả IDM và Klein cùng pin trong VRAM 24 GB. UI vẫn prepare phase IDM trước, còn phase Klein có thể xuất hiện `loading_model` trong job.

Artifacts thường gặp:

```text
klein_lora_prompt.txt
klein_lora_request.json
klein_lora_result.png
local_worker_stdout.txt
local_worker_stderr.txt
```

### Klein LoRA bnb 4-bit

Mode `Klein LoRA bnb 4-bit` vẫn là Klein + Try-On LoRA. Nó không bỏ LoRA và không đổi sang model khác.

Khác biệt duy nhất so với `klein_lora` mặc định là backend ép runtime preset:

```text
device_map = cuda
quantization = bnb_4bit
quantize_components = ["transformer", "text_encoder"]
tensorrt_profile = none
```

Worker vẫn load base model `FLUX.2-klein-9B`, sau đó gọi `load_lora_weights(...)` với `flux-klein-tryon.safetensors` và set adapter `tryon`. Nói ngắn gọn: 4-bit là cách nạp base pipeline nhẹ hơn, còn Try-On LoRA vẫn được gắn lên pipeline.

Mode này phù hợp khi cần tốc độ như các run Klein 4 steps khoảng 39-42s trước đó. Đổi lại, 4-bit có thể kém ổn định/chất lượng hơn full/cpu-offload ở một số case khó.

### IDM + Klein hybrid

Đây là mode chất lượng cao hiện tại cho nhiều case innerwear.

Ý tưởng:

- IDM xác định vùng thay đúng và giữ body/background tốt.
- Klein đưa chi tiết garment tốt hơn.
- Hybrid lấy IDM làm base, rồi lấy detail của Klein chỉ trong vùng IDM thực sự đã thay đổi.

Luồng chạy:

```text
person + garment + mask
  -> run IDM
  -> composite IDM về person bằng mask = hybrid_idm_base.png
  -> run Klein
  -> composite Klein về person bằng mask = hybrid_klein_detail.png
  -> tính difference giữa person và IDM base
  -> tạo hybrid_idm_delta_mask.png
  -> composite Klein detail lên IDM base bằng delta mask
  -> hybrid_result.png/result.png
```

Artifacts quan trọng:

```text
hybrid_idm_raw.png
hybrid_idm_base.png
hybrid_klein_raw.png
hybrid_klein_detail.png
hybrid_idm_delta_mask.png
hybrid_result.png
```

Vai trò:

- `hybrid_idm_base.png`: output IDM đã composite về person. Thường giữ body/background tốt nhưng hơi mờ detail.
- `hybrid_klein_detail.png`: output Klein đã composite về person. Thường có garment detail tốt hơn.
- `hybrid_idm_delta_mask.png`: vùng pixel IDM thực sự thay đổi so với person. Đây là mask fuse cuối.
- `hybrid_result.png`: kết quả fuse.

Lý do mode này hiệu quả:

- Không lấy toàn bộ output Klein nên giảm rủi ro Klein làm đen hoặc sai vùng da ngoài garment.
- Không chỉ dùng IDM nên giữ được chi tiết garment hơn.
- Delta mask thường hẹp hơn mask debug ban đầu và bám sát vùng garment đã thay.

Case đã test gần nhất:

```text
input_undergarment/female/KGA1151-PK001_P6_15.webp
workflow: female_underwear_02_kga1151_pk001_p6_15_idm_klein_hybrid_api.json
engine: idm_klein_hybrid
seed: 2026070203
resolution: 512x768
steps: 4
```

Kết quả đo trên RunPod:

```text
ComfyUI wall time: khoảng 95s
Backend finalize: khoảng 89.1s
Generation stage: khoảng 86.4s
IDM-VTON part: khoảng 25.5s
Refine: skipped
```

Nhận xét case này:

- IDM base tạo form đúng nhưng hơi mờ.
- Klein detail đưa lại màu đỏ, texture và viền hông rõ hơn.
- Hybrid final không bị đen vùng da xung quanh.
- Mask preview rộng hơn vùng thay thật, nhưng delta mask hybrid co lại đúng vùng underwear.

### Flux Redux + CatVTON qua ComfyUI

Mode này gửi mask động, person và garment reference qua local ComfyUI graph.

Ưu điểm:

- Reproduce được workflow bằng ComfyUI.
- Có thể kết hợp Flux Fill/Redux với CatVTON.

Nhược điểm:

- Phụ thuộc ComfyUI server port `8188`.
- Phụ thuộc model files trong ComfyUI.
- Runtime và memory phụ thuộc graph.

Nếu ComfyUI tắt, mode này sẽ fail hoặc báo unavailable.

## 7. Nên dùng mode nào trên UI

### Test nhanh và ổn định

```text
Engine: IDM-VTON
Resolution: 512x768
Steps: 4
Debug: off nếu chỉ cần output
```

Phù hợp để kiểm tra pipeline sống, test standard top/bottom/dress, hoặc cần warm latency thấp.

### Innerwear chất lượng cao

```text
Category: Men underwear / Women underwear / Bra
Engine: IDM + Klein hybrid
Resolution: 512x768 trước
Steps: 4 trước
Deterministic: on nếu cần reproduce
Debug: on nếu cần xem mask/artifacts
```

Phù hợp khi IDM giữ người tốt nhưng detail garment mờ, còn Klein giữ detail tốt nhưng có nguy cơ thay đổi ngoài vùng mong muốn.

Nếu cần bản hybrid nhanh hơn sau warmup, dùng:

```text
Engine: IDM compile + Klein 4-bit pro
```

Mode này ép IDM dùng resident worker `torch_compile` và ép Klein dùng preset bnb 4-bit kèm Try-On LoRA. Job đầu sau restart/model switch vẫn có thể lâu vì IDM compile, nhưng các job warm sau đó nhanh hơn.

### So chi tiết garment thuần Klein

```text
Engine: Klein LoRA experimental
```

Dùng để xem Klein có đọc đúng màu, texture và form garment không. Nếu Klein làm sai da/background, dùng hybrid thay vì dùng Klein raw làm output cuối.

Nếu muốn so phiên bản nhanh/ít VRAM hơn, dùng:

```text
Engine: Klein LoRA bnb 4-bit
```

### Reproduce bằng ComfyUI

Load workflow:

```text
virtual_tryon/comfyui_workflows/omnitry_innerwear_repro/*_idm_klein_hybrid_ui.workflow.json
```

Hoặc queue file API JSON qua ComfyUI `/prompt`.

## 8. Artifacts và cách đọc kết quả

Mỗi job có folder:

```text
data/outputs/{job_id}
```

File quan trọng:

| File | Ý nghĩa |
|---|---|
| `person.png` | Ảnh person đã normalize |
| `garment.png` | Ảnh garment đã normalize |
| `prompt.txt` | Prompt chính |
| `prompt_core.txt` | Prompt đưa vào core engine |
| `raw_mask.png` | Mask ban đầu |
| `agnostic_mask.png` | Mask đưa vào engine |
| `soft_mask.png` | Mask mềm để composite |
| `mask_preview.png` | Overlay xem nhanh |
| `core_output.png` | Output core engine sau composite |
| `core_output_raw.png` | Output raw của engine nếu có |
| `result.png` | Output cuối |
| `quality_report.json` | Heuristic quality |
| `artifact_manifest.json` | Danh sách artifact |
| `job.json` | Trạng thái, timing, debug URLs |

Với hybrid, xem thêm:

```text
hybrid_idm_base.png
hybrid_klein_detail.png
hybrid_idm_delta_mask.png
hybrid_result.png
```

Cách debug nhanh:

1. Vùng thay sai: xem `mask_preview.png`, `mask_innerwear_shape.png`, `mask_body_silhouette.png`.
2. IDM mờ detail: xem `hybrid_idm_base.png`.
3. Klein làm sai da/background: xem `hybrid_klein_detail.png` và `hybrid_idm_delta_mask.png`.
4. Final không lấy đủ chi tiết Klein: delta mask có thể quá hẹp.
5. Final ăn ra ngoài garment: delta mask hoặc soft mask có thể quá rộng.

## 9. History

History backend đọc từ các job folder trong `data/outputs`.

Endpoint:

```text
GET /tryon/history?limit=100&gender=woman&success_only=true
```

Filter:

- `gender=all`
- `gender=man`
- `gender=woman`
- `success_only=true`

Backend lọc trước rồi mới limit. Nghĩa là tab `Woman` lấy 100 job woman mới nhất, không phải lấy 100 job all rồi lọc woman ở frontend.

UI có:

- All
- Man
- Woman
- Success only

Khi job kết thúc, UI reload history.

## 10. Determinism và seed

UI có seed và deterministic best-effort.

Deterministic trong diffusion không đảm bảo tuyệt đối nếu khác:

- GPU/CUDA/PyTorch.
- Dtype.
- Scheduler/model version.
- Resolution/shape.
- Kernel non-deterministic.
- Worker restart hoặc graph compile khác.

Nhưng trên cùng remote, cùng code, cùng model, cùng seed, cùng resolution và steps, kết quả nên reproduce được tương đối ổn.

ComfyUI workflows trong `omnitry_innerwear_repro` đã ghi seed cố định cho từng case.

## 11. Hiệu năng

### IDM warm vs cold

Nếu IDM resident worker đã load và warm:

- Job IDM có thể chỉ mất vài giây.
- `torch_compile` có thể nhanh hơn eager một chút trên warm job.

Nếu cold:

- Lần đầu sau restart có thể lâu vì load model và compile graph.
- Nếu worker bị unload khi switch sang Klein, quay lại IDM có thể cần warmup lại.

Tóm tắt:

```text
Cold compile: lần đầu cho graph/shape/process mới
Warm job: các lần sau khi worker còn sống
Switch model: có thể cold lại nếu phải unload worker
```

Với UI mới, cold load/compile nên xảy ra ở lúc chọn engine hoặc đổi engine. Khi model đã `ready`, nút Generate mới mở. Nếu vẫn thấy job có stage `loading_model`, nghĩa là engine đó phải đổi resident worker trong lúc chạy, thường gặp ở hybrid IDM + Klein do không đủ VRAM để giữ cả hai pipeline.

### Cache compile

Có thể cache một phần:

```text
TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor
```

Cho TensorRT:

```text
--tensorrt-engine-cache-dir data/temp/trt_engine_cache_idm_full_safe
```

Nhưng cache disk không thay thế hoàn toàn resident process. Cách giảm latency tốt nhất:

- Giữ worker sống.
- Warm sẵn shape hay dùng.
- Tránh switch qua lại liên tục giữa IDM và Klein nếu VRAM không đủ giữ cả hai.

### Klein và hybrid

Klein local nặng hơn IDM warm. Hybrid chạy:

```text
IDM runtime + Klein runtime + fuse/composite + poll overhead
```

Pure Klein (`klein_lora` hoặc `klein_bnb_4bit`) hưởng lợi rõ nhất từ resident worker vì pipeline được giữ lại giữa các lần gen. History dùng `request_config.engine_mode`, nên hai dòng `klein_lora` và `klein_bnb_4bit` không còn bị hiển thị chung thành `klein_tryon_lora`.

Vì vậy hybrid thường chất lượng cao hơn nhưng chậm hơn IDM only.

## 12. ComfyUI reproduce workflow

Thư mục workflow:

```text
virtual_tryon/comfyui_workflows/omnitry_innerwear_repro
```

Ví dụ case KGA:

```text
female_underwear_02_kga1151_pk001_p6_15_idm_klein_hybrid_api.json
female_underwear_02_kga1151_pk001_p6_15_idm_klein_hybrid_ui.workflow.json
```

Workflow này làm:

```text
LoadImage(person)
LoadImage(garment)
VTONPhase2BackendTryOnAPI(
  category="women_underwear",
  engine_mode="idm_klein_hybrid",
  seed=2026070203,
  output_width=512,
  output_height=768,
  steps=4,
  deterministic=true
)
SaveImage(result)
SaveImage(mask_preview)
```

ComfyUI output nằm ở:

```text
/workspace/ComfyUI/output/vton_omnitry_repro/{case_id}/{engine_mode}
```

Backend artifacts vẫn nằm ở:

```text
virtual_tryon/data/outputs/{job_id}
```

## 13. Khởi động trên RunPod

Không commit secrets vào repo. Env thật nằm ngoài repo.

Backend:

```bash
cd /workspace/Project_Phase2/virtual_tryon/backend
set -a
source /workspace/secrets/virtual_tryon.env
set +a
/workspace/venvs/project_phase2/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
export PATH=/workspace/node-v20.19.0-linux-x64/bin:$PATH
cd /workspace/Project_Phase2/virtual_tryon/frontend
npm run dev -- --host 0.0.0.0 --port 8080
```

ComfyUI:

```bash
cd /workspace/ComfyUI
./run_vton_phase2_comfyui.sh
```

Health check:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8188/system_stats
```

## 14. Lỗi thường gặp

### Backend is offline or unreachable

Nguyên nhân thường gặp:

- Backend port `8000` chưa chạy.
- Uvicorn crash vì import/model config.
- Frontend trỏ sai API base.
- Pod port mapping bị đổi.

Kiểm tra:

```bash
curl -sS http://127.0.0.1:8000/health
ps -eo pid,cmd | grep uvicorn
tail -n 100 /tmp/virtual_tryon_backend.log
```

### Klein Try-On LoRA unavailable: FAL_KEY is not set

Nếu dùng local Klein thì không cần `FAL_KEY`. Lỗi này nghĩa là config đang trỏ sang backend external/API hoặc Klein local chưa được enable đúng cách.

Cần kiểm tra:

- `TRYON_KLEIN_BACKEND`
- `TRYON_KLEIN_MODEL_PATH`
- `TRYON_KLEIN_LORA_PATH`
- `TRYON_KLEIN_PYTHON`

Không in giá trị secret ra terminal, log hoặc file repo.

### ComfyUI workflow không chạy

Kiểm tra:

```bash
curl http://127.0.0.1:8188/system_stats
ls -l /workspace/ComfyUI/custom_nodes/vton_phase2_nodes
tail -n 100 /tmp/comfyui_vton.log
```

Nếu ComfyUI sống nhưng node không thấy, custom node symlink hoặc dependency import có vấn đề.

### Mask nhìn rộng hơn output thật

Điều này có thể đúng với hybrid:

- `mask_preview.png` là vùng cho phép engine thay.
- `hybrid_idm_delta_mask.png` là vùng fuse cuối dựa trên pixel IDM thực sự đã thay.

Nếu output đúng, mask preview rộng hơn không nhất thiết là lỗi.

### Output Klein bị đen vùng da

Klein raw/detail có thể over-edit. Dùng hybrid để lấy Klein detail chỉ trong IDM delta mask.

Kiểm tra:

```text
hybrid_klein_detail.png
hybrid_idm_delta_mask.png
hybrid_result.png
```

### Job đầu sau switch model quá lâu

Thường là cold load/compile:

- Model mới được load vào VRAM.
- Graph mới được compile.
- Cache disk có thể giúp, nhưng resident process mới là cách nhanh nhất.

## 15. Checklist đánh giá output

1. Vùng thay có đúng category không?
2. Có còn sót đồ cũ trong mask không?
3. Face, hair, hand, skin, background có bị đổi không?
4. Garment có đúng màu, form, texture, đường viền không?
5. So sánh `hybrid_idm_base.png` với `hybrid_klein_detail.png`.
6. Xem `hybrid_idm_delta_mask.png` nếu final lấy thiếu hoặc lấy thừa chi tiết.
7. Xem `quality_report.json`, nhưng không phụ thuộc tuyệt đối vào heuristic.
8. Nếu cần reproduce, ghi lại category, engine, seed, resolution, steps, prompt và job_id.

## 16. Khuyến nghị hiện tại

Cho demo chất lượng:

- Innerwear: dùng `IDM + Klein hybrid`.
- Standard clothing: dùng IDM resident warm trước, rồi mới so với Klein/Flux.
- Khi cần debug mask: bật Debug và xem artifacts.
- Khi cần tốc độ: tắt Debug, dùng resolution thấp trước, giữ worker resident.
- Khi cần reproduce: dùng ComfyUI workflow đã sinh sẵn với seed cố định.

Cho tối ưu tiếp:

- Warmup sẵn các resolution preset hay dùng sau backend start.
- Lưu persistent `TORCHINDUCTOR_CACHE_DIR`.
- Chỉ unload IDM khi VRAM thật sự không đủ cho Klein/Flux.
- Benchmark hybrid theo từng category: women underwear, men underwear, bra, top, bottom.
- Nếu delta mask lấy thiếu detail, thêm knob dilation riêng cho `hybrid_idm_delta_mask`.
