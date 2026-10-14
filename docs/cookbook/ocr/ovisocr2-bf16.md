# OvisOCR2 BF16 on vLLM-MUSA

## Overview

OvisOCR2 uses the in-process vLLM `LLM` API on one S5000. The model is loaded once, and a list of image requests is passed to `LLM.generate`. This recipe covers the vLLM engine and request configuration.

## At a glance

| Setting | Value | Where it is set |
|---|---|---|
| Precision | BF16 resolved from checkpoint | Engine startup log; no `dtype` override |
| Tensor parallel size | `1` | `LLM(tensor_parallel_size=1)` |
| GPU memory utilization | `0.8` | `LLM(gpu_memory_utilization=0.8)` |
| GDN prefill backend | `triton` | `LLM(gdn_prefill_backend="triton")` |
| Prefix caching | disabled | `LLM(enable_prefix_caching=False)` |
| Sampling | `temperature=0.0`, `max_tokens=16384` | `SamplingParams` |
| Image processor bounds | `min_pixels=448*448`, `max_pixels=2880*2880` | Per-request `mm_processor_kwargs` |

The reference parser passes the first three engine settings directly. Its wrapper injects `enable_prefix_caching=False` into the `LLM` constructor, so the effective configuration is shown here as one constructor call. The model configuration resolves BF16 at runtime; the parser does not pass a `dtype` argument.

Set the vLLM worker method before starting the Python process. Select the GPU through the MUSA device mask used by the deployment environment.

```bash
export MTHREADS_VISIBLE_DEVICES="${MTHREADS_VISIBLE_DEVICES:?set the assigned S5000 GPU index}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OVIS_CLIENT_SCRIPT="${OVIS_CLIENT_SCRIPT:?set the OvisOCR2 client script path}"
python3 "$OVIS_CLIENT_SCRIPT"
```

The caller supplies `model_path`, the Ovis instruction text, and a list of open PIL images. This class loads the engine once and returns raw vLLM outputs. Image loading and output processing are outside this recipe.

```python
import torchada  # Activate MUSA mappings before importing vLLM.
from vllm import LLM, SamplingParams


class OvisVllm:
    def __init__(self, model_path: str):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            gdn_prefill_backend="triton",
            enable_prefix_caching=False,
        )
        self.sampling = SamplingParams(max_tokens=16384, temperature=0.0)

    def generate(self, images, instruction: str):
        prompt = self.llm.get_tokenizer().apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        requests = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
                "mm_processor_kwargs": {"images_kwargs": {
                    "min_pixels": 448 * 448,
                    "max_pixels": 2880 * 2880,
                }},
            }
            for image in images
        ]
        return self.llm.generate(requests, self.sampling)
```

The reference workload submits up to 32 image requests per `generate` call. This client batch size is separate from the engine scheduler limits. The parser and wrapper leave `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, and CUDA Graph settings at the installed vLLM-MUSA defaults. Read their resolved values from the engine startup log before comparing runs; this recipe does not prescribe values for them.

Return to the [OCR recipe index](README.md).
