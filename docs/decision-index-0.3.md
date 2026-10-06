# Decision Index 0.3: the code and settings behind our runs

This page is for the Decision Index maintainers, who rerun submitted models on private
tests. It says which commit of this repository served each run, and gives the exact commands
to start the same `/v1/systemone` server.

| run | board | tag | serving source archive (SHA-256) | prompt texts |
|---|---|---|---|---|
| `di02-sharded-pmscr2` | main | `di03-main-pmscr2` | `6a1bd3ec68de1926a2704497c7fd91d7145c2a1df4dcb26d00c0898ba4e52d8d` | `configs/decision-index/pmscr.json` |

At the tag, `serve_h100.py`, every file under `src/reflex/`, and
`configs/decision-index/pmscr2-build-info.json` (shipped as `reflex/build-info.json`) are
byte-identical to the archive the GPU jobs ran. You can check this yourself:

```bash
git checkout di03-main-pmscr2
python scripts/decision_index/build_source_archive.py configs/decision-index/pmscr2-build-info.json
```

It prints every member's SHA-256 and the archive's. On the machine that built the run, the
rebuilt archive hash is exactly `6a1bd3ec…`. With a different zlib, the archive hash can
differ even when every member matches.

## What is pinned

- **Engine image:** `lmsysorg/sglang@sha256:b27fce60bc5494c118c4910702812bcfa8cee67abcdd1ff8b0902f21647552f4`.
  Use its Python (`/opt/sglang/bin/python`). It already has every package Reflex needs.
  Reflex is not pip-installed: it runs from `src/` on `PYTHONPATH`.
- **Model:** `Qwen/Qwen3.8-27B-FP8` at revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- **Hardware:** one NVIDIA H100 per server.
- **Prompt texts:** `configs/decision-index/pmscr.json` (SHA-256
  `35cb8653bc0372482006b02724ca0d5629711f15e8c1592fd14f560ecadfb35a`). The file sets three
  switches; every other prompt text keeps its default:
  - `question_preview`: lists the request's questions before the evidence when there is more
    than one question;
  - `preview_single_choice: "1"`: also lists the question when the request has a single
    choice question;
  - `choice_system_prompt`: a short reading-rules system prompt, used only when every question
    is a choice or score question.

  All three were chosen on a development suite that contains no Decision Index data.
- **Option orders:** `--permutations 2`. Each question is read in two option orders and the
  results are averaged. The harness also sends `"permutations": 2` with each request.
- **Context:** SGLang runs with `--context-length 65536`, and the Reflex front receives the
  same value. If the question preview would push a request past that window, the request is
  served without the preview. Nothing is ever truncated. At this tag, a request that is still
  too long goes to SGLang anyway. SGLang rejects it, and the front returns HTTP 502, which the
  harness records as an error.
- **Admission:** `REFLEX_EXPERIMENT_ROUTING=pressure` and `REFLEX_EXPERIMENT_MAX_REQUESTS=6`.
  These settings change only how a request's work is scheduled on SGLang. They never change
  what is asked, the probabilities, or the answers (see the commit "sglang: selective request
  admission").

## Start the server (as the run did)

Run everything inside the pinned image, from the root of this repository at the tag.

```bash
export PY=/opt/sglang/bin/python
export SNAP=$($PY -c 'from huggingface_hub import snapshot_download; print(snapshot_download(
    "Qwen/Qwen3.8-27B-FP8", revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]))')
export PYTHONPATH=$PWD/src
export REFLEX_EXPERIMENT_ROUTING=pressure
export REFLEX_EXPERIMENT_MAX_REQUESTS=6
export TOKENIZERS_PARALLELISM=false

# 1. SGLang (wait for http://127.0.0.1:30000/health to return 200)
$PY -m sglang.launch_server --model-path "$SNAP" --served-model-name Qwen/Qwen3.8-27B-FP8 \
  --host 127.0.0.1 --port 30000 --dtype bfloat16 --mem-fraction-static 0.85 \
  --context-length 65536 --max-running-requests 32 --max-total-tokens 65536 \
  --max-mamba-cache-size 320 --mamba-radix-cache-strategy extra_buffer \
  --cuda-graph-max-bs-decode 32 --chunked-prefill-size 8192 --attention-backend fa3 \
  --enable-metrics --mamba-ssm-dtype bfloat16 &

# 2. Reflex front (wait for http://127.0.0.1:8008/healthz to return 200)
$PY -m reflex.server --backend sglang --sglang-url http://127.0.0.1:30000 \
  --model "$SNAP" --permutations 2 --served-name reflex-27b \
  --host 127.0.0.1 --port 8008 --context-length 65536 \
  --prompt-texts configs/decision-index/pmscr.json
```

These are the commands that `deployment_commands()` in `serve_h100.py` builds. The run's job
entry took them from there, made two changes, and started both processes with their working
directory at the source root: it bound the front to `127.0.0.1` instead of `0.0.0.0`, and it
appended `--prompt-texts`. You can also run `python serve_h100.py`, which downloads the
snapshot and starts both processes the same way. It keeps the front on `0.0.0.0` and does not
pass prompt texts, so to reproduce this run, use the commands above.

The endpoint is `POST http://127.0.0.1:8008/v1/systemone`, with model name `reflex-27b`.

## Harness settings

We used the Decision Index kit's `reflex_http` engine with these engine options:
`base_url http://127.0.0.1:8008`, `model reflex-27b`, `extra {"permutations": 2}`,
`timeout 600`. Requests were sent by the frozen `run_parallel` with 32 workers.

One small adapter, `scripts/decision_index/reflex_http_overlay.py`, adds a single capacity
marker. At this tag, the server refuses a choice wider than its tokenizer's label alphabet with
HTTP 422 and the message "...this tokenizer supports at most N". The marker lets the kit score
that refusal as *unsupported* instead of as a retryable error.

The full run used four independent servers, one H100 each, with the rows split into four
shards.

## Latency

These figures come from one H100 serving requests one at a time. The sample was a 750-row mix
drawn the same way as the board's latency sample, run as two passes (1,500 requests), with
SGLang's prefix cache flushed before each pass. Times are client wall times:

| median | mean | p80 | p95 |
|---|---|---|---|
| 142 ms | 194 ms | 207 ms | 657 ms |
