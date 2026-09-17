# Summarize & Chat

**A two-stage NLP pipeline combining abstractive summarization with conversational language modeling.**

This project chains a fine-tuned summarization model with an instruction-tuned large language model to transform long-form text into a concise summary, then generate a natural-language conversational response grounded in that summary — all runnable on a free-tier Google Colab GPU.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Models Used](#models-used)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Example](#example)
- [Performance Notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Large volumes of text — articles, reports, transcripts — are often too long to reason about directly, whether by a human reader or a downstream language model with limited context. This project addresses that with a **two-stage pipeline**:

1. **Summarization stage** — An abstractive summarization model condenses arbitrary input text into a compact, information-dense summary.
2. **Conversational stage** — That summary is passed to an instruction-tuned chat model, which produces a natural-language response, discussion, or elaboration grounded in the summarized content.

This design mirrors a common pattern in production NLP systems: use a smaller, task-specific model to compress or pre-process content, then hand off to a larger general-purpose model for reasoning or dialogue — reducing token load and improving relevance of the final output.

The project is designed to run end-to-end in **Google Colab**, including on the free tier, using 4-bit quantization to fit an 8B-parameter chat model into a T4 GPU's memory budget.

---

## Architecture

```
                    ┌────────────────────┐
                    │   Raw Text Input    │
                    └──────────┬─────────┘
                               │
                               ▼
                ┌───────────────────────────┐
                │  BART (bart-large-cnn)     │
                │  Abstractive Summarization │
                │  • Beam search (n=4)       │
                │  • max_length = 150        │
                └──────────────┬────────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │   Text Summary      │
                    └──────────┬─────────┘
                               │
                               ▼
                ┌───────────────────────────┐
                │  Qwen3-8B (Instruct)       │
                │  Conversational Generation │
                │  • Chat template applied   │
                │  • 4-bit quantized (NF4)   │
                └──────────────┬────────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │  Generated Response  │
                    └────────────────────┘
```

**Design rationale:**
- BART handles the *compression* problem — extracting salient content from long text.
- Qwen3-8B handles the *reasoning/dialogue* problem — producing coherent, context-aware natural language output.
- Quantization (4-bit via `bitsandbytes`) makes the chat stage feasible on consumer-grade or free-tier GPU hardware without materially degrading output quality for conversational use cases.

---

## Models Used

| Stage | Model | Parameters | Purpose | Source |
|---|---|---|---|---|
| Summarization | `facebook/bart-large-cnn` | 406M | Abstractive summarization, fine-tuned on CNN/DailyMail | [Hugging Face](https://huggingface.co/facebook/bart-large-cnn) |
| Conversation | `Qwen/Qwen3-8B` | 8B (4-bit quantized) | Instruction-following chat generation | [Hugging Face](https://huggingface.co/Qwen/Qwen3-8B) |

---

## Requirements

- Python 3.10+
- CUDA-enabled GPU (T4, L4, A100, or equivalent) — a free Colab T4 instance is sufficient
- ~20 GB free disk space for model weights (first run only; cached afterward)
- CUDA-enabled PyTorch build (**not** the CPU-only wheel)

### Python Dependencies

```
transformers>=4.44
bitsandbytes>=0.46.1
torch>=2.4 (CUDA build)
accelerate
```

---

## Installation

```bash
pip install -U transformers bitsandbytes accelerate
```

If running in an environment where `torch` defaults to a CPU-only build (common on some Colab sessions), reinstall the CUDA build explicitly:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU availability before proceeding:

```python
import torch
print(torch.cuda.is_available())   # should print True
print(torch.__version__)           # should include a +cuXXX suffix, not +cpu
```

---

## Setup

```python
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch

# --- Summarization model ---
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# --- Conversational model (4-bit quantized) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model1 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
```

---

## Usage

Run the interactive loop below. Each iteration summarizes the input text, then generates a conversational response based on that summary.

```python
while True:
    text_input = input("Enter your text here (or 'quit' to stop): ")
    if text_input.lower() == "quit":
        break

    # --- Stage 1: Summarize ---
    inputs = bart_tokenizer([text_input], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = bart_model.generate(
        inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True
    )
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("\nSummary:", summary)

    # --- Stage 2: Generate conversational response ---
    messages = [{"role": "user", "content": summary}]
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = tokenized["input_ids"].to(model1.device)
    attention_mask = tokenized["attention_mask"].to(model1.device)

    output = model1.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
    )
    generated_tokens = output[0][input_ids.shape[-1]:]
    print("Response:", tokenizer.decode(generated_tokens, skip_special_tokens=True))
    print("-" * 60)
```

---

## Example

**Input:**
> A long article describing recent advances in renewable energy storage, covering battery chemistry improvements, grid-scale deployment, and cost trends over the past five years.

**Stage 1 Output (BART summary):**
> Recent advances in battery chemistry have improved energy density and lifespan, driving down costs for grid-scale renewable storage over the past five years.

**Stage 2 Output (Qwen3-8B response):**
> That's a significant shift — falling storage costs are often the bottleneck standing between intermittent renewable generation and reliable grid supply. The improvements in battery chemistry you mentioned are likely doing a lot of the heavy lifting here...

*(Actual outputs will vary by run and input.)*

---

## Performance Notes

Approximate timings on a **free-tier Colab T4 GPU** (times vary with network conditions and Colab load):

| Step | Approximate Time |
|---|---|
| Install dependencies | 10–20 sec |
| Download BART weights (~1.6 GB) | 20–40 sec |
| Download Qwen3-8B weights (~16 GB, bf16) | 2–4 min |
| Load Qwen3-8B in 4-bit onto GPU | 30–60 sec |
| Per-response inference | 5–15 sec |

Using 4-bit quantization reduces Qwen3-8B's GPU memory footprint from roughly 16 GB (fp16) to approximately 4–6 GB, making it compatible with a T4's 15 GB VRAM alongside BART and system overhead.

---

## Troubleshooting

**`ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes`**
Install or upgrade the package, then **restart the runtime** — bitsandbytes must be re-registered in a fresh Python process:
```bash
pip install -U bitsandbytes
```

**`RuntimeError: Expected all tensors to be on the same device`**
Move tokenized inputs to the model's device explicitly:
```python
input_ids = input_ids.to(model1.device)
```

**`TypeError: 'Tensor' object is not subscriptable`**
`apply_chat_template` returns a raw tensor by default. Pass `return_dict=True` to get a dictionary with both `input_ids` and `attention_mask`.

**Model download is very slow or appears stuck**
Confirm the runtime has an active GPU and CUDA-enabled torch:
```python
import torch
print(torch.cuda.is_available())
```
If `False`, go to **Runtime → Change runtime type → T4 GPU**, then restart and reinstall torch with a CUDA-enabled build.

**CPU-only torch installed despite selecting a GPU runtime**
Explicitly reinstall the CUDA build:
```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Limitations

- `bart-large-cnn` is fine-tuned on news-style text; summarization quality may degrade on technical, legal, or creative writing.
- 4-bit quantization introduces a small, generally acceptable quality trade-off in exchange for substantially lower memory usage.
- The current implementation has no persistent conversation memory — each loop iteration is stateless with respect to prior turns.
- Model weights are re-downloaded on every fresh Colab runtime unless explicitly cached (e.g., via Google Drive).

---

## Roadmap

- [ ] Multi-turn conversation memory across loop iterations
- [ ] Support for domain-specific summarizers (e.g., `pegasus-arxiv` for research papers, `pegasus-pubmed` for biomedical text)
- [ ] Lightweight web UI (Gradio or Streamlit) in place of console I/O
- [ ] Google Drive model caching to avoid repeated downloads across sessions
- [ ] Batch processing mode for summarizing multiple documents at once

---

## Project Structure

```
.
├── slm_mark1.ipynb   # Main Colab notebook
├── README.md                  # Project documentation
```

---

## Contributing

Contributions, issues, and feature requests are welcome. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Open a pull request describing the change and motivation

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers) for model access and tooling
- [Facebook AI (Meta)](https://huggingface.co/facebook) for `bart-large-cnn`
- [Qwen Team, Alibaba](https://huggingface.co/Qwen) for the Qwen3 model family
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for enabling efficient quantized inference
