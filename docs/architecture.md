# Architecture decision record

## Objective

Build the strongest model that can be understood, tested, and trained under the
available compute. Similarity to frontier assistants is a direction, not an
evaluation metric. Each stage must have measurable loss, task quality, speed,
and memory baselines.

## Baseline: dense decoder-only transformer

The first implementation deliberately avoids mixture-of-experts, multi-head
latent attention, reinforcement learning, and distributed training. Those
features multiply failure modes before the basic gradients are trusted.

### Selected components

- Pre-normalized residual blocks for stable optimization.
- RMSNorm, with conventional LayerNorm retained as a comparison implementation.
- Rotary position embeddings (RoPE).
- Bias-free Q, K, V, output, and feed-forward projections.
- Standard causal multi-head attention for the first correct implementation.
- SwiGLU feed-forward network with hidden width rounded to a hardware-friendly
  multiple even though NumPy training starts on CPU.
- Tied input embeddings and language-model output weights.
- Stable log-sum-exp cross-entropy; no explicit probability tensor is needed for
  training loss.

### Deferred components

- Grouped-query attention: valuable for key/value-cache efficiency, but less
  useful before autoregressive inference becomes a bottleneck.
- Mixture-of-experts: adds routing, load balancing, communication, and sparse
  kernel complexity that is unjustified at this scale.
- Multi-head latent attention: promising for large key/value caches, but not the
  shortest route to a verified first model.
- Long-context modifications: a 256-token context keeps gradient debugging and
  attention memory manageable.

## Initial configuration

| Field | Value |
|---|---:|
| Vocabulary | 256 bytes |
| Context | 256 tokens |
| Layers | 4 |
| Model width | 256 |
| Heads | 4 |
| Head width | 64 |
| SwiGLU hidden width | 704 |
| Parameters | 3,279,104 |

The byte vocabulary avoids coupling the first model to a tokenizer-training
pipeline. A subword tokenizer becomes worthwhile for the GPU-scale model because
bytes consume context inefficiently.

## Scaling gates

The model is not enlarged until all conditions at the current scale pass:

1. Every primitive passes finite-difference gradient checks in float64.
2. The assembled model deliberately overfits a tiny sequence set.
3. Checkpoint restore reproduces the next update exactly.
4. Training and validation loss are reported separately.
5. Generation is compared with a bigram baseline under fixed seeds.
6. Memory and tokens-per-second are recorded.

## Primary research references

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [OLMo 2 Furious](https://arxiv.org/abs/2501.00656)

These references are design inputs, not instructions to copy every frontier
feature. OLMo is especially useful because its artifacts and training decisions
are unusually transparent; Llama 3 informs dense-model practice; DeepSeek-V3 is
an advanced branch for later efficiency experiments.

## Implementation status - 2026-10-02

Verified with deterministic and finite-difference tests:

- Stable log-sum-exp, softmax, and cross-entropy.
- Linear and Embedding forward/backward operations.
- RMSNorm and conventional LayerNorm forward/backward operations.
- Overflow-safe sigmoid, SiLU, and the elementwise SwiGLU core.
- RoPE forward and inverse-rotation backward operations, including offsets.
- Causal scaled-dot-product attention forward/backward operations.
- Full bias-free multi-head self-attention composition.
- Complete bias-free three-projection SwiGLU feed-forward composition.
- Separate gate/value input projections, SwiGLU activation, output projection,
  and summed gate/value input-gradient branches.

Not yet implemented:

- Residual transformer blocks, optimizer, training loop, and generation.
