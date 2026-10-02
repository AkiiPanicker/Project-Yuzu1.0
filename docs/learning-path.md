# Learning path: rebuild the project in your head

This guide is pinned to the source layout as of project state version 23. Source
line numbers will move as the implementation grows, so update this file whenever
the referenced code changes substantially.

## How to study each step

Use three passes instead of trying to memorize code:

1. **Intent:** explain in one sentence what the file or function must accomplish.
2. **Shapes:** write the shape and dtype of every input, output, weight, and gradient.
3. **Math:** derive the forward equation and backward equation on paper, then read
   the test that tries to disprove the implementation.

Do not treat a passing test as an explanation. First predict the expected output,
then compare your prediction with the assertion.

## Track A: model mathematics

### Step 1 - Understand the target before the implementation

Read:

- `README.md:1-7` - purpose of the repository.
- `README.md:8-16` - constraints that prevent hidden framework behavior.
- `README.md:18-36` - initial model and residual-block data flow.
- `README.md:38-47` - directory layout.
- `README.md:61-73` - staged roadmap.
- `docs/architecture.md:3-8` - objective and evaluation philosophy.
- `docs/architecture.md:10-38` - selected and deferred architecture components.
- `docs/architecture.md:40-55` - initial dimensions and parameter budget.
- `docs/architecture.md:57-66` - gates that must pass before scaling.

You should be able to answer:

- Why is the first model byte-level rather than subword-level?
- Why are RoPE, RMSNorm, and SwiGLU selected while MoE is deferred?
- Why must a tiny model overfit a tiny sample before full training begins?

### Step 2 - Read the model dimensions as a contract

Read:

- `configs/tiny.json:1-14` - the experiment values a user can change.
- `src/numpy_gpt/config.py:12-27` - the same values represented in Python.
- `src/numpy_gpt/config.py:29-56` - invalid configurations rejected early.
- `src/numpy_gpt/config.py:58-60` - head width calculation.
- `src/numpy_gpt/config.py:62-85` - exact parameter-count decomposition.
- `src/numpy_gpt/config.py:87-96` - conversion to dictionaries and loading JSON.
- `tests/test_config.py:16-28` - the contract expressed as tests.

Write these shapes on paper:

```text
token embedding table: (256, 256)
attention input/output: (batch, sequence, 256)
one attention head:     (batch, sequence, 64)
SwiGLU hidden state:    (batch, sequence, 704)
```

Derive the current parameter count:

```text
embedding table
+ 4 layers * (Q/K/V/output projections + three SwiGLU matrices + two norm scales)
+ final norm scale
= 3,279,104 parameters
```

### Step 3 - Learn stable probability calculations

Read:

- `src/numpy_gpt/numerics.py:15-23` - shared floating-point validation.
- `src/numpy_gpt/numerics.py:26-40` - stable log-sum-exp.
- `src/numpy_gpt/numerics.py:43-49` - stable softmax.
- `src/numpy_gpt/numerics.py:52-95` - cross-entropy forward and logits gradient.
- `tests/test_numerics.py:23-45` - stability and known-value tests.
- `tests/test_numerics.py:47-59` - cross-entropy gradient check.
- `tests/test_numerics.py:61-69` - invalid-input tests.

Key derivations:

```text
softmax(z_i) = exp(z_i - max(z)) / sum_j exp(z_j - max(z))

cross_entropy(z, y) = logsumexp(z) - z_y

d(loss)/d(z_i) = softmax(z_i) - 1[i = y]
```

Subtracting the maximum does not change softmax probabilities, but it prevents
large positive logits from overflowing `exp`.

### Step 4 - Understand how we verify hand-written derivatives

Read:

- `src/numpy_gpt/gradcheck.py:16-21` - gradient-check result record.
- `src/numpy_gpt/gradcheck.py:24-48` - centered finite differences.
- `src/numpy_gpt/gradcheck.py:51-88` - analytical/numerical comparison.
- `tests/test_numerics.py:47-59` - a complete use of the checker.

The centered finite-difference estimate for coordinate `i` is:

```text
df/dx_i ~= [f(x + epsilon*e_i) - f(x - epsilon*e_i)] / (2*epsilon)
```

The numerical gradient is slow and only suitable for tiny test tensors. Its job
is to audit the fast analytical backward pass, not to train the model.

### Step 5 - Learn the Linear layer

Read:

- `src/numpy_gpt/layers.py:16-24` - layer input validation.
- `src/numpy_gpt/layers.py:27-33` - values cached between forward and backward.
- `src/numpy_gpt/layers.py:36-63` - Linear forward pass.
- `src/numpy_gpt/layers.py:66-85` - input, weight, and optional bias gradients.
- `tests/test_layers.py:24-30` - a direct forward example.
- `tests/test_layers.py:32-63` - independent gradient checks for input, weight,
  and bias.

Shape and equations:

```text
X: (..., input_width)
W: (input_width, output_width)
b: (output_width,)
Y = XW + b

dX = dY W^T
dW = X^T dY                 # leading dimensions are flattened first
db = sum(dY over all rows)
```

Transformers reuse this one operation for Q, K, V, attention output, every
feed-forward projection, and the language-model output head.

### Step 6 - Learn the Embedding layer

Read:

- `src/numpy_gpt/layers.py:88-94` - embedding backward cache.
- `src/numpy_gpt/layers.py:97-120` - integer token lookup.
- `src/numpy_gpt/layers.py:123-142` - table-gradient accumulation.
- `tests/test_layers.py:66-71` - direct row lookup.
- `tests/test_layers.py:73-84` - repeated token IDs accumulate into one row.
- `tests/test_layers.py:86-99` - finite-difference weight check.
- `tests/test_layers.py:101-104` - invalid token rejection.

An embedding is a learnable table, not a mysterious language operation:

```text
E:        (vocabulary_size, model_width)
token IDs:(batch, sequence)
output:   (batch, sequence, model_width)
```

Token IDs are discrete, so there is no gradient with respect to an ID. Gradients
flow only to the selected rows of `E`. Repeated tokens require addition rather
than assignment, which is why the backward pass uses `numpy.add.at`.

### Step 7 - Compare RMSNorm and LayerNorm

Read RMSNorm first:

- `src/numpy_gpt/normalization.py:15-41` - numeric, shape, and epsilon checks.
- `src/numpy_gpt/normalization.py:44-49` - shared scale-gradient reduction.
- `src/numpy_gpt/normalization.py:52-58` - RMSNorm backward cache.
- `src/numpy_gpt/normalization.py:61-79` - RMSNorm forward pass.
- `src/numpy_gpt/normalization.py:82-104` - RMSNorm backward pass.
- `tests/test_normalization.py:24-58` - definition and gradient tests.

For a vector `x` of width `d`:

```text
r = 1 / sqrt(mean(x^2) + epsilon)
n = x * r
y = n * gamma

g = dY * gamma
dX = r * [g - n * mean(g * n)]
dGamma = sum(dY * n over batch and sequence)
```

Then read LayerNorm:

- `src/numpy_gpt/normalization.py:107-114` - LayerNorm backward cache.
- `src/numpy_gpt/normalization.py:117-145` - LayerNorm forward pass.
- `src/numpy_gpt/normalization.py:148-179` - LayerNorm backward pass.
- `tests/test_normalization.py:61-130` - definition, gradients, constant inputs,
  and invalid settings.

```text
mu = mean(x)
centered = x - mu
s = 1 / sqrt(mean(centered^2) + epsilon)
n = centered * s
y = n * gamma + beta

g = dY * gamma
dX = s * [g - mean(g) - n * mean(g * n)]
dGamma = sum(dY * n over batch and sequence)
dBeta = sum(dY over batch and sequence)
```

RMSNorm avoids mean subtraction and the learned bias. That makes it slightly
simpler and matches the selected baseline. LayerNorm remains available for direct
comparison and for understanding the original Transformer normalization.

### Step 8 - Build the SwiGLU activation from SiLU and a gate

Read SiLU first:

- `src/numpy_gpt/activations.py:15-23` - floating-point validation.
- `src/numpy_gpt/activations.py:26-35` - overflow-safe sigmoid.
- `src/numpy_gpt/activations.py:38-43` - SiLU backward cache.
- `src/numpy_gpt/activations.py:46-52` - SiLU forward pass.
- `src/numpy_gpt/activations.py:55-69` - SiLU backward pass.
- `tests/test_activations.py:25-52` - definition, extreme inputs, and gradients.

```text
sigmoid(x) = 1 / (1 + exp(-x))
SiLU(x) = x * sigmoid(x)

dSiLU/dx = sigmoid(x) * [1 + x * (1 - sigmoid(x))]
```

The sigmoid implementation uses separate formulas for nonnegative and negative
inputs so `exp` never receives a dangerous large positive argument.

Then read the SwiGLU core:

- `src/numpy_gpt/activations.py:72-78` - values cached for backward.
- `src/numpy_gpt/activations.py:81-101` - gated forward pass.
- `src/numpy_gpt/activations.py:104-119` - gate and value gradients.
- `tests/test_activations.py:55-92` - definition, both gradients, and shape errors.

```text
a = SiLU(gate)
output = a * value

dGate = dOutput * value * dSiLU/dGate
dValue = dOutput * a
```

In the final feed-forward network, two Linear projections create `gate` and
`value`, SwiGLU combines them, and a third Linear projection returns from hidden
width 704 to model width 256. Keeping those pieces separate lets each derivative
be tested before composition.

### Step 9 - Understand RoPE as position-dependent rotation

Read:

- `src/numpy_gpt/position.py:15-23` - floating-point validation.
- `src/numpy_gpt/position.py:26-42` - frequencies, positions, cosine, and sine.
- `src/numpy_gpt/position.py:45-51` - cached rotation values.
- `src/numpy_gpt/position.py:54-93` - adjacent-pair forward rotations.
- `src/numpy_gpt/position.py:96-117` - inverse-rotation backward pass.
- `tests/test_position.py:18-82` - identity, explicit rotation, norm preservation,
  gradients, position offsets, and invalid settings.

For adjacent features `(x_even, x_odd)` at a given position and frequency:

```text
y_even = x_even * cos(theta) - x_odd * sin(theta)
y_odd  = x_even * sin(theta) + x_odd * cos(theta)

dX_even = dY_even * cos(theta) + dY_odd * sin(theta)
dX_odd  = -dY_even * sin(theta) + dY_odd * cos(theta)
```

This is an ordinary two-dimensional rotation, so it preserves each pair's squared
length. Position changes the angle, not the vector magnitude. Queries and keys
will receive the same positional rotations before attention compares them.

The implementation expects either:

```text
(batch, sequence, head_dimension)
```

or:

```text
(batch, heads, sequence, head_dimension)
```

The offset test proves that rotating positions `3..N` separately produces the
same values as slicing positions `3..N` from a full rotation. That property will
matter when generation uses a key/value cache.

### Step 10 - Build causal scaled-dot-product attention

Read:

- `src/numpy_gpt/attention.py:15-23` - finite floating-point validation.
- `src/numpy_gpt/attention.py:26-35` - values cached for backward.
- `src/numpy_gpt/attention.py:38-82` - scores, causal mask, softmax, and output.
- `src/numpy_gpt/attention.py:85-115` - gradients for value, probabilities,
  scores, query, and key.
- `tests/test_attention.py:22-114` - mask structure, prefix means, future-token
  isolation, all three gradients, one-token behavior, and invalid settings.

Forward equations for one head are:

```text
S = (Q K^T) / sqrt(head_dimension)
S[i, j] = -infinity when j > i
P = softmax(S, axis=keys)
O = P V
```

The triangular mask lets position `i` attend to positions `0..i`, including
itself, but never `i+1..N`. Masking happens before softmax, so prohibited entries
receive exactly zero probability.

Backward equations are applied in reverse order:

```text
dV = P^T dO
dP = dO V^T
dS = P * [dP - sum(dP * P, axis=keys)]
dQ = (dS K) / sqrt(head_dimension)
dK = (dS^T Q) / sqrt(head_dimension)
```

The softmax derivative subtracts the probability-weighted projection from each
row. Future positions remain zero through backward because their probabilities
and masked score gradients are zero.

At this stage Q, K, and V are already projected and split into heads. The next
module will create those projections, apply RoPE to Q and K, call this primitive,
merge the heads, and apply the output projection.

### Step 11 - Compose full multi-head attention

Read the composition data structures and shape helpers:

- `src/numpy_gpt/multi_head_attention.py:23-45` - weight-gradient and backward
  cache records.
- `src/numpy_gpt/multi_head_attention.py:48-57` - split model width into heads.
- `src/numpy_gpt/multi_head_attention.py:60-64` - merge heads back to model width.
- `src/numpy_gpt/multi_head_attention.py:67-91` - head and weight validation.

Then read the complete computation:

- `src/numpy_gpt/multi_head_attention.py:94-147` - forward composition.
- `src/numpy_gpt/multi_head_attention.py:150-198` - reverse-mode composition.
- `tests/test_multi_head_attention.py:24-205` - primitive equivalence, shapes,
  causality, all five gradient groups, zero values, and invalid configurations.

Forward flow:

```text
X -> Linear(Wq) -> split heads -> RoPE -------\
X -> Linear(Wk) -> split heads -> RoPE --------> causal attention
X -> Linear(Wv) -> split heads ----------------/
causal attention -> merge heads -> Linear(Wo) -> output
```

For model width 256 and four heads:

```text
before split: (batch, sequence, 256)
after split:  (batch, 4, sequence, 64)
after merge:  (batch, sequence, 256)
```

Backward follows the arrows in reverse. Because Q, K, and V all originated from
the same input `X`, their three input gradients must be added:

```text
dX = dX_from_Q + dX_from_K + dX_from_V
```

The large gradient test independently perturbs the input and all four weight
matrices. Passing only an output-shape test would not validate this composition.

### Step 12 - Compose the complete SwiGLU feed-forward network

Read:

- `src/numpy_gpt/feed_forward.py:18-34` - weight-gradient and cache records.
- `src/numpy_gpt/feed_forward.py:37-66` - the three compatible weight shapes.
- `src/numpy_gpt/feed_forward.py:69-88` - forward composition.
- `src/numpy_gpt/feed_forward.py:91-120` - backward composition.
- `tests/test_feed_forward.py:26-183` - primitive equivalence, shapes, zero-value
  behavior, all four gradient groups, branch addition, and invalid weights.

Forward flow for one token matrix `X`:

```text
gate  = X W_gate
value = X W_value
hidden = SiLU(gate) * value
output = hidden W_output
```

For the initial model:

```text
X:        (batch, sequence, 256)
W_gate:   (256, 704)
W_value:  (256, 704)
hidden:   (batch, sequence, 704)
W_output: (704, 256)
output:   (batch, sequence, 256)
```

Backward reverses the output projection, splits at the multiplication inside
SwiGLU, and returns through both input projections. Both branches originated from
`X`, so their gradients add:

```text
dX = dX_from_gate + dX_from_value
```

This module acts independently at each sequence position. Attention mixes
information between positions; the feed-forward module transforms the resulting
representation within each position.

### Step 13 - Assemble one pre-norm residual transformer block

Read:

- `src/numpy_gpt/transformer_block.py:33-50` - grouped parameter gradients and
  the four nested backward caches.
- `src/numpy_gpt/transformer_block.py:53-107` - both normalized sublayers and
  both forward residual additions.
- `src/numpy_gpt/transformer_block.py:110-146` - reverse composition and both
  residual-gradient additions.
- `tests/test_transformer_block.py:26-321` - primitive equivalence, arbitrary
  leading dimensions, causality, ten numerical gradient checks, residual
  identity, and invalid configurations.

Forward flow:

```text
A = RMSNorm(X, attention_norm_scale)
Z = X + MultiHeadAttention(A)
F = RMSNorm(Z, feed_forward_norm_scale)
Y = Z + FeedForward(F)
```

Pre-normalization means each sublayer receives normalized values while the
residual stream itself remains unnormalized. Each addition provides a direct
identity route through the block.

Backward must include those identity routes:

```text
dZ = dY + RMSNormBackward(FeedForwardBackward(dY))
dX = dZ + RMSNormBackward(AttentionBackward(dZ))
```

The first term in each sum is the direct residual gradient. Omitting either term
would still produce correctly shaped arrays, but training signals would be wrong.

### Step 14 - Assemble the complete four-block language model

Read:

- `src/numpy_gpt/model.py:33-73` - block parameters, complete model parameters,
  matching gradient records, and the full backward cache.
- `src/numpy_gpt/model.py:76-147` - configuration and parameter-shape contracts.
- `src/numpy_gpt/model.py:150-212` - embedding lookup, four-block forward stack,
  final RMSNorm, and tied output projection.
- `src/numpy_gpt/model.py:215-250` - reverse block traversal and tied-gradient
  addition.
- `tests/test_model.py:93-331` - primitive equivalence, four-block shapes and
  parameter count, causality, tied gradients, every model-parameter gradient,
  and invalid contracts.

Forward flow for the configured model:

```text
byte IDs
    -> token embedding table E
    -> transformer block 0
    -> transformer block 1
    -> transformer block 2
    -> transformer block 3
    -> final RMSNorm
    -> matrix multiply by E^T
    -> logits over 256 byte values
```

For token IDs shaped `(batch, sequence)`, the embedding and block stream is
`(batch, sequence, d_model)`. The final tied projection changes only the final
dimension, producing `(batch, sequence, vocab_size)` logits.

The embedding table is used twice: first to look up input vectors and later,
transposed, to project hidden vectors into vocabulary logits. Backward must add
both contributions into the one shared parameter:

```text
dE = dE_from_input_lookups + transpose(dW_from_output_projection)
```

The blocks run in order during forward and in reverse order during backward. The
finite-difference test checks the shared embedding, final norm, and all nine
arrays inside each of the four blocks.

### Step 15 - Initialize every parameter reproducibly

Read:

- `src/numpy_gpt/config.py:16-27` - the configured base initialization scale and
  random seed.
- `src/numpy_gpt/initialization.py:16-26` - the shared Gaussian-array helper.
- `src/numpy_gpt/initialization.py:29-118` - dtype/seed validation, one random
  stream, all parameter allocations, and residual-output scaling.
- `src/numpy_gpt/model.py:253-270` - unique parameter names in deterministic
  optimizer/checkpoint order.
- `tests/test_initialization.py:37-139` - reproducibility, seed changes, exact
  count, dtypes, finite values, distribution scales, and invalid settings.

The base initialization is:

```text
ordinary weights ~ Normal(0, initializer_std^2)
norm scales = 1
```

Each transformer block writes twice to the residual stream: once after attention
and once after its feed-forward network. Their output matrices therefore use:

```text
residual_std = initializer_std / sqrt(2 * n_layers)
```

For four layers and `initializer_std = 0.02`, this is approximately `0.007071`.
The smaller residual-writing scale limits variance accumulation at the beginning
of training without changing the verified forward or backward equations.

One seeded `numpy.random.Generator` produces every random array in a fixed order.
The named traversal exposes each unique parameter exactly once; the tied embedding
appears once even though forward uses it for both lookup and output projection.

### Step 16 - Update parameters with clipping and AdamW

Read:

- `src/numpy_gpt/model.py:253-290` - matching deterministic parameter and gradient
  traversal.
- `src/numpy_gpt/optimizer.py:16-42` - clipping results, optimizer state, and
  per-step terminal statistics.
- `src/numpy_gpt/optimizer.py:64-138` - named-array validation, overflow-resistant
  global norm, and copy-based clipping.
- `src/numpy_gpt/optimizer.py:141-190` - zero moment allocation and state
  validation.
- `src/numpy_gpt/optimizer.py:193-307` - one atomic, bias-corrected AdamW step.
- `tests/test_optimizer.py:29-265` - name/shape alignment, clipping, two update
  steps, decay exclusions, and failure-before-mutation contracts.

One norm covers every model gradient rather than clipping each tensor separately:

```text
G = sqrt(sum over every gradient element g_i^2)
c = min(1, max_norm / (G + clipping_epsilon))
g_clipped = c * g
```

The norm calculation first divides by the largest absolute gradient. This avoids
overflow in the intermediate squares without changing the mathematical norm.
Clipping returns copies, so the gradients produced by backward remain available
for debugging.

AdamW then updates its first and second moments at step `t`:

```text
m_t = beta1 * m_(t-1) + (1 - beta1) * g_clipped
v_t = beta2 * v_(t-1) + (1 - beta2) * g_clipped^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)

parameter = parameter
          - learning_rate * m_hat / (sqrt(v_hat) + adam_epsilon)
          - learning_rate * weight_decay * parameter
```

The final term is decoupled weight decay: it acts directly on the parameter
rather than being mixed into the adaptive gradient moments. Matrix parameters,
including the tied embedding table, receive decay. One-dimensional RMSNorm scales
do not. The entire update is calculated and checked for finite values before any
parameter array is modified, preventing half-applied invalid steps.

### Step 17 - Save and restore exact training state

Read:

- `src/numpy_gpt/checkpoint.py:23-34` - format version and returned checkpoint
  record.
- `src/numpy_gpt/checkpoint.py:37-133` - expected model names/shapes and strict
  parameter/moment validation.
- `src/numpy_gpt/checkpoint.py:136-206` - JSON-safe RNG-state encoding without
  Python object arrays.
- `src/numpy_gpt/checkpoint.py:210-256` - archive assembly, file synchronization,
  and atomic replacement.
- `src/numpy_gpt/checkpoint.py:259-328` - manifest validation and reconstruction
  of the nested model parameters.
- `src/numpy_gpt/checkpoint.py:331-422` - non-pickle loading and all archive
  contracts.
- `tests/test_checkpoint.py:72-251` - exact round trip, identical next update,
  non-object arrays, version/shape rejection, and overwrite protection.

Each checkpoint is one NumPy `.npz` archive containing:

```text
manifest bytes:
    format version
    complete ModelConfig
    deterministic parameter names
    optimizer step
    optional random-generator state

for every parameter i:
    parameter_i
    first_moment_i
    second_moment_i
```

The JSON manifest is stored as raw `uint8` bytes. Model and optimizer values are
ordinary numeric arrays, and loading always uses `allow_pickle=False`. This keeps
the file inspectable without allowing a checkpoint to deserialize arbitrary
Python objects.

Saving first validates all state, writes a temporary file in the destination
directory, flushes it to storage, and then atomically replaces the destination.
The most important test does more than compare saved arrays: it updates a model,
saves and reloads it, applies the same next gradient to both copies, and requires
the next parameters and Adam moments to match exactly.

### Step 18 - Turn text into deterministic next-byte batches

Read:

- `src/numpy_gpt/byte_data.py:16-31` - immutable split and batch records.
- `src/numpy_gpt/byte_data.py:34-48` - positive-size and byte-range contracts.
- `src/numpy_gpt/byte_data.py:51-72` - UTF-8 encoding and generated-byte decoding.
- `src/numpy_gpt/byte_data.py:75-105` - contiguous train/validation splitting.
- `src/numpy_gpt/byte_data.py:108-138` - seeded next-token window sampling.
- `tests/test_byte_data.py:24-157` - Unicode round trips, invalid bytes, split
  integrity, shifted targets, RNG restoration, and invalid contracts.

UTF-8 converts text into the model's fixed vocabulary of 256 byte values:

```text
text -> UTF-8 bytes -> integers in 0..255
```

An English character is usually one byte, while many other characters use
multiple bytes. The model predicts bytes, not Unicode characters. During
generation, replacement decoding keeps invalid partial byte sequences visible
instead of crashing the terminal.

The corpus is divided once at a fixed boundary:

```text
[---------------- train ----------------|------ validation ------]
```

Training windows sample only from the train array, and validation windows sample
only from the validation array. No context window crosses the boundary. Each
sampled window contains `context_length + 1` bytes; the last dimension is split
into inputs and one-position-shifted targets:

```text
window:  [b0, b1, b2, b3, b4]
inputs:  [b0, b1, b2, b3]
targets: [b1, b2, b3, b4]
```

Sampling uses an explicit `numpy.random.Generator`. Saving its state in the
checkpoint and restoring it must reproduce the exact next batch, which prevents
silent data-order changes after resuming training.

### Step 19 - Compose one complete update without hiding the mathematics

Read:

- `src/numpy_gpt/training.py:27-46` - immutable evaluation and training records.
- `src/numpy_gpt/training.py:49-83` - batch contracts and perplexity conversion.
- `src/numpy_gpt/training.py:86-101` - the read-only evaluation path.
- `src/numpy_gpt/training.py:104-148` - loss, backward, clipping, and AdamW update.
- `tests/test_training.py:31-48` - the tiny configuration and fixed byte batch.
- `tests/test_training.py:51-253` - direct composition, mutation, clipping, and
  invalid-input tests.

The training path is an explicit composition of already-tested pieces:

```text
ByteBatch
    -> language_model_forward
    -> mean cross_entropy_with_logits
    -> language_model_backward
    -> named gradients
    -> global-norm clipping inside AdamW
    -> parameter mutation + next optimizer state
```

The mean reduction matters. If a batch contains `N` target bytes, the logits
gradient is divided by `N`, so changing batch size does not silently multiply the
effective learning rate. The reported loss belongs to the parameters before the
update; the optimizer diagnostics describe the update just applied.

Evaluation deliberately stops after mean cross-entropy:

```text
ByteBatch -> forward -> mean loss -> metrics
```

It does not call backward and does not receive optimizer state. The test calls
evaluation twice and requires bit-identical parameters afterward. This is the
boundary that will later keep validation measurements from contaminating
training.

Perplexity is `exp(mean loss)`. It is interpretable as the model's effective
number of equally plausible next-token choices, but byte-level perplexity cannot
be compared directly with the subword-token perplexity of another model.

### Step 20 - Control repeated updates without losing reproducibility

Read:

- `src/numpy_gpt/trainer.py:52-139` - loop configuration, terminal metrics, and
  returned run artifacts.
- `src/numpy_gpt/trainer.py:143-226` - token validation, SHA-256 split identity,
  atomic run manifests, and JSON-lines helpers.
- `src/numpy_gpt/trainer.py:229-394` - initialization, validation, updates,
  metrics, checkpoint events, and return state.
- `configs/smoke.json:1-14` - the deliberately small CPU smoke model.
- `scripts/train_smoke.py:26-51` - CLI options and synthetic text construction.
- `scripts/train_smoke.py:54-115` - fresh/resumed setup and visible execution.
- `tests/test_trainer.py:30-54` - tiny integration-test fixtures.
- `tests/test_trainer.py:57-257` - validation, observability, checkpoint, and
  exact-resume tests.

`max_steps` is an absolute optimizer-step target. If a checkpoint is already at
step 20, resuming with `--steps 30` performs ten more updates. Treating 30 as
"thirty additional steps" would silently overshoot the intended run.

The two random streams have different jobs:

```text
training RNG -> new sampled batch each step -> saved in every checkpoint
fixed validation seed -> same validation batch -> comparable validation loss
```

Restoring the training RNG, parameters, Adam moments, and step must make the next
updates bit-identical to uninterrupted training. The integration test compares
all resulting parameters and first moments after an interrupted/resumed run.

Every logged step is written both to the terminal and `metrics.jsonl`. It reports
step, token-equivalent epoch, train loss, validation loss and perplexity, learning
rate, gradient norm, clip coefficient, tokens per second, and elapsed time.
Checkpoint events are separate JSON records. The epoch is an estimate based on
sampled target-token count; random windows can repeat, so it is not a claim that
every corpus byte was visited once.

`run_manifest.json` records the model configuration, optimizer hyperparameters,
token counts, and SHA-256 fingerprints of the train and validation arrays. A
resume using different data or learning settings is rejected before another
update, rather than silently mixing incompatible experiments.

The smoke script defaults to generated repetitive text. This is a mechanics and
overfitting check, not useful language-model training, and it does not ingest the
audited books or papers.

### Step 21 - Understand the public package surface

Read `src/numpy_gpt/__init__.py:3-195` last. It contains little mathematics; it
selects which names users can import directly from `numpy_gpt`. Reading it earlier
would show names without explaining their behavior.

### Step 22 - Run the current learning checkpoint

From the project root, run:

```powershell
python -m unittest discover -s tests -v
```

The 100-test checkpoint passed with final status `OK`. This verifies the trainer's
configuration contracts, required terminal fields, metrics/checkpoint files,
final-step behavior, exact resume, and incompatible-session rejection. It still
does not prove that a real run learns; that is the next checkpoint.

### Step 23 - Run and interpret the synthetic smoke training

From the project root, run:

```powershell
python scripts/train_smoke.py --steps 20
```

This is the first deliberate training execution. It uses the 7,280-parameter
model in `configs/smoke.json` and generated repetitive text, not the audited book
or paper folders. Each step should print fields in this shape:

```text
step=000001 epoch=... train_loss=... validation_loss=... validation_ppl=...
lr=... grad_norm=... clip=... tok/s=... elapsed=...s
```

The run is mechanically complete when the terminal prints `complete step=000020`
and the run directory contains:

```text
run_manifest.json
metrics.jsonl
checkpoint-step-00000010.npz
checkpoint-step-00000020.npz
```

Learning evidence requires more than completion. Compare the first and final
`validation_loss` values. The final value must be finite and lower than the first.
Training loss may fluctuate because each step samples random windows. Also inspect
`clip`: a value below `1.0` means the global gradient was clipped; clipping on
every step is a warning that the learning rate or gradient scale needs review.

Do not call this model conversational after the smoke run. The synthetic corpus
only tests whether repeated updates learn a tiny byte pattern. Sampling mechanics
are now verified later in this guide, and one-shot checkpoint generation is a
diagnostic interface rather than a dialogue capability.

Observed result on 2026-10-02:

- The run reached step 20 in 0.14 seconds and wrote both expected checkpoints,
  `metrics.jsonl`, and `run_manifest.json`.
- Training loss changed from 5.546263 to 4.147532.
- Validation loss changed from 5.544793 to 3.974303, a drop of 1.570490 or about
  28.3 percent.
- Validation perplexity changed from 255.902 to 53.213, about 4.81 times lower.
- All 20 raw gradient norms exceeded the configured limit of 1.0, so every update
  was clipped. The clipping became mild by step 20 (`clip=0.9486`), and the loss
  remained stable, so this is a diagnostic to monitor rather than evidence of a
  failed run.

Validation loss repeats on odd-numbered rows because validation runs every two
steps; the most recent value is carried into the intervening log record. The
small rise in training loss from step 19 to step 20 is also expected because the
training windows are sampled randomly. Neither behavior contradicts the clear
downward trend.

### Step 24 - Understand and verify autoregressive sampling

Read:

- `src/numpy_gpt/sampling.py:20-27` - immutable generation result and stop reason.
- `src/numpy_gpt/sampling.py:30-49` - temperature and top-k contracts.
- `src/numpy_gpt/sampling.py:52-90` - greedy and seeded categorical selection.
- `src/numpy_gpt/sampling.py:93-108` - prompt validation and copying.
- `src/numpy_gpt/sampling.py:111-162` - full-context autoregressive generation.
- `tests/test_sampling.py:24-33` - tiny generation configuration.
- `tests/test_sampling.py:36-163` - tie behavior, seeded reproducibility, top-k
  exclusion, manual composition, context stopping, and invalid inputs.

Autoregressive generation repeats one operation:

```text
visible token IDs
    -> complete model forward pass
    -> logits at the final visible position
    -> choose one next token
    -> append it to the visible token IDs
    -> repeat
```

Greedy mode chooses the first token with the largest logit. Stochastic mode
requires an explicit `numpy.random.Generator`, so identical seeds reproduce the
same sequence of categorical draws. Temperature rescales candidate logits before
normalization:

```text
probability_i proportional to exp((logit_i - max_logit) / temperature)
```

Lower temperatures concentrate probability on high-logit bytes. Top-k first
removes every token outside the `k` highest logits; excluded bytes therefore have
exactly zero probability. These controls change decoding, not trained weights.

The first reference implementation recomputes the entire visible context for
every generated token. This is inefficient but provides the clearest comparison
against `language_model_forward`. A key/value cache will be accepted only after
its token-by-token logits match this reference.

The smoke model has a context length of 16 bytes. Prompt bytes and generated bytes
share that limit. If a prompt uses 12 bytes and requests 10 new bytes, generation
produces four and returns `context_limit`; it never discards prompt history
silently.

Run the expanded checkpoint from the project root:

```powershell
python -m unittest discover -s tests -v
```

The 106-test checkpoint passed with final status `OK`. It verifies deterministic
ties, seeded draws, top-k exclusion, manual repeated-forward equivalence, prompt
copying, context stopping, and invalid-input rejection. It does not judge whether
the generated bytes form useful language.

### Step 25 - Understand one-shot checkpoint inference

Read:

- `src/numpy_gpt/inference.py:20-32` - immutable decoded result and timing fields.
- `src/numpy_gpt/inference.py:35-38` - explicit seed validation.
- `src/numpy_gpt/inference.py:41-93` - checkpoint load, UTF-8 conversion,
  generation, timing, decoding, and throughput.
- `src/numpy_gpt/inference.py:96-115` - terminal-safe result formatting.
- `scripts/generate_checkpoint.py:16-27` - command-line arguments and defaults.
- `scripts/generate_checkpoint.py:30-51` - visible one-shot execution.
- `tests/test_inference.py:32-58` - deliberately tiny checkpoint fixtures.
- `tests/test_inference.py:61-215` - direct-generation equivalence, seeded
  repetition, byte-counting, context limits, timing, escaping, and invalid inputs.

The wrapper does not create a second inference algorithm. It loads the exact
saved model and delegates token choice to the verified sampler:

```text
checkpoint path + prompt text
    -> load config, weights, optimizer step
    -> encode prompt as UTF-8 bytes
    -> generate_tokens(...)
    -> decode bytes with replacement for invalid UTF-8
    -> print escaped text plus timing and stop reason
```

Prompt length is measured in bytes, not displayed characters. For example, `é`
uses two UTF-8 token IDs. A generated byte sequence may also stop midway through
a multibyte character, so replacement decoding is required to keep the diagnostic
command robust.

The formatter uses Python string representations instead of printing generated
text directly. That makes newline, escape, and other control bytes visible as
escaped sequences rather than allowing arbitrary model output to control the
terminal. Generation timing excludes checkpoint loading and reports zero
throughput if the prompt already occupies the complete context.

Run the expanded checkpoint from the project root:

```powershell
python -m unittest discover -s tests -v
```

The expected checkpoint is 112 tests followed by `OK`. Only after it passes, run
the one-shot generator against the completed smoke checkpoint:

```powershell
python scripts\generate_checkpoint.py runs\smoke-20261002-140621\checkpoint-step-00000020.npz --prompt "Yuzu " --max-new-tokens 10 --greedy
```

The smoke configuration allows only 16 total bytes, including the prompt. Its
output will reflect one memorized repetitive sentence and should not be evaluated
as a chatbot response.

## Track B: corpus and project operations

Study this track after Track A. It does not explain transformer mathematics.

### Step 26 - Understand why files are not automatically training data

Read:

- `docs/data-policy.md:3-9` - source folders and non-copying rule.
- `docs/data-policy.md:11-25` - seven admission gates.
- `docs/data-policy.md:27-32` - default exclusions.
- `docs/data-policy.md:34-39` - domain and diversity limitation.
- `docs/corpus-audit.md:6-19` - counts.
- `docs/corpus-audit.md:21-35` - duplicates, sensitive data, and OCR holds.
- `docs/corpus-audit.md:37-47` - scale estimate and limitations.
- `docs/corpus-audit.md:49-58` - future preprocessing sequence.

### Step 27 - Read the corpus auditor as an independent utility

Read:

- `scripts/inventory_corpus.py:23-34` - paths and policy constants.
- `scripts/inventory_corpus.py:37-58` - hashing and safe subprocess execution.
- `scripts/inventory_corpus.py:60-93` - page counting and non-retained text samples.
- `scripts/inventory_corpus.py:95-107` - file discovery and sensitive-name check.
- `scripts/inventory_corpus.py:109-195` - record creation, status assignment,
  duplicate grouping, and summary construction.
- `scripts/inventory_corpus.py:197-213` - command-line arguments.
- `scripts/inventory_corpus.py:215-236` - main program and output writing.
- `data/manifests/corpus_inventory.json:1-27` - manifest metadata and summary only;
  do not begin by reading all 1,420 lines of individual records.

### Step 28 - Read future identity work separately

Read:

- `docs/identity-and-personality.md:3-16` - Yuvika/Yuzu naming behavior.
- `docs/identity-and-personality.md:18-27` - personality and consent boundary.
- `docs/identity-and-personality.md:29-39` - later implementation stages.
- `docs/identity-and-personality.md:41-51` - decisions intentionally deferred.

Identity routing and voice are application layers around the language model. They
should not be confused with the transformer operations being built now.

## Track C: repository mechanics

Read these only after the model and data tracks:

- `pyproject.toml:1-19` - package metadata, Python requirement, and NumPy dependency.
- `.gitignore:1-14` - generated data and model artifacts kept out of Git.
- `AGENTS.md:1-40` - rules for future repository work.
- `PROJECT_STATE.md` - read the whole file because it is intentionally updated
  after every project prompt; fixed line references would become stale.

## Current stopping point

The implemented learning chain currently ends here:

```text
configuration
    -> stable probabilities and loss
    -> numerical gradient auditor
    -> Linear forward/backward
    -> Embedding forward/backward
    -> RMSNorm and LayerNorm forward/backward
    -> SiLU and SwiGLU forward/backward
    -> RoPE forward/backward
    -> causal scaled-dot-product attention
    -> full multi-head attention composition
    -> complete SwiGLU feed-forward composition
    -> one pre-norm residual transformer block
    -> four-block byte language model with tied embeddings
    -> deterministic depth-scaled parameter initialization
    -> matching named parameter/gradient traversal
    -> global-norm clipping and AdamW
    -> exact versioned checkpoint save/restore
    -> UTF-8 byte splits and deterministic next-token batches
    -> one complete train/evaluation step
    -> observable, resumable multi-step trainer
    -> synthetic tiny-run loss-reduction proof
    -> deterministic autoregressive sampling
    -> terminal-safe checkpoint generation (awaiting verification)
```

The next evidence and implementation chain will be:

```text
interactive generation
    -> conversation-history management
```
