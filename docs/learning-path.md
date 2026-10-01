# Learning path: rebuild the project in your head

This guide is pinned to the source layout as of project state version 4. Source
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

- `configs/tiny.json:1-13` - the experiment values a user can change.
- `src/numpy_gpt/config.py:11-25` - the same values represented in Python.
- `src/numpy_gpt/config.py:27-46` - invalid configurations rejected early.
- `src/numpy_gpt/config.py:48-50` - head width calculation.
- `src/numpy_gpt/config.py:52-75` - exact parameter-count decomposition.
- `src/numpy_gpt/config.py:77-86` - conversion to dictionaries and loading JSON.
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

### Step 7 - Understand the public package surface

Read `src/numpy_gpt/__init__.py:3-29` last. It contains little mathematics; it
selects which names users can import directly from `numpy_gpt`. Reading it earlier
would show names without explaining their behavior.

### Step 8 - Run the current learning checkpoint

From the project root, run:

```powershell
python -m unittest discover -s tests -v
```

The expected checkpoint is 16 tests followed by `OK`. If a test fails, read the
test first, state what behavior it expected, and only then inspect the associated
implementation.

## Track B: corpus and project operations

Study this track after Track A. It does not explain transformer mathematics.

### Step 9 - Understand why files are not automatically training data

Read:

- `docs/data-policy.md:3-9` - source folders and non-copying rule.
- `docs/data-policy.md:11-25` - seven admission gates.
- `docs/data-policy.md:27-32` - default exclusions.
- `docs/data-policy.md:34-39` - domain and diversity limitation.
- `docs/corpus-audit.md:6-19` - counts.
- `docs/corpus-audit.md:21-35` - duplicates, sensitive data, and OCR holds.
- `docs/corpus-audit.md:37-47` - scale estimate and limitations.
- `docs/corpus-audit.md:49-58` - future preprocessing sequence.

### Step 10 - Read the corpus auditor as an independent utility

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

### Step 11 - Read future identity work separately

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
```

After the 16-test checkpoint passes, the next chain will be:

```text
RMSNorm and LayerNorm
    -> SwiGLU
    -> RoPE
    -> causal scaled-dot-product attention
    -> multi-head attention
    -> one transformer block
```

