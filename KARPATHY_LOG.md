# Karpathy "Let's build GPT": my log

One entry per chapter: key idea, formulas/shapes, what confused me and how it got resolved.

## 1. Reading the data + tokenization (up to ~13:31), done

**Key idea:** a GPT predicts the next token. A neural net needs numbers, so text is
tokenized: each unique character gets an integer ID.

**Code/shapes:**
- `text`: one string, 1,115,394 characters
- `charss = sorted(list(set(text)))`: 65 unique characters, so vocab size = 65
- `string_to_integer` / `integer_to_string`: lookup dicts (char -> id, id -> char)
- `encode("to be or not to be")` -> `[58, 53, 1, 40, 43, 1, ...]`; `decode` reverses it
- `data = torch.tensor(encode(text), dtype=torch.long)` -> shape `[1115394]` (1D), `torch.int64`

**Trade-off:** character-level gives a tiny vocab (65) but long sequences; subword
tokenizers (tiktoken, ~50k vocab) give shorter sequences.

**Confusions resolved:**
- `with open(...) as f`: `with` closes the file automatically, even on error; `'r'` = read mode; `f.read()` returns one string.
- Dict/enumerate/comprehension/lambda: wrote the long `for`-loop versions first.
- Bugs fixed: used the whole string `s` as a dict key instead of the loop variable (KeyError); `append[...]` vs `append(...)`.
- Token IDs are integers, not vectors. Vectors come later from the embedding table.
- "torch is not accessed" is a Pylance unused-import hint, not a kernel problem (kernel = .venv).

- "Import torch could not be resolved" (yellow): Pylance was on a different Python than the kernel. Fix: Python: Select Interpreter -> `Machine_Learning/.venv/bin/python`.

## 2. Train/val split + data loader: batches of chunks (~13:31 to ~22:00), done (teach-back pending)

**Key idea:** we never feed the whole text at once. We take random chunks of `block_size`
characters, and one chunk of `block_size + 1` characters holds `block_size` training
examples (every context length from 1 to 8). Chunks are stacked into a batch so they
are processed in parallel.

**Code/shapes:**
- `n = int(0.9 * len(data))`; `train_data = data[:n]` -> `[1003854]`; `val_data = data[n:]`
- `train_data[:block_size+1]` -> `torch.Size([9])` (1D, not `[1, 9]`)
- `x = data[:block_size]`, `y = data[1:block_size+1]`: y is x shifted one step right, so `y[t]` = `data[t+1]`
- Examples: context `x[:t+1]` -> target `y[t]`, from `c0 -> c1` up to `c0...c7 -> c8`
- `get_batch(split)`:
  - `ix = torch.randint(len(data) - block_size, (batch_size,))`: 4 random start positions, shape `[4]`. Inputs: the upper limit (excluded) and the output shape, as a tuple.
  - `x = torch.stack([data[i:i+block_size] for i in ix])` -> `[4, 8]`; `y` is the same with `i+1`
  - `xb, yb`: shape `(B, T)` = `(4, 8)`, so 32 examples per batch
- `batch_size` = how many chunks (rows); `block_size` = how long each chunk is (columns)

**Confusions resolved:**
- Why the first 90%, not random: validation must be truly unseen text. Random characters would sit between training characters, so the model would be tested on what it memorised.
- `y[t]`, not `y[t+1]`: the +1 shift is already built into y. `y[t+1]` would skip a character and run off the end at t = 7.
- `x[:t+1]`: this +1 is not a shift. Slices stop before the end number, so `:t+1` includes position t.
- `ix` is a tensor. `(batch_size,)` is a tuple that only describes the shape you want; the comma makes it a tuple.
- `- block_size` in randint: keeps every chunk (and its y) inside the data, so all chunks are the same length and `torch.stack` works.
- `torch.stack` makes a list of `[8]` tensors into `[4, 8]` (new dimension). `torch.cat` would glue them into one flat `[32]`.
- f-string bug: `"{context}"` printed literally; needs `f"..."`.
- "Notebook controller is disposed": VS Code lost its kernel link after the interpreter change. Fix: Select Kernel -> .venv, then Run All (a new kernel has empty memory).
- Git: green U = untracked, A = staged, M = modified. Running cells alone changes `execution_count`, so the notebook shows M.

## 3. Bigram language model (~22:00 onwards), in progress

**Key idea:** the simplest language model predicts the next character from the current
character only. It is a 65 x 65 table of scores: row = current char, column = next char.

**Code/shapes:**
- `import torch.nn as nn` (layers, have learnable weights); `from torch.nn import functional as F` (plain maths, no weights)
- `class BigramLanguageModel(nn.Module)`: `__init__` builds the weights (`super().__init__()` first); `forward` runs the data through
- `self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)`: weight matrix `W`, shape `(65, 65)`, random at first
- `logits = self.token_embedding_table(idx)`: `(B, T)` -> `(B, T, C)` = `(4, 8, 65)`
- Call the model as `m(xb, yb)`, not `m.forward(...)`

**Embedding:**
- Token IDs are labels, not quantities (the sort order is arbitrary: L = 24 and M = 25 aren't similar). Embedding gives each ID its own learned vector.
- Lookup: each ID k is replaced by row `W[k]`. Same as `one_hot(k) @ W`: every row is multiplied by 0 except row k. It's a linear layer with no bias, done as a cheap lookup.
- B and T come from the input. C comes from the table's row length (a design choice; 65 here so each row is next-char scores).
- Only looked-up rows get gradients during training.
- Same character -> same 65 scores wherever it appears, so a bigram model has no context.

**Confusions resolved:**
- Channels (C) = how many numbers describe each position (word comes from RGB channels in images; Andrew Ng calls them features).
- `token_embedding_table` is a name Karpathy chose; the object is PyTorch's `nn.Embedding`, a callable layer.
- `forward(self, idx, targets)` receives 2 inputs, but the table only needs `idx`. `targets` is the answer key, used later for the loss. Python only errors on the wrong number of inputs, not unused ones.
- Indexing removes dimensions: `logits` `(4, 8, 65)` -> `logits[2]` `(8, 65)` -> `logits[2, 5]` `(65,)` -> `logits[2, 5, 10]` `[]`. Indexes start at 0.
- Logits are raw scores. Softmax `e^z_i / sum_j e^z_j` makes them positive and sum to 1.
- Generating has no real next character, so `targets` will become optional (`targets=None`) and the loss is skipped.

**Concept quiz (15 questions):** solid on shapes, softmax, val split, bigram limits.
Weak spots: stack vs cat (re-run the `a`/`b` experiment), off-by-one (`c0...c7 -> c8`, there is no c9), 0-based indexing.

**Predictions to check against the video:**
- Untrained loss should be about `-ln(1/65) = ln 65 ≈ 4.17`. Higher means confidently wrong.

**Next:**
- Type the `BigramLanguageModel` class into cell 14, run `m(xb, yb).shape`, send `check: bigram`.
- Rewrite cell 13's markdown in my own words (teach-back).
- Teach-back for chapter 2 (batches).
- Still open: Q5 blanks ("`y[t]` because y is already ___, so `y[t]` = `data[___]`").
