# Karpathy "Let's build GPT": my log

One entry per chapter: key idea, formulas/shapes, what confused me and how it got resolved.

## 1. Reading the data + tokenization (up to ~13:31), in progress

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

**Next:** 90/10 train/val split. Open questions: tokens in each split? why the first 90% rather than random?
