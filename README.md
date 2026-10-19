# gpt-from-scratch

Building a small GPT (a character-level language model) from scratch in PyTorch,
following Andrej Karpathy's video
["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY).

This is a learning project. I type all the code myself while watching, predict
outputs before running cells, and write down what confused me in
[`KARPATHY_LOG.md`](KARPATHY_LOG.md). I don't copy from the reference repo.

The model trains on Tiny Shakespeare (~1.1M characters) and learns to generate
text that looks like Shakespeare.

## Progress

- [x] 1. Data, tokenizer, batches
- [ ] 2. Bigram model trains *(in progress: forward pass done, logits shape `(4, 8, 65)`)*
- [ ] 3. Single-head self-attention
- [ ] 4. Full Transformer block (multi-head attention, feed-forward, residuals, LayerNorm)
- [ ] 5. Scaled-up model trained

## Files

| File | What it is |
|---|---|
| `gpt_dev.ipynb` | Main notebook, where the code gets written step by step |
| `data/input.txt` | Tiny Shakespeare dataset |
| `KARPATHY_LOG.md` | Notes for each chapter: key ideas, tensor shapes, confusions and how I resolved them |
| `PROTOCOL.md` | How I work through each chapter (preview, watch, predict, check, teach-back, log) |

## Running it

Requires Python 3 and PyTorch.

```bash
pip install torch jupyter
jupyter notebook gpt_dev.ipynb
```

Training will run on CUDA (NVIDIA GPU), MPS (Apple Silicon), or CPU, whichever is available (coming in chapter 2).

## Credits

- Andrej Karpathy, for the video and the original
  [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) code
- Tiny Shakespeare dataset from Karpathy's
  [char-rnn](https://github.com/karpathy/char-rnn)
