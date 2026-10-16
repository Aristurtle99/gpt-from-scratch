# Protocol: Karpathy "Let's build GPT" with Claude

Video: "Let's build GPT: from scratch, in code, spelled out" (Andrej Karpathy), https://www.youtube.com/watch?v=kCc8FmEb1nY
Reference code (open only AFTER finishing a chapter): github.com/karpathy/ng-video-lecture
Data: `data/input.txt` (Tiny Shakespeare, ~1.1M characters)

Claude can't see the video. Wildan brings what Karpathy said, wrote, or ran.

## The loop, once per YouTube chapter

1. **Preview**: Wildan sends `preview: <chapter>`.
   Claude gives 3-4 lines on what the chapter is about and why it matters, plus 1-2 questions to keep in mind. No code spoilers.
2. **Watch**: Wildan pauses whenever something is unclear.
3. **Ask**: `confused at mm:ss: he said "..." / wrote ...`
   Claude explains it in a different way from the video, using math and shapes Wildan already knows (NNw2, PyTorch basics).
4. **Predict**: before Karpathy runs a cell, Wildan writes down the expected output or shapes, then runs their own code.
5. **Check**: `check: <chapter>` + code.
   Claude reviews it, points to bugs without silently fixing them, and asks 2 questions.
6. **Teach-back**: Wildan explains the chapter in 3-5 sentences, in their own words.
   Claude corrects anything wrong or vague. Move on only when it's right.
7. **Log**: Claude appends an entry to `KARPATHY_LOG.md` (chapter, key idea, formulas/shapes, confusions and how they were resolved).

## Rules for Claude
- Never give code ahead of where Wildan is in the video.
- CLAUDE.md teaching rules apply: hint levels, no silent fixes, predict first.
- If the teach-back has a gap, say so plainly.
- Wildan types all the code in `dl_specialization/gpt/`. No copying from the reference repo.

## Checkpoints (also in PROGRESS.md)
1. Data + tokenizer + batches
2. Bigram model trains
3. Single-head self-attention
4. Full block (multi-head + FFN + residual + LayerNorm)
5. Scaled-up model trained (RTX 5070 laptop if the Mac is too slow)

## Hardware
Device line at the top of the script, so the same code runs on the Mac or the RTX 5070:
`device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')`
