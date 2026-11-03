# Chapter 04 Decoder Structure and Attention

Assemble attention, FFN, and DecoderLayer into one working transformer block.

## What you will do

- Implement multi-head attention
- Add LayerNorm and FFN
- Validate the first layer with real weights

## What you will have at the end

- a validated DecoderLayer
- A clear bridge to the next chapter
- A visible checkpoint for the English route

## Status

This is the English landing page for the chapter. The detailed Chinese pages remain the source of truth for now.

## Next step

Continue to [summary](summary) or move to [the next chapter](/en/chapter05_quantization/intro).

## Related Reading

- Prerequisites: read the Chapter 03 MatMul pages first, because this chapter reuses the operator boundary and the Linear / MatMul setup.
- Later use: `chapter05_quantization` first replaces the weight representation used by this `DecoderLayer`; `chapter06_generation` then embeds it into the generation runtime.
- Roadmap: if you want the full 03–06 path first, start from [the Chapter 03 intro](/en/chapter03_matmul/intro). It summarizes the reading order from operators to quantization.
