![S1 Robot](../../Images/Screenshot%202024-12-29%20at%202.36.45 PM.png)

# Features of Diffusion Transformer

From [The Ingredients of Robotic Diffusion Transformers](https://arxiv.org/pdf/2410.10088)

**DiT-Block Policy:** novel architecture that outperforms SOTA on 1500+ timestep bimanual tasks

Solves typical instability of Diffusion Transformer

Improvements:

1. Scalable Attention Blocks: adds adaptive LayerNorm blocks to DiT policy layers
2. Efficient Observation Tokenization: use ResNet image tokenizer and Transformer policy

## Problem Setting

Goal-conditioned policy $\pi_\theta(a_t | o_t,g)$ predicts action dist $a_t \sim \pi(.|o_t,g)$ with env $T:S \times A → S$ with $o_t \in S$ and $a_t \in A$

$\pi$ optimized to match optimal action dist $D ={\tau_1..\tau_n}$ where each $\tau_i = \{g,o_0,a_0, o_1, ... \}$ collected from an expert (human teleoperator)

We have the same action-adjusted Diffusion setting as before.

## Architecture

1. observations (text goal, proprioception, and timestep k through sinusoidal Fourier + small MLP) turned into embedding vectors and combined with input noisy action with encoder-decoder Transformer architecture to produce predicted noise \epsilon_k
2. Transformer Encoder
    1. images processed separately by CNN backbone ResNet-26
    2. text incorporated into vision encoder with FiLM layers
    3. proprioceptive inputs regularized with per-dimension obs dropout before tokenization
    4. learned positional encoding added to input tokens
    5. all processed with Block Attention transformer encoder Octo
    6. results n transformer joint embedding tokens $e^1 ... e^L$ where $L$ is num layers
3. Transformer Decoder
    1. given current noised action input, timestep $k$, and encoder embeddings
    2. decoder block $I$ processes corresponding embedding from encoder $e^i$ 
    3. usually use cross attention, instead we use adaptive LayerNorm (adaLN) for extra stability, which shifts and scales layer outputs where shift/scale are functions of conditioning (in this case embedding from encoder block $e^i$)
    4. output scale projection layers initialized to 0 before residual layer (initializes noise network with skip connections)

Trained with AdamW optimizer and cosine learning schedule.

[[Prev]](../2.1:%20DiT/DiT.md) [[Next]](../2.3:%20RDT-1B/RDT-1B.md)