![DiT](../../Images/Screenshot%202024-12-29%20at%202.32.11 PM.png)
# Diffusion Transformer (DiT)

From [Scalable Diffusion Models with Transformers](https://arxiv.org/pdf/2212.09748)

Diffusion Transformer use the best practices of Vision Transformer and uses them to implement the Diffusion Model.

**Classifier-free Guidance**: we take in some class label/conditioningas input as well, so we encourage denoising such that $\log p(c|x)$ is high which means we want to guide it to have high $p(x|c)$

- $p(c|x)$ is image belongs to desired property, so want to maximize this
- compares conditional diffusion that knows class with unconditional process that doesn’t know, and difference tells us how to modify to better match target
- also want high $p(x|c)$ since means $x$ is natural under $c$
- don't use classifier, rather use gradient of log $p(c|x)$ by comparing score funs of conditioned and unconditioned models, and guidance scale $w$ controls how much conditional vs unconditional path is emphasized
- estimated noise is weighted sum of conditioned and unconditioned predicted noise (weight is $s$)
    
$$
\nabla_x \log p(x|c) \propto \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))
$$
    
- also randomly drop out c during training and replace with learned null embeddings

**Latent Diffusion Model**

- first learn autoencoder encoding of x into latent space of zs, then learn Diffusion Model in space zs, then generate with diffusion model and learned decoder
- Theory is better Diffusion Model since vastly lower latent dim than data dim

## Architecture

Like Vision Transformer (ViT), operates on sequence of patches

1. **Patchify:** converts spatial input into sequence of T tokens of dim d with linear embedding
2. Apply standard ViT sine-cosine frequency-based positional embeddings  to all input tokens
3. Tokens processed by transformer block sequence
4. Conditioning can optionally be added
    1. **In-context conditioning:** append vector embeddings of conditioning as two additional tokens in input sequence (after final block, remove conditioning tokens from sequence) (very little overhead)
    2. **Cross-attention block:** have concatenated conditioning embeddings and have input tokens cross-attent to the conditioning tokens (roughly 15% overhead)
    3. **Adaptive LayerNorm (adaLN) Block:** scale and shift values determined by functions of conditioning (regressed from sum of conditioning embedding vectors), then applied to layer output (least overhead, same function to all tokens)
    4. **adaLN-Zero Block:** initializing residual block as identity function (zero initializing scale factor in block) (significantly outperforms adaLN)
5. Transformer Decoder (image tokens to noise prediction diagonal covariance prediction)
    1. standard linear decoder (apply final layer norm, linearly decode each token into 2 x same dim as spatial input)
    2. We predict covariance because different patches have different uncertainty/information levels

Use standard VAE model from stable diffusion as encoder/decoder and train DiT in latent space

## Observations

1. larger DiT models ar more compute efficient
2. Scaling up Diffusion sampling compute generally cannot compensate for lack of model compute

2. actions with noise $A_t^k$ passed in as input tokens for transformer decoder, with sinusoidal embeddings for diffusion iteration $k$ prepended as first token
3. observation transformed into obs embedding sequence by shared mLP, then passed to transformer decoder as input features
4. gradient predicted noise $\epsilon_\theta$ predicted by each output token of decoder stack


For a real-time visualization of DiT-Policy denoising actions to generate trajectories in genesis simulator, check out this Colab Notebook: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YHs_KuSM5f3AuykjkCBfaHNYTZKg08-a)

[[Prev]](../../1:%20Diffusion%20Policy/1.2:%20Components%20of%20Diffusion%20Policy/DP%20Components.md) [[Next]](../2.2:%20Components%20of%20Diffusion%20Transformers/DiT%20Components.md)