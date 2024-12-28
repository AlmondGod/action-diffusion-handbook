# 1. Diffusion Conditional VFs

From [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)

Diffusion is a subset of flow matching. How do we get the Denoising Diffusion process from flow matching?

Diffusion Variants;

1. Variance Exploding: noise added by scaling up variance (we don’t use this anymore), but for this we would have 
    
$$
\mu_t(x_1) = x_1 \text{ and } \sigma_t(x_1) = \sigma_{1-t}
$$
    
2. Variance Preserving: noise added and preserving total variance (modern diffusion with alpha scaling)

For Variance Preserving Diffusion, we choose 

$$
\sigma_t(x_1) = \sqrt{1 - \alpha_{1-t}^2}\\
\mu_t(x_1) = \alpha_{1-t}x_1 \\
\text{ where } \alpha_t = e^{-\frac{1}{2}T(t)}, \quad T(t) = \int_0^t \beta(s)ds
$$

We are combining the Diffusion Condition Vector Field with Flow Matching objective, which they claim is better than score matching objective. They argue Diffusion technically never approaches true datapoints $x_1$ but just approximates/approaches them and can’t reach them in finite time, whereas in Flow Matching we exactly define the ends of our flow paths to be our $x_1$s

Much better theoretical guarantees essentially!

[[Prev]](../3.3:%20Conditional%20Flow%20Matching/Conditional%20Probability%20Paths%20and%20Vector%20Fields.md) [[Next]](./Optimal%20Transport%20Flow.md)