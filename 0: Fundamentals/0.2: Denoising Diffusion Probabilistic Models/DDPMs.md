# Denoising Diffusion Probabilistic Models

From [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)

DDPMs are an implementation of the idea of variational inference. We sample from $p(x | z)$ by starting from noise and denoising until we have a sample from the model distribution.

# Inference

1. start from $x^K$ sampled from Gaussian noise, perform $K$ iterations of denoising to produce increasingly less noisy intermediate actions $x^k…x^0$, where $x^0$ is the desired noise-free output
2. denoising is according to the equation 
    
$$
\mathbf{x}^{k-1} = \alpha(\mathbf{x}^k - \gamma\varepsilon_\theta(\mathbf{x}^k, k) + \mathcal{N}(0, \sigma^2I))
$$
    
where $\epsilon_\theta$ is the noise prediction network (”diffusion model”)

essentially the next diffusion timstep is the scaled(previous - weighted network noise prediction + small amount of Gaussian noise (for nondeterministic generation))

we scale by $\alpha$ (usually < 1) in denoising (and in noising) so our values don't grow without bound as we progress in timesteps

> This is the same as noisy gradient descent step, where the noise prediction network is predicting the gradient field, and $\gamma$ is the learning rate!

$$
\mathbf{x}' = \mathbf{x} - \gamma\nabla E(\mathbf{x})
$$

**Noise schedule**: choice of $\alpha$, $\gamma$, and $\sigma$ as functions of iter step $k$ is same as learning rate scheduling!

**This is sick:** its like running gradient descent on an already learned gradient, and the point of diffusion training is for the noise prediction network to learn this gradient!

# Training

1. randomly draw unmodified dataset sample $x^0$
2. select denoising iteration k and sample random noise \epsilon^k with appropriate variance for iteration k (according to WHAT? also why do start at some iteration k instead of going in order)
3. Noise prediction network \epsilon_\theta predicts noise added at iter k, and loss is 
    
$$
\mathcal{L} = \text{MSE}(\varepsilon^k, \varepsilon_\theta(\mathbf{x}^0 + \varepsilon^k, k))
$$

Same as minimizing KL-divergence between data dist $p(x^0)$ and DDPM sample dist $q(x^0)$ from inference or maximizing variational lower bound/log-likelihood of data 

- **variational lower bound/evidence-based lower bound(VLB/ELBO):** lower-bound on log likelihood of data (summed log probabilities of each datapoint)
    - reconstruction term (how well original data reconstructed) and KL div term (how close posterior is to true posterior)
    - true log likelihood is what we want to optimize, but too hard so VLB gives us easier lb to optimize (and if lower bound improves we know log likelihood is improving)
    - ensures denoising matches noising
- **KL-divergence**: how different two data distributions are

Here's an accompanyng Colab Notebook to familiarize oneself with DDPMs on a simple 2D example: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DC9PVdfgKzvGv73yoAOw4RRFgf35Qr1-#scrollTo=3Ncc92M6W3Aa&uniqifier=1)

With this base of DDPMs, we can see how Action Diffusion works next.

[[Prev]](../0.1:%20Variational%20Generative%20Inference/Variational%20Generative%20Inference.md) [[Next]](../../1:%20Diffusion%20Policy/1.1:%20Action%20Diffusion/Action%20Diffusion.md)