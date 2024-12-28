# Inference

1. start from x^K sampled from Gaussian noise, perform K iterations of denoising to produce increasingly less noisy intermediate actions x^k…x^0, where x^0 is the desired noise-free output
2. denoising is according to the equation 
    
$$
\mathbf{x}^{k-1} = \alpha(\mathbf{x}^k - \gamma\varepsilon_\theta(\mathbf{x}^k, k) + \mathcal{N}(0, \sigma^2I))
$$
    
where \epsilon_\theta is the noise prediction network (”diffusion model”)

essentially the next diffusion timstep is the scaled(previous - weighted network noise prediction + small amount of Gaussian noise (for nondeterministic generation))

we scale by alpha (usually < 1) in denoising (and in noising) so our values don't grow without bound as we progress in timesteps

same as noisy gradient descent step (noise pred network is predicting gradient field, gamma is learning rate) (CRAZY/COOL!)

$$
\mathbf{x}' = \mathbf{x} - \gamma\nabla E(\mathbf{x})
$$

**Noise schedule**: choice of /alpha, /gamma, and /sigma as functions of iter step k is same as learning rate scheduling!

**This is sick:** its like running gradient descent on an already learned gradient, and the point of diffusion training is for the noise prediction network to learn this gradient!

# Training

1. randomly draw unmodified dataset sample x^0
2. select denoising iteration k and sample random noise \epsilon^k with appropriate variance for iteration k (according to WHAT? also why do start at some iteration k instead of going in order)
3. Noise prediction network \epsilon_\theta predicts noise added at iter k, and loss is 
    
$$
\mathcal{L} = \text{MSE}(\varepsilon^k, \varepsilon_\theta(\mathbf{x}^0 + \varepsilon^k, k))
$$

Same as minimizing KL-divergence between data dist p(x^0) and DDPM sample dist q(x^0) from inference or maximizing variational lower bound/log-likelihood of data 

- **variational lower bound/evidence-based lower bound(VLB/ELBO):** lower-bound on log likelihood of data (summed log probabilities of each datapoint)
    - reconstruction term (how well original data reconstructed) and KL div term (how close posterior is to true posterior)
    - true log likelihood is what we want to optimize, but too hard so VLB gives us easier lb to optimize (and if lower bound improves we know log likelihood is improving)
    - ensures denoising matches noising
- **KL-divergence**: how different two data distributions are