![AD](../../Images/Screenshot%202024-12-29%20at%203.10.32 PM.png)
# Action Diffusion

From [Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/pdf/2303.04137)

## Summary

Conditional denoising diffusion on robot action spaces

## Background

Denoising Diffusion Probabilistic Models: output generation is modeled as a denoising process (Stochastic Langevin Dynamics) 

1. Starts from $x^K$ sampled from Gaussian noise
2. K iterations of denoising to produce intermediate actions while decreasing noise $x^k .. x^0$ until desired noise-free output $x^0$ formed
3. follows equation 
    
$$
\mathbf{x}^{k-1} = \alpha(\mathbf{x}^k - \gamma\varepsilon_\theta(\mathbf{x}^k,k) + \mathcal{N}(0,\sigma^2I))
$$

where $\epsilon_\theta$ is noise prediction network and $N(0, \sigma^2 I)$ is gaussian noise added each iteration
    
4. same as single noisy gradient step 

 
$$
\mathbf{x}' = \mathbf{x} - \gamma\nabla E(\mathbf{x})
$$

1. Goal: minimize KL divergence between data dist p(x^0) and samples of DDPM q(x^0) with loss for each diffusion step as mean-squared error between pred and true noise:
   
$$
\mathcal{L} = \text{MSE}(\varepsilon^k, \varepsilon_\theta(\mathbf{x}^0 + \varepsilon^k, k))
$$
    

### Modifications to regular Diffusion:

1. output x represents robot actions
2. denoising process is conditioned on input observation $O_t$
    1. DP approximates conditional dist $p(A | O)$ instead of joint dist of both $(p(A,O))$
        1. Thus no need to infer future states to speed up process
3. New Denoising step: 
    
$$
\mathbf{A}_t^{k-1} = \alpha(\mathbf{A}t^k - \gamma\varepsilon\theta(\mathbf{O}_t, \mathbf{A}_t^k, k) + \mathcal{N}(0,\sigma^2I))
$$

(scaled prev action - predicted noise + noise for non-deterministic actions)

1. New Loss: 
    
$$
\mathcal{L} = MSE(\varepsilon^k, \varepsilon_\theta(\mathbf{O}_t, \mathbf{A}_t^0 + \varepsilon^k, k))
$$
    

(diff between predicted action and actual action)

**Closed Loop action-sequence prediction**:  

## Advantages

1. High scalability to high-dimensional action spaces so can predict sequence of actions
2. Energy-based policies: require negative sampling to estimate normalization constant (causes training instability) (WHAT is negative sampling?)
3. Receding-horizon control: readjustment but still smooth trajectories
    1. using latest $T_o$ steps of obs $O_t$, predict $T_p$ steps of action and execute $T_o$ of them
    2. encourages action consistency over time while remaining adaptable (HOW?)
4. Visual conditioning
    1. “visual obs treated as conditioning instead of part of joint data distribution”: WHAT does joint data dist consist of?
5. Time series diffusion transformer
    1. “minimizes the over-smoothing effects of typically CNN-based models”
    2. actions with noise $A_t^k$ passed in as input tokens for transformer decoder, with sinusoidal embeddings for diffusion iteration $k$ prepended as first token
    3. observation transformed into obs embedding sequence by shared mLP, then passed to transformer decoder as input features
    4. gradient predicted noise $\epsilon_\theta$ predicted by each output token of decoder stack

# Architecture

Common choices are CNNs and Transformers (both use the same visual encoders)

### CNN-Based Diffusion Policy

1. 1D Temporal CNN (WHAT is this?)
    1. instead of sliding 2D across height and width of image, it slides 1D across time and learns features between timesteps
    2. For a sequence of images (our observations), we first encode into feature vectors with our pretrained backbone (ResNet-18)
2. condition on obs $O$ with Feature-wise Linear Modulations(FiLM) and iter $k$
    1. **Feature-wise Linear Modulation:** NN layer that does feature-wise affine transformation conditioned on arbitrary input
        
$$
\text{FiLM}(F_{i,c}|\gamma_{i,c}, \beta_{i,c}) = \gamma_{i,c}F_{i,c} + \beta_{i,c}
$$
    
or output $= γ(c) * h + β(c)$ where $\gamma$ (scaling) and $\Beta$ (shifting) are functions of the conditioning, usually neural nets, and $h$ is the feature to modify (each has its own gamma and beta)
    
3. Don’t condition on goal due to receding horizon, still possible to add back in though

CNN performs poorly when desired trajectory changes sharply (perhaps since temporal convolutions bias to low-frequency outputs)

### Time-Series Diffusion Transformer

**Goal:** reduce over-smoothing effect of CNNs, adopts transformer architecture from minGPT to action prediction

1. actions with noise $A^k$ plus prepended sinusoidal embedding for diffusion iter $k$ passed into transformer decoder (these are our tokens)
2. observation $O$ becomes observation embedding sequence through MLP (shared between all obs), then passed to transformer decoder and the action tokens cross-attend to the observation tokens (each attention layer has access to the same observations)
3. added noise/”gradient” $\epsilon_\theta(O_t, A_t^k, k)$ predicted by each output token of decoder stack (action space dim)

### Visual Encoder

non-pretrained ResNet-18 with

1. spacial softmax pooling instead of global average pooling to maintain spatial info
    1. spatial softmax pooling: softmax (e^xa/sum e^xa over vector) region where a is learned, then expected 2D pos of softmax prob dist computed then linearly downsampled
2. replacing batchnorm with groupnorm
    1. groupnorm: divide channels of feature vector into groups and 1-norm each group
    2. important when norm layer used with Exponential Moving Average (in DDPMS usually) which is weighted previous EMA plus 1 - weighted current val

### Noise Schedule

controls extent to which DP learns high/low frequency characteristics of actions

Square Cosine Schedule from iDDPM experimentally works best

### Accelerating Inference for real-time control

Denoising Diffusion Implicit Models decouple number of denoising iters in train and inference, so can use fewer iters for inference 

Experimentally, DDIm with 100 training iters and 10 inference iters leads to 0.1s inference latency on NVIDIA 3080 GPU.

# Properties

### Model Multi-Modal Action Distributions

1. DP excels at multi-modal action inference (can take multiple different trajectories for the same objective)
    1. **action basins:** valleys in diffusion gradient map which are groups of similar actions
    2. Due to stochastic initialization and noise added in the navigation (denoising) stage, we can end off in different action basins which achieve the same goal, for example:
    
    ![Diffusion Trajectories](../../Images/Diffusion%20Trajectories.webp)
    

### Synergy with Position Control

1. Position control action space (directly controlling joint positions) consistently better than velocity control because:
    1. action multimodality more pronounced in pos-control
    2. less compounding error effects so better for action-sequence prediction
    3. In both position and velocity control we use feedback to control force to control acceleration based on desired position or velocity

### Benefits of Action-Sequence Prediction

1. Predicting a sequence of actions (entire trajectory) usually leads to difficulty sampling in high-dimensional spaces, but DPPM built for high output dimension spaces (like images)
2. Leads to smooth trajectories across time, long-horizon planning, and robust to idle actions (pauses in teleoperation data) since predicts sequences

### Training Stability

**Energy-Based Model:** lower energy to more desirable/likely states and higher to less likely/desirable states to create energy landscape

the probability of a state is proportional to $e^{-E[x]}$:

$$
p_\theta(\mathbf{a}|\mathbf{o}) = \frac{e^{E_\theta(\mathbf{o},\mathbf{a})}}{Z(\mathbf{o}, \theta)}
$$

or essentially, the probability of a state conditioned on an action is equal to the probability of that state-action pair in the energy landscape divided by an intractable normalization constant to make the landscape a probability distribution

Diffusion models can be seen as learning the gradient of the energy landscape through the noise prediction network, and stepping down to desirable states through denoising

Instead of needed $Z(o, \theta)$, Diffusion models estimate $Z(a, \theta)$ all at once by modeling the score function, and $z$ term in gradient of log prob is 0 so score function only depends on energy gradient, so more stable than methods like Implicit Behavioral Cloning since no need to estimate $Z$ using negative samples (estimates which should have high energy)

but $z$ is necessary in those methods to normalize and prevent exploding energy values. Diffusion is auto-anti exploding eneergyv alues because the model is estimating finite noise values for the gradient!

# Key Findings

1. DP expresses short-horizon multimodality (multiple ways to achieve the same goal)
2. DP expresses long-horizion multimodality (different subgoals in inconsistent order)
3. DP better leverages position control (vs velocity control)
4. longer action horizon better for consistent actions and to overcome idle actions 
5. low latency 
6. stable training


For a more hands-on look at Action Diffusion, check out this Colab Notebook by the original authors of Diffusion Policy: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing#scrollTo=OknH8Qfqrtc9)

[[Prev]](../../0:%20Fundamentals/0.2:%20Denoising%20Diffusion%20Probabilistic%20Models/DDPMs.md) [[Next]](../1.2:%20Components%20of%20Diffusion%20Policy/DP%20Components.md)