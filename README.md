![Sora Diffusion Particle Generation](./Images/Screenshot%202024-12-29%20at%202.30.30 PM.png "Sora Diffusion Particle Generation")

# Action Diffusion Handbook

### TODO: Revise

This handbook teaches from the fundamentals of diffusion through diffusion policy and transformers to flow matching in pi0. 

Each section contains a markdown file(s), original papers, and some example code. The markdown files explain the contents of papers more clearly and concisely. Feel free to skip around, and also add your own notes/papers/code you think are worth including through a PR! Start at section 1 if you are already familiar with regular diffusion.

### Contents: 

**Section 0: Fundamentals**

[0.1: Variational Generative Inference](./0:%20Fundamentals/0.1:%20Variational%20Generative%20Inference/Variational%20Generative%20Inference.md)

[0.2: Denoising Diffusion Probabilistic Models (DDPMs)](./0:%20Fundamentals/0.2:%20Denoising%20Diffusion%20Probabilistic%20Models%20(DDPMs)/Denoising%20Diffusion%20Probabilistic%20Models%20(DDPMs).md)

**Section 1: Diffusion Policy** 

[1.1: Action Diffusion](./1:%20Diffusion%20Policy/1.1:%20Action%20Diffusion/Action%20Diffusion.md)

[1.2: Components of Diffusion Policy](./1:%20Diffusion%20Policy/1.2:%20Components%20of%20Diffusion%20Policy/DP%20Components.md)

**Section 2: Diffusion Transformer**

[2.1: Diffusion Transformer (DiT)](./2:%20Diffusion%20Transformer/2.1:%20DiT/DiT.md)

[2.2: Ingredients of Robotic Diffusion Transformers](./2:%20Diffusion%20Transformer/2.2:%20Ingredients%20of%20Robotic%20Diffusion%20Transformers/Ingredients%20of%20Robotic%20Diffusion%20Transformers.md)

[2.3: RDT-1B: Diffusion Foundation Model for Bimanual Manipulation](./2:%20Diffusion%20Transformer/2.3:%20RDT-1B/RDT-1B.md)

**Section 3: Flow Matching**

[3.1: Continuous Normalizing Flows](./3:%20Flow%20Matching/3.1:%20Continuous%20Normalizing%20Flows/CNFs.md)

[3.2: Flow Matching](./3:%20Flow%20Matching/3.2:%20Flow%20Matching/Flow%20Matching.md)

[3.3: Conditional Flow Matching](./3:%20Flow%20Matching/3.3:%20Conditional%20Flow%20Matching/Conditional%20Flow%20Matching.md)

[3.4: Diffusion and Optimal Transport Flows](./3:%20Flow%20Matching/3.4:%20Diffusion%20and%20Optimal%20Transport%20as%20Flows/Diffusion%20Flow.md)

[3.5: pi0](./3:%20Flow%20Matching/3.5:%20pi0/pi0.md)

### Diffusion Visualization with Genesis

This Colab Notebook has the elements set up for training a Diffusion Transformer Policy on any data dsitribution, including a simple generated one, then visualizing the forming of the learned Diffusion Transformer trajectories from randomly distributed noise particles to a stable aligned trajectory.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YHs_KuSM5f3AuykjkCBfaHNYTZKg08-a)

It uses [this forked DiT repository](https://github.com/AlmondGod/dit_policy).

![R2](./Images/Screenshot%202024-12-29%20at%206.29.43 PM.png)

# Compressed Journey from Diffusion to $\pi_0$ Flow Matching

## Denoising Diffusion

In Denoising Diffusion Probabilistic Models (DDPMs), output generation is modeled as a denoising process.

1. Starts from $x^k$ sampled from Gaussian noise
2. K iterations of denoising to produce intermediate actions while decreasing noise $x^k .. x^0$ until desired noise-free output $x^0$ formed
3. Denoising follows the equation
    
$$
\mathbf{x}^{k-1} = \alpha(\mathbf{x}^k - \gamma\varepsilon_\theta(\mathbf{x}^k,k) + \mathcal{N}(0,\sigma^2I))
$$

where $\epsilon_\theta$ is noise prediction network and $N(0, \sigma^2 I)$ is gaussian noise added each iteration
    
4. For your intuition, this is the same as single noisy gradient step:

$$
\mathbf{x}' = \mathbf{x} - \gamma\nabla E(\mathbf{x})
$$

1. Goal: minimize KL divergence between data dist p(x^0) and samples of DDPM q(x^0) with loss for each diffusion step as mean-squared error between pred and true noise:
   
$$
\mathcal{L} = \text{MSE}(\varepsilon^k, \varepsilon_\theta(\mathbf{x}^0 + \varepsilon^k, k))
$$
    

## Action Diffusion

1. Output $x$ represents robot actions instead of the typical image
2. Denoising process is conditioned on input observation $O_t$
    1. Action Diffusion approximates conditional dist $p(A | O)$ instead of joint dist of both $(p(A,O))$
        1. Thus, there is no need to infer future states/observations, which speeds up the process
3. Our new Denoising step is 
    
$$
\mathbf{A}_t^{k-1} = \alpha(\mathbf{A}t^k - \gamma\varepsilon\theta(\mathbf{O}_t, \mathbf{A}_t^k, k) + \mathcal{N}(0,\sigma^2I))
$$

, which is similarly the scaled previous action minus predicted noise plus noise for non-deterministic action output.

4. Our new loss is the mean squared difference between predicted actions and actual actions:
    
$$
\mathcal{L} = MSE(\varepsilon^k, \varepsilon_\theta(\mathbf{O}_t, \mathbf{A}_t^0 + \varepsilon^k, k))
$$

## Diffusion Transformer


- first learn autoencoder encoding of x into latent space of zs, then learn Diffusion Model in space zs, then generate with diffusion model and learned decoder
- Theory is better Diffusion Model since vastly lower latent dim than data dim

### Architecture

Like Vision Transformer (ViT), operates on sequence of patches

1. Patchify: converts spatial input into sequence of T tokens of dim d with linear embedding
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

## RDT-1B
**Robotics Diffusion Transformer:** 1.2B param lagnuage-condtioned bimanual manipulation with vision foundation model

- fine-tuned on self-created dataset, 0-shot generalizable to new objects/scenes and 1-5 demo learning of new skills

**Physically Interpretable Unified Action Space:** same action representation for different robots to preserve universal semantics of actions for cross-embodiment knowledge transfer

language instruction $l$, observation $o_t$ at time $t$, produces action $a_t$, controls two arms to achieve goal specified in $l$

$o_t$ is a triple of:

1. $(X_t - T_img + 1 , … , X_t)$: sequence of past RGB images of size $T_img$
2. $z_t$: the low-dimensional robot proprioception (action $a_t$ is some subset of this)
3. $c$: control frequency

Not enough hardware-specific data, so pre-train on multi-robot data using a unified action space for any robot hardware, then fine-tune on the specific robot hardware.

For model architecture, we need **expressiveness** for multi-modal action distribution and **scalability** for generalization.

1. Encoding inputs (probabilistic masking to prevent overreliance on one modality)
    1. **Proprioception**/**action-chunk**/**control frequency** (all low-dim) encoded with MLP with Fourier Features to capture high-f changes (learned)
    2. **Images** to compact representations with image-text-aligned pre-trained vision encoder sigLIP (weights fixed)
    3. **Language** to embeddings with pre-trained Transformer language model T5-XXL (weights fixed)
2. Diffusion Transformer (DiT) backbone modified:
    1. **QKNorm:** to avoid gradient instability in attention from dramatically changing values and different joint ranges of robot proprioception data
        1. normalizes query and key matrixes before computing attention scores
    2. **RMSNorm**: root mean square normalization by squaring elements, taking mean, taking sqrt mean, dividing by that value
        1. instead of LayerNorm, which usually subtracts mean then normalizes by stdev then scales and shifts
        2. no centering operation (only normalizing) so no token/attention shift
    3. **MLP Decoder:** for nonlinear robot actions, replace final linear decoder with nonlinear MLP decoder 
    4. **Alternating Condition Injection:** use cross-attention to accommodate the varied-length image text conditions
        1. As opposed to typical class label condition compressed into single token then  Adaptive Layer Norm (class label embedding are inputs for function to generate scale and shift params for layer) applied
        2. inject images and text at alternating layers since image tokens usually way more and overshadow text tokens f simultaneously injected

## Flow Matching

**Continuous Normalizing Flows (CNFs):** generative model for arbitrary probability paths (superset of paths modeled by Diffusion processes)

**Flow Matching:** simulation-free method to train CNFs by regressing vector fields of fixed conditional probability paths

- When applied to Diffusion paths, flow matching is a robust/stable training method
- Flow Matching can also use non-Diffusion probability paths like Optimal Transport

Start with pure noise $p_0$, neural net vector field defines flow that smoothly moves each point along path, points carry probability masses with them and determinant ensures probability preserves (all points in the distribution integrate to 1), and ends in complex distribution $p_1$.

**Goal:** 

In flow matching, we don’t have the vector field CNF model. We only have data points we know are desirable, and we want to create a vector field which will naturally flow any random point to those desirable points!

Essentially, we just need to learn the vector field! if we learn this, then we can sample any arbitrary point and follow the flow defined by the vector field to some desired probability distribution we want.

Formally,

$q_1$ is a RV distributed according to unknown dist $q(x_1)$, and we only have access to samples

**Goal**: match target probability path $p_t$ flowing from simple dist $p_0$ (ex some random normal dist) to dist $p_1$ which Is approximately equal in dist to $q$ (essentially we have data about desirable ending probability dists, and we want to create a **quality vector field (generative model)** which flows to areas close around these samples from $q_1$)

Given target $p_1(x)$ and vector field $u_1(x)$ to generate $p_t(x)$, Flow Matching loss is 

$$
\mathcal{L}{\text{FM}}(\theta) = \mathbb{E}_{t,p_t(x)}\|v_t(x) - u_t(x)\|^2
$$

or the expected difference between our neural network predicting the vector field and the actual vector field $u_t$

At zero loss the learned CNF model generates $p_t(x)$ (our desired final data dist)

The problem is that **we do not know $p_t$ directly or $u_t$ at all**! So the above is intractable on its own since we don't know what appropriate $p_t$ and $u_t$ are

However, we can use Conditional Flow Matching.

### Conditional Flow Matching

Conditional Flow Matching objective: 

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,q(x_1),p_t(x|x_1)} \|v_t(x) - u_t(x|x_1)\|^2
$$

$$
\text{where } \ t \sim \mathcal{U}[0,1], \ x_1 \sim q(x_1), \ \text{and} \ x \sim p_t(x|x_1)
$$

or the expected difference between vector field and sampling-estimated true field

Essentially, we sample $x_1$ from data dist and $x$ (end dist) from $p(x|x_1)$ to get $(x,u-t(x))$ pairs to train final vector field $v_t$


So, at last, we have our final Flow Matching Method:

1. Have a bunch of data samples $x_1$ which are our desirable instances samples by true desirable distribution $p_1$ 
2. sample from some standard normal $p_0$ (a bunch of random points weighted according to normal distribution)
3. Sample $x$ from $p_0$ and $x_1$, and now $p_t(x|x_1)$ is some path which is an interpolation between $x$ and $x_1$ (which WE DEFINE however we want as long as it is continuous, thus it is easy to sample from!)
4. sample a point along that path, and compare our current neural net vector field $v_t(x)$ with $u_t(x|x_1)$ at that point (we design $u_t(x|x_1)$, can be simple $(x_1 - x)$ which always pushes points towards $x_1$, so we always know exactly what it is)

Conditional Flow Matching loss works with any conditional PP $p_t(x|x_1)$ and any conditional VF $u_t(x|x_1)$

However, the best choice for:

- $p_t(x|x_1)$: a Gaussian at each timestep where mean $u_t(x_1)$ moves from 0 to $x_1$ and stdev $\sigma_t(x_1)$ shrinks from 1 to $\sigma_{\text{min}}$ (final stdev around $x_1$ for $p_1$), so: 
    
$$
p_t(x|x_1) = \mathcal{N}(x|\mu_t(x_1), \sigma_t(x_1)^2I)
$$
    
- $u_t(x|x_1)$: a simple vector field which pushes points toward the means along the path $p_t$ and accounts for shrinking variance: 
    
$$
u_t(x|x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)}(x - \mu_t(x_1)) + \mu_t'(x_1)
$$
    

### Diffusion as Flow Matching

Diffusion is a subset of flow matching. How do we get the Denoising Diffusion process from flow matching?

**Variance Preserving:** noise added and preserving total variance (modern diffusion with alpha scaling)

For Variance Preserving Diffusion, we choose 

$$
\sigma_t(x_1) = \sqrt{1 - \alpha_{1-t}^2}\\
\mu_t(x_1) = \alpha_{1-t}x_1 \\
\text{ where } \alpha_t = e^{-\frac{1}{2}T(t)}, \quad T(t) = \int_0^t \beta(s)ds
$$

We are combining the Diffusion Condition Vector Field with Flow Matching objective, which they claim is better than score matching objective. They argue Diffusion technically never approaches true datapoints $x_1$ but just approximates/approaches them and can’t reach them in finite time, whereas in Flow Matching we exactly define the ends of our flow paths to be our $x_1$s

Essentially, it gives us much better theoretical guarantees!

## Physical Intelligence's VLA Flow Foundation Model

A generalist robot foundation model consisting of an “action expert” which uses conditional flow matching to augment a pretrained Vision-Language Model (VLM).

“high precision and multimodal modeling” so ideal for “high-frequency dexterous tasks” up to 50Hz.

### Data Space

$p(A_t|o_t)$ where $A_T = [a_t, a_{t + 1}, …, a_{t + H - 1}]$ is an action chunk of future actions with horizon H = 50  and $o_t$ is an observation

$o_t$ is multiple RGB images, language command, and proprioception so $o_t = [I_t^1, …, T_t^n, l_t, q_t]$ where $I_t^i$ is the ith image, $l_t$ is the language token sequence, $q_t$ is the joint angle vector (proprioception)

### Architecture

Uses OpenSource 3B param VLM PaliGemma and add 300M param action expert init from scratch

Images passed through Vision Transformers (ViT) then with language passed to pretrained 3B VLM. 

images and states encoded with corresponding encoders and projected through linear projection layer into same space as language tokens

Then these outputs and proprioception and noise passed through the denoising action expert to output a sequence of future actions $A_T = [a_t, a_{t + 1}, …, a_{t + H}]$

for each $a_t$ in $A_T$, there is a corresponding action token fed through action expert

The architecture is also inspired by [Transfusion](https://arxiv.org/pdf/2408.11039), which trains single transformer using multiple objectives. Unlike Transfusion, $\pi_0$ also uses a separate set of weights for robotics specific (action and state) tokens led to improved performance. This is analogous to Mixture of Experts with 2 mixture elements:
1. **VLM**: for image and text inputs
2. **Action expert:**   robotics specific inputs/outputs such as proprioception and actions

action expert uses bidirectional attention mask so all action tokens attend to each other

### Training

tokens for discrete outputs (language) supervised by cross-entropy loss (standard for decoder-only transformers)

tokens for continuous outputs (vision/actions/states) supervised by flow-matching loss applied on individual sequence elements 

in training, supervise action tokens using conditional flow matching loss

$$
L^\tau(\theta) = \mathbb{E}_{p(\mathbf{A}_t|\mathbf{o}_t),q(\mathbf{A}_t^\tau|\mathbf{A}t)}|\mathbf{v}\theta(\mathbf{A}_t^\tau, \mathbf{o}_t) - \mathbf{u}(\mathbf{A}_t^\tau|\mathbf{A}_t)|^2
$$

where subscripts are robot timesteps and superscripts and flow matching timesteps between 0 and 1

essentially minimize expected difference between predicted vf and actual vf  over actions conditioned on the current obs

CFM using linear Gaussian (can also be Optimal Transport) probability path

$$
q(\mathbf{A}_t^\tau|\mathbf{A}_t) = \mathcal{N}(\tau\mathbf{A}_t, (1-\tau)\mathbf{I})
$$

### Action Expert Training Process:

sample random noise $\epsilon \sim N(o,I)$, 

compute noisy actions $A_t^{\tau} = \tau A_t + (t - \tau)\epsilon$, 

train network outputs $v_\theta(A_t^\tau, o_t)$ to match denoising vector field $u(A_t^\tau | A_t) = \epsilon - A_t$

in training, sample flow matching timestep $\tau$ from beta distribution that emphasizes lower (noisier) timesteps

### Inference

In inference, generate actions by integrating learned vector field from $\tau = 0…1$ starting with random noise $A_t^0 \sim N(0,I)$

and using forward Euler integration rule

$$
\mathbf{A}_t^{\tau+\delta} = \mathbf{A}t^\tau + \delta\mathbf{v}\theta(\mathbf{A}_t^\tau, \mathbf{o}_t)
$$

where $\delta$= 0.1 is integration step size (10 integration steps)

Can infer efficiently by caching attention keys and values from prefix $o_t$ and only recomputing suffix corresponding to action tokens for each integration step

## Future

We've gone from simple diffusion to conditional flow matching for robotic foundation models in pi0.

The above was a highly compressed version of what's in the repository, I encourage you to check it out starting at the [beginning](./0:%20Fundamentals/0.1:%20Variational%20Generative%20Inference/Variational%20Generative%20Inference.md), as well as the original papers linked in each folder.