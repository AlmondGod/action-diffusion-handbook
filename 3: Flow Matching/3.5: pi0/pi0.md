![pi0](../../Images/Screenshot%202024-12-29%20at%203.30.10 PM.png)

# Physical Intelligence's VLA Flow Foundation Model

From [$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control](https://www.physicalintelligence.company/download/pi0.pdf)

“action expert” uses conditional flow matching to augment a pretrained Vision-language model (VLM).

“high precision and multimodal modeling” so ideal for “high-frequency dexterous tasks” up to 50Hz.

## Data Space

$p(A_t|o_t)$ where $A_T = [a_t, a_{t + 1}, …, a_{t + H - 1}]$ is an action chunk of future actions with horizon H = 50  and $o_t$ is an observation

$o_t$ is multiple RGB images, language command, and proprioception so $o_t = [I_t^1, …, T_t^n, l_t, q_t]$ where $I_t^i$ is the ith image, $l_t$ is the language token sequence, $q_t$ is the joint angle vector (proprioception)

## Architecture

![piarch](../../Images/Screenshot%202024-12-28%20at%2011.56.42 PM.png "from the paper")

Uses OpenSource 3B param VLM PaliGemma and add 300M param action expert init from scratch

Images passed through Vision Transformers (ViT) then with language passed to pretrained 3B VLM. 

images and states encoded with corresponding encoders and projected through linear projection layer into same space as language tokens

Then these outputs and proprioception and noise passed through the denoising action expert to output a sequence of future actions $A_T = [a_t, a_{t + 1}, …, a_{t + H}]$

for each $a_t$ in $A_T$, there is a corresponding action token fed through action expert

The architecture is also inspired by [Transfusion](https://arxiv.org/pdf/2408.11039), which trains single transformer using multiple objectives. Unlike Transfusion, $\pi_0$ also uses a separate set of weights for robotics specific (action and state) tokens led to improved performance. This is analogous to Mixture of Experts with 2 mixture elements:
1. **VLM**: for image and text inputs
2. **Action expert:**   robotics specific inputs/outputs such as proprioception and actions

action expert uses bidirectional attention mask so all action tokens attend to each other

## Training

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

## Inference

In inference, generate actions by integrating learned vector field from $\tau = 0…1$ starting with random noise $A_t^0 \sim N(0,I)$

and using forward Euler integration rule

$$
\mathbf{A}_t^{\tau+\delta} = \mathbf{A}t^\tau + \delta\mathbf{v}\theta(\mathbf{A}_t^\tau, \mathbf{o}_t)
$$

where $\delta$= 0.1 is integration step size (10 integration steps)

Can infer efficiently by caching attention keys and values from prefix $o_t$ and only recomputing suffix corresponding to action tokens for each integration step

## Dataset

each training example is a timestep tuple of $(o_t, A_t)$

7 different robot configs: UR5e (7 DOF arm, wrist+shoulder camera), Bimanual UR5e, Franka, Bimanual Trossen, Bimanual ARX/AgileX, Mobile Trossen/ARX, Mobile Fibocom (2 6DOF arms+ 4DOF mobile base, 2 wrist+base cam) 

68 tasks

### Pretraining:

OXE, Bridge v2, DROID and $\pi$ dataset

lower frequency. (~5-10 Hz), diverse, 100s of Millions of timesteps

### Posttraining

fine-tuning on task-specific data

anywhere from 5-100 hours of data per task depending on complexity

[Here's a video](https://www.physicalintelligence.company/blog/pi0) of their State-of-the-Art results on a variety of tasks, including end-to-end laundry folding: 

[[Prev]](../3.4:%20Diffusion%20and%20Optimal%20Transport%20as%20Flows/Optimal%20Transport%20Flow.md)