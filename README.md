![Sora Diffusion Particle Generation](./images/Screenshot%202024-12-29%20at%202.30.30 PM.png "Sora Diffusion Particle Generation")

# Action Diffusion Handbook

This handbook teaches from the fundamentals of diffusion through diffusion policy and transformers to flow matching in pi0. Start at section 1 if you are already familiar with regular diffusion.

Each section contains a markdown file(s), original papers, and some example code. The markdown files explain the contents of papers more clearly and concisely. Feel free to skip around, and also add your own notes/papers/code you think are worth including through a PR!

## Contents: 

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

### TODO: add header images 
### TODO: Revise

### TODO: add main section

## Diffusion Visualization with Genesis

This Colab Notebook has the elements set up for training a Diffusion Transformer Policy on any data dsitribution, including a simple generated one, then visualizing the forming of the learned Diffusion Transformer trajectories from randomly distributed noise particles to a stable aligned trajectory.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YHs_KuSM5f3AuykjkCBfaHNYTZKg08-a)

It uses [this forked DiT repository](https://github.com/AlmondGod/dit_policy).