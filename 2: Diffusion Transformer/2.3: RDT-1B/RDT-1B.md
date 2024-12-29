![RDT-1B](../../Images/Screenshot%202024-12-29%20at%203.03.54 PM.png)

# Action Diffusion Transformer Foundation Model

From [RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/pdf/2410.07864)

**Robotics Diffusion Transformer:** 1.2B param lagnuage-condtioned bimanual manipulation with vision foundation model

- fine-tuned on self-created dataset, 0-shot generalizable to new objects/scenes and 1-5 demo learning of new skills

**Physically Interpretable Unified Action Space:** same action representation for different robots to preserve universal semantics of actions for cross-embodiment knowledge transfer

# Problem Setting

language instruction $l$, observation $o_t$ at time $t$, produces action $a_t$, controls two arms to achieve goal specified in $l$

$o_t$ is a triple of:

1. $(X_t - T_img + 1 , … , X_t)$: sequence of past RGB images of size $T_img$
2. $z_t$: the low-dimensional robot proprioception (action $a_t$ is some subset of this)
3. $c$: control frequency

Not enough hardware-specific data, so pre-train on multi-robot data then fine-tune on this one

cross-embodiment data: $N$ tuples of sequences of $(l, o, a)$ where tuple $i$ is a sequence of length $T_i$ and data point $(i,t)$ is the $i$th tuple in the $t$th datapoint

# Motivation for Diffusion

If modeled deterministically as $(l, o_t) \rightarrow a_t$, we regress the $(l,o,a)$ tuples and will likely learn infeasible OOD actions like the average of different modes 

- for example: data contains go around obstacle one way, then around the other way, we learn the infeasible “going through” trajectory

We instead learn continuous conditional dist $p(a_t|l,o_t)$, where Diffusion models excel in diversity and quality

# Architecture

need a model with **expressiveness** for multi-modal action distribution and **scalability**

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

# Data

We want each dimension in the unified space to have clear meaning (ex: right shoulder, elbow, etc), padding any missing elements

We can simply manually correspond each dimension of the current robot's joint space to the semantic equivalent in the unified action space. 

RDT-1B implements this as a 128-dimensional vector with the below correspondence:

![RDT-1B Unified Action Space](../../Images/Screenshot%202024-12-28%20at%2011.49.11 PM.png)

[[Prev]](../2.2:%20Components%20of%20Diffusion%20Transformers/DiT%20Components.md) [[Next]](../../3:%20Flow%20Matching/3.1:%20Continuous%20Normalizing%20Flows/CNFs.md)