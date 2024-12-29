![DP Components](../../Images/Screenshot%202024-12-29%20at%202.55.36 PM.png)

# Features of Diffusion Policy

From [Unpacking the Individual Components of Diffusion Policy](https://arxiv.org/pdf/2412.00084)

1. **Observation Sequence as Input:** critical for tasks requiring Absolute Control but  irrelevant for Delta Control Mode
    1. Absolute Control: policy outputs absolute position/velocity
    2. Delta Control Mode: policy outputs delta change in position/velocity
2. **Action Sequence as Output:** for real-time control tasks, shorter/single action horizons better, otherwise long action horizons better
    1. Also means every single action is planned with future context accounted for
3. **Receding Horizon Actions:** DP predicts long sequence of actions but only executes first few (main effect is only on long-horizon tasks)
4. **Denoising network architecture:** MLP vs U-net vs Diffusion Transformer as denoising network backbone (MLP sufficient for easy tasks, U-net for harder)
5. **Observation inputs taken as FiLM** (Feature-wise Linear Modulation) conditioning instead of network input (significant enhancement on harder tasks)

[[Prev]](../1.1:%20Action%20Diffusion/Action%20Diffusion.md) [[Next]](../../2:%20Diffusion%20Transformer/2.1:%20DiT/DiT.md)