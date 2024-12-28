# Flow Matching


From [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)


In flow matching, we don’t have the vector field CNF model. We only have data points we know are desirable, and we want to create a vector field which will naturally flow any random point to those desirable points!

Formally,

$q_1$ is a RV distributed according to unknown dist $q(x_1)$, and we only have access to samples

**Goal**: match target probability path $p_t$ flowing from simple dist $p_0$ (ex some random normal dist) to dist $p_1$ which Is approximately equal in dist to $q$ (essentially we have data about desirable ending probability dists, and we want to create a **quality vector field (generative model)** which flows to areas close around these samples from $q_1$)

Given target $p_1(x)$ and vector field $u_1(x)$ to generate $p_t(x)$, Flow Matching loss is 

$$
\mathcal{L}{\text{FM}}(\theta) = \mathbb{E}_{t,p_t(x)}\|v_t(x) - u_t(x)\|^2
$$

or the expected difference between our neural network predicting the vector field and the actual vector field $u_t$

At zero loss the learned CNF model generates $p_t(x)$ (our desired final data dist)

But of course, **we do not know $p_t$ directly or $u_t$ at all**! So the above is intractable on its own since we don't know what appropriate $p_t$ and $u_t$ are

However, **we can construct $p_t$ and $u_t$ using per-sample methods**, which gives a much more tractable Flow Matching objective

[[Prev]](../3.1:%20Continuous%20Normalizing%20Flows/CNFs.md) [[Next]](./Sampling%20Flows.md)