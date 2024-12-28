# Conditional Flow Matching

From [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)

Conditional Flow Matching objective: 

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,q(x_1),p_t(x|x_1)} \|v_t(x) - u_t(x|x_1)\|^2
$$

$$
\text{where } \ t \sim \mathcal{U}[0,1], \ x_1 \sim q(x_1), \ \text{and} \ x \sim p_t(x|x_1)
$$

or the expected difference between vector field and sampling-estimated true field

Essentially, we sample $x_1$ from data dist and $x$ (end dist) from $p(x|x_1)$ to get $(x,u-t(x))$ pairs to train final vector field $v_t$

Note: $p(x|x_1)$ is any continuous flow from $p_0$ to Gaussian around $x_1$, which we can design 

Allows easily sampling unbiased estimates provided efficient:

1. sampling from $p_t(x|x_1)$
2. computing $u_t(x|x_1)$

Flow Matching Loss and Conditional Flow Matching loss have identical gradients w.r.t $\theta$

So optimizing CFM loss is equivalent in expectation to optimizing FM loss

Thus can train CNF to generate marginal probability path $p_t$ which approximates unknown data dist $q$ at $t = 1$, **without needing marginal** PP or VF**, only need conditional** PP and VF

$$\textbf{Theorem 2: } \text{Assuming that } p_t(x) > 0 \text{ for all } x \in \mathbb{R}^d \text{ and } t \in [0,1], \\
\text{ then, up to a constant} \text{ independent of } \theta, \mathcal{L}_{\text{CFM}} \text{ and } \mathcal{L}_{\text{FM}} \text{ are equal. } \\
\text{Hence, }\nabla_\theta\mathcal{L}_{\text{FM}}(\theta) = \nabla\theta\mathcal{L}_{\text{CFM}}(\theta). $$

So, at last, we have our final Flow Matching Method:

1. Have a bunch of data samples $x_1$ which are our desirable instances samples by true desirable distribution $p_1$ 
2. sample from some standard normal $p_0$ (a bunch of random points weighted according to normal distribution)
3. Sample $x$ from $p_0$ and $x_1$, and now $p_t(x|x_1)$ is some path which is an interpolation between $x$ and $x_1$ (which WE DEFINE however we want as long as it is continuous, thus it is easy to sample from!)
4. sample a point along that path, and compare our current neural net vector field $v_t(x)$ with $u_t(x|x_1)$ at that point (we design $u_t(x|x_1)$, can be simple $(x_1 - x)$ which always pushes points towards $x_1$, so we always know exactly what it is)

[[Prev]](../3.2:%20Flow%20Matching/Sampling%20Flows.md) [[Next]](./Conditional%20Probability%20Paths%20and%20Vector%20Fields.md)