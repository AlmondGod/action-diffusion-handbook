![FM](../../Images/Screenshot%202024-12-29%20at%203.17.21 PM.png)

# Constructing $p_t$ and $u_t$ by sampling


From [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)


**Idea:** instead of trying to match our neural network vectro field to an entire true vector field, let’s find the individual vector fields which turn $p_0$ into each of the datapoint dists $x_1$

**Conditional Probability Path:** $p_t(x|x_t)$ such that $p_0(x|x_1) = p(x)$ and $p_1(x|x_1)$ is concentrated around $x = x_1$ such as normal dist with mean $x_1$

essentially $p_t(x|x_1)$ is the path from $p_0$ to a Gaussian centered at $x_1$

Remember, $q_1$ is the RV we can sample from which has unknown true dist $q(x_1)$, and which $p_1$ is approximately equal to:

**Marginal Probability Path:** marginalizing above over $q(x_1)$: 

$$
p_t(x) = \int p_t(x|x_1)q(x_1)dx_1
$$

which is essentially saying the path $x$ (from $p_0$ normal to $p_1$ approx $q$) is the integral of the final distribution times the path that goes from $x$ to $x_1$ centered dist

Essentially this is the **overall combined flows from all the conditional flows to each individual datapoint**

And we already defined that the marginal probability $p_1$ is close to data dist $q$, so:

$$
p_1(x) = \int p_1(x|x_1)q(x_1)dx_1 \approx q(x)
$$

**Marginalized Vector Field:**

$$
u_t(x) = \int u_t(x|x_1)\frac{p_t(x|x_1)q(x_1)}{p_t(x)}dx_1
$$

where $u_t(.|x_1): R^d → R^d$ is a conditional vector field that generates $p_t(.|x_1)$

Aggregating conditional VFs creates correct VF for modeling marginal probability path.

>Essentially this is the **overall combined VF from all the conditional VFs to each individual datapoint**

We can breakdown unknown intractable marginal VF into simpler conditional VFs which only depend on one data sample $x_1$

But due to intractable integrals in marginal PP and VF definitions, it is still intractable to compute $u_i$ and unbiased estimator of original Flow Matching objective

[[Prev]](./Flow%20Matching.md) [[Next]](../3.3:%20Conditional%20Flow%20Matching/Conditional%20Flow%20Matching.md)