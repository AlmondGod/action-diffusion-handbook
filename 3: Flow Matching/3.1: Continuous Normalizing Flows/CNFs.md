![CNFs](../../Images/Screenshot%202024-12-29%20at%203.00.30 PM.png)

# 3.1: Continuous Normalizing Flows

From [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)

**Continuous Normalizing Flows (CNFs):** generative model for arbitrary probability paths (superset of paths modeled by Diffusion processes)

**Flow Matching:** simulation-free method to train CNFs by regressing vector fields of fixed conditional probability paths

- applied to Diffusion paths is a robust/stable training method
- Can also use non-Diffusion probability paths like Optimal Transport

Regular Diffusion has confined space of sampling probability paths so long training times and inefficient sampling

**Time-Dependent Vector-Field**: a field of [(0 or 1) by d] vectors at each point

$$
v : [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d
$$

**Probability Density Path:** on that vector field, we have a certain probability of being at this point (can by a multi-d probability)

$$
p : [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}_{>0}
$$

**Flow:** time-dependent diffeomorphic (WHAT is this?) map constructed with vector field v_t:

$$
\phi : [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d
$$

- Defined by ODE:
    
$$
\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x)) \\
\phi_0(x) = x
$$
    
Essentially the vector field defines the gradient of the flow, and the flow itself is just current position
    

**Continuous Normalizing Flow:** model of a vector field v_t with a neural network which gives a deep parametric model of the flow \phi_t

- reshapes simple prior density p_0 (pure noise) to more complex p_1 with “push-forward”:
    
$$
p_t = [\phi_t]_*p_0
$$
    
- where * operator is defined:
    
$$
[\phi_t]_*p_0(x) = p_0(\phi_t^{-1}(x)) \det\left[\frac{\partial\phi_t^{-1}}{\partial x}(x)\right]
$$
    

or essentially the current inverse flow x the determinant of the gradient of the inverse flow w.r.t x

Takes a point x and finds out where it came from/’goes backwards” (\phi^-1)d, 

adjust prob density by determining to account for how flow adjusts the space (stretch/transform)

if flow compresses region, prob density increases to preserve total prob (and inverse)

starts with pure noise p_0, neural net vector field defines flow that smoothly moves each point along path, points carry probability masses with them and determinant ensures probability preserves (all points in the distribution integrate to 1)

Ends in complex distribution

vector field generates a prob density path if flow satisfies the push-forward equation

**Goal:** 

Essentially, we just need to learn the vector field! if we learn this, then we can sample any arbitrary point and follow the flow defined by the vector field to some desired probability distribution we want.

> Originally, Continuous Normalizing Flows were trained with Maximum-Likelihood objectives (maximize likelihood of each datapoint being generated from path) 
> This Involved expensive Ordinary Differential Equation simulations so high time complexity and infeasible for high-dimensional state spaces (images/actions)



[[Prev]](../../2:%20Diffusion%20Transformer/2.3:%20RDT-1B/RDT-1B.md) [[Next]](../3.2:%20Flow%20Matching/Flow%20Matching.md)