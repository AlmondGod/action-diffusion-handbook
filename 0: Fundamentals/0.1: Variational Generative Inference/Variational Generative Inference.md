# Variational Generative Inference  

Heavily inspired by https://xyang35.github.io/2017/04/14/variational-lower-bound/ by Xitong Yang
and https://blog.evjang.com/2016/08/variational-bayes.html by Eric Jang

In machine learning, we are often presented with **statistical inference** problems, where we want to infer the value of a random variable given the value of another random variable.

How do we do this?

In **optimization problem**, we find parameters that minimize an objective function.

The **inference-optimization duality** tells us that we can in fact view statistical inference problems as optimization problems. 

**Variational Bayesian (VB) Methods:** have the inference-optimization duality feature

**variational/evidence-based lower bound:** 

- plays essential role in Variationa Bayesian Method derivations

## Setup

$X$ observations/data and $Z$ hidden/latent variables (such as params)

$P(X)$: prob dist over $X$ 

$p(x)$: density function dist of $X$

Posterior distribution of hidden variables ($z$ given a particular $x$): 

$$
p(z|x) = \frac{p(x | z)p(z)}{p(x)} = \frac{p(x|z)p(z)}{\int_z p(x,z)}
$$

Sampling from $p(z|x)$ can, for example, give us a classifier where $z$ is “is a cat” and $x$ is “raw pixel observations”

$P(z)$ is prior probability, for example, 1/3 of all existent images are cats

$p(x|z)$ is likelihood, for example how probable a certain image is given we know it is a cat. Sampling $x$ from this distribution will give us a generative model!

$p(x|z)$ is too hard to compute, so we start with a parametric dist (ex Gaussian) $Q_\phi(z|x)$ and adjust $\phi$ to make $Q$ close to $p$  by minimizing the KL-Divergence between the two(see Deriving KL-Divergence)

## Deriving the Variational Lower Bound

If we want the log probabilities of our datapoints, we have that log dist of x equal log dist of x and z over all z:

$$
\log p(X) = \log \int_Z p(X,Z) 
$$

then multiply by q(Z) over q(Z) where q(Z) is an approximation of tru eposterior distribution p(Z|X)

$$
= \log \int_Z p(X,Z)\frac{q(Z)}{q(Z)} 
$$

Then this is the expectation over q of p(x,z)/q(z) (the top q(z) is included in the expectation as prob of that variable, p(x,z)/q(z) is the value)

$$
= \log \left(\mathbb{E}_q\left[\frac{p(X,Z)}{q(Z)}\right]\right) 
$$

apply Jensen’s Inequality: $f(E[X]) \leq E[f(x)]$ (a function of an expectation of X is always at most the expectation of a function of X) to move log inside expectation

$$
\geq \mathbb{E}_q\left[\log \frac{p(X,Z)}{q(Z)}\right] 
$$

and finally,change $\log(p(x,z)/q(z))$ to $\log(p(x,z)) - \log(q(z))$, use LOE to separate into $E_q[\log p(x,z)] - E_q[\log(q(z))]$, and substitute in Shannon Entropy $H[Z] = -E_q[\log(q(z))]$.

Then our final lower bound for the log probability of the observations, our Variational Lower Bound, is:

$$
L = \mathbb{E}_q[\log p(X,Z)] + H[Z], \text{where } H[Z] = -\mathbb{E}_q[\log q(Z)]
$$


## Deriving KL-Divergence

$p(Z)$ doesn’t even show up in the above equation

computation of $p(Z|X)$ is often intractable/difficult to derive (ex: integrating all configurations of $z$ to compute the denominator). But we need it because (WHY?)

Variational Methods: approximate $q(Z)$ as close as possible to true posterior distribution $p(Z|X)$

parametrized as $q(Z|\theta)$, and find parameters to make $q$ close to posterior of interest $p(Z|X)$

Thus we need an objective to optimize which represents the closeness of $q(Z)$ and $p(Z|X)$, which is KL Divergence:

$$
\text{KL}[q(Z)||p(Z|X)] = \int_Z q(Z)\log \frac{q(Z)}{p(Z|X)} \\
= -\int_Z q(Z)\log \frac{p(Z|X)}{q(Z)} \\
= -\left(\int_Z q(Z)\log \frac{p(X,Z)}{q(Z)} - \int_Z q(Z)\log p(X)\right) \\
= -\int_Z q(Z)\log \frac{p(X,Z)}{q(Z)} + \log p(X)\int_Z q(Z) \\
= -L + \log p(X)
$$

(last step is because $\sum_z q(z) = 1$)

Rearranging we finally get 

$$
L = \log p(X) - \text{KL}[q(Z)||p(Z|X)]
$$

KL divergence is always > 1

thus $L \leq \log p(X)$, a lower bound of the log probability of the observations/data

Also, the difference between them is exactly the KL divergence between the approx and true posterior distributions

so the lower bound $L$ is exactly the log probability of the data if and only if the approximate posterior distribution is exactly the true posterior distribution

So, maximizing our ELBO is the same as minimizing the KL-divergence between $q(Z)$ and $q(x,z)!$