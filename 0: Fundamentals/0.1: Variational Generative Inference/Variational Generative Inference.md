![VGI](../../Images/Flowing%20Wave.png)

# Variational Generative Inference 

Heavily inspired by https://xyang35.github.io/2017/04/14/variational-lower-bound/ by Xitong Yang
and https://blog.evjang.com/2016/08/variational-bayes.html by Eric Jang

Diffusion operates under the assumption that the data is generated from a prior distribution, and we want to sample from this distribution.

How do we create a simple generative model under this assumption, with no access to the distribution,using only data sampled from that distribution?

More concretely, we have $X$ observations/data and $Z$ hidden/latent variables (such as the parameters of such a model).

$p(x)$: probability density function distribution of $X$

$p(z)$ is prior probability, for example, 1/3 of all existent images are cats

$p(x|z)$ is likelihood, for example how probable a certain image is given we know it is a cat. Sampling $x$ from this distribution will give us a generative model! If we wanted to generate cats, we could set z to be "cat" and sample from this distribution.

So our goal is to have an accurate generative model $p(x|z)$ that we can sample from.

We can do so by trying to maximize the likelihood that our model will produce the data we can access.

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

This is the quantity we want to maximize to get the best generative model! However, we don't have access to the true posterior distribution $p(Z|X)$, so we need to approximate it.

## Deriving KL-Divergence

In variational methods (like Diffusion!), we approximate $q(Z)$ as close as possible to true posterior distribution $p(Z|X)$

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



$p(x|z)$ is too hard to compute, so we start with a parametric dist (ex Gaussian) $Q_\phi(z|x)$ and adjust $\phi$ to make $Q$ close to $p$ by minimizing the KL-Divergence between the two(see Deriving KL-Divergence)

# Why do we need this in diffusion?

This is, at its core, what Diffusion is trying to do: maximize the likelihood of the data by minimziing the KL-Divergence between the true posterior distribution and our approximation. See exactly how:

[[Next]](../0.2:%20Denoising%20Diffusion%20Probabilistic%20Models/DDPMs.md)