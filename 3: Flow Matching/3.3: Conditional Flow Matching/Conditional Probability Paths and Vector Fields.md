# Conditional Probability Paths and Vector Fields

From [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)

Conditional Flow Matching loss works with any conditional PP $p_t(x|x_1)$ and any conditional VF $u_t(x|x_1)$

However, the best choice for:

1. $p_t(x|x_1)$: Gaussian at each timestep where mean $u_t(x_1)$ moves from 0 to $x_1$ and stdev $\sigma_t(x_1)$ shrinks from 1 to $\sigma_{\text{min}}$ (final stdev around $x_1$ for $p_1$), so: 
    
$$
p_t(x|x_1) = \mathcal{N}(x|\mu_t(x_1), \sigma_t(x_1)^2I)
$$
    
2. $u_t(x|x_1)$ is a simple vector field which pushes points toward the means along the path $p_t$ and accounts for shrinking variance: 
    
$$
u_t(x|x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)}(x - \mu_t(x_1)) + \mu_t'(x_1)
$$
    

### Derivation if Interested:

For a given prob path, there are infinite number of vector fields to generate it, but many have components which don't effect underlying dist

The simplest flow itself would be adding mean and scaling by stdev:

$$
\psi_t(x) = \sigma_t(x_1)x + \mu_t(x_1)
$$

which pushes all random noise p_0 to the path p_t(x|x_1):

$$
[\psi_t]_* p(x) = p_t(x|x_1)
$$

The equivalent vector field that implements this flow is just the derivative of the flow for a given x:

$$
\frac{d}{dt}\psi_t(x) = u_t(\psi_t(x)|x_1)
$$

We can solve for $u_t$ in closed form since our flow $\phi_t$ is a simple affine (of form $Ax + b$ for linear $A$ and constant $b$), then just taking $d/dt\phi_t(x)$ gets us:

$$
\sigma’_t(x_1)x + \mu’_t(x_1)
$$

We know from rearranging our simple flow that 

$$
x = \phi_t(x) - \mu_t(x_1) / \sigma_t(x_1)
$$

So substituting in we get 

$$
d/dt \phi_t(x) = \sigma’_t(x_1) ((\phi_t(x) - \mu_t(x_1))/(\sigma_t(x_1)) + \mu’_t(x_1))
$$

Then we know

$$
d/dt \phi_t(x) = u_t(\phi_t(x)|x_1)
$$

Thus we finally get 

$$
u_t(x|x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)}(x - \mu_t(x_1)) + \mu_t'(x_1)
$$

For a code example, here's a simple 2D Flow Matching colab notebook: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DPXh1Qw1GXGGpgBLy7qjGvkH1MgB9lxz#scrollTo=SRoDWaO7mEW9)

[[Prev]](./Conditional%20Flow%20Matching.md) [[Next]](../3.4:%20Diffusion%20and%20Optimal%20Transport%20as%20Flows/Diffusion%20Flow.md)