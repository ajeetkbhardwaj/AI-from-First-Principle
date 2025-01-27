#%%
import torch

class LinearNoiseScheduler:
    r"""
    

    """
    # read and explain code step by step for ddpm
    # 
    def __init__(self, num_timesteps, beta_start, beta_end):
        # 1. Initialization of all arguments of LinearNoiseScheduler class
        # num_timesteps: int = 1000
        self.num_timesteps = num_timesteps
        # beta_start: float = 1e-6
        self.beta_start = beta_start
        # beta_end: float = 0.1
        self.beta_end = beta_end
        
        # 2. linearly spaced betas from beta_start to beta_end
        # self.betas: torch.Tensor = torch.linspace(beta_start, beta_end, num_timesteps)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        # self.alphas: torch.Tensor = 1. - self.betas
        self.alphas = 1. - self.betas
        # self.alphas = product of all alphas
        self.alpha_cum_prod = torch.cumprod(self.alphas, 0)

        # Initialize all the varables for forward and reverse pass of equations
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    
    # adding noise to images each time steps
    def add_noise(self, original, noise, t):
        # BxHxW
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
    
    # learn the reverse sampled distribution
    def sample_prev_timesteps(self, xt, noise_pred, t):
        x0 = ()