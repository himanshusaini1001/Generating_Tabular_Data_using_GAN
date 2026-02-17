# train/training.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

# -------------------------
# ⚡ Disable Flash/Efficient SDP
# -------------------------
os.environ.setdefault("PYTORCH_SDP_BACKEND_DISABLE", "1")
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

# -------------------------
# Device info helper
# -------------------------
def get_device_info():
    if torch.cuda.is_available():
        return f"✅ Using GPU (CUDA) - {torch.cuda.get_device_name(0)}"
    else:
        return "⚠️ Using CPU (CUDA not available, training will be slower)"

# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        return self.pe[:L].unsqueeze(0).to(x.device)

# -------------------------
# Spectral Normalization
# -------------------------
def spectral_norm(module):
    """Apply spectral normalization to module."""
    try:
        from torch.nn.utils import spectral_norm as sn
        return sn(module)
    except:
        return module

# -------------------------
# Improved Generator with Layer Norm and Residual
# -------------------------
class Generator(nn.Module):
    def __init__(self, seq_len, vocab_size, noise_dim=128, cond_dim=32, d_model=256,
                 nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Improved input projection with layer norm
        self.input_fc = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, seq_len * d_model),
            nn.LayerNorm(seq_len * d_model)
        )
        
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        
        # Enhanced transformer with pre-norm architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, 
            batch_first=True,
            norm_first=True  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Improved output with layer norm
        self.out_fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Better weight initialization for faster convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, z, cond_vec, temperature=1.0):
        B = z.size(0)
        x = torch.cat([z, cond_vec], dim=1)
        x = self.input_fc(x)
        x = x.view(B, self.seq_len, -1)
        x = x + self.pos_enc(x)
        x = self.transformer(x)
        logits = self.out_fc(x) / temperature  # Temperature scaling for diversity
        return logits

# -------------------------
# Improved Discriminator with Spectral Normalization
# -------------------------
class Discriminator(nn.Module):
    def __init__(self, seq_len, vocab_size, cond_dim=32, d_model=256,
                 nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # Improved input with layer norm
        self.input_fc = nn.Sequential(
            nn.Linear(vocab_size, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Conditional embedding
        self.cond_fc = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        
        # Enhanced transformer with pre-norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Improved pooling with attention
        self.pool_fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.2)
        )
        
        # Apply spectral normalization to output layer
        self.out_fc = spectral_norm(nn.Linear(d_model, 1))
        
        self._init_weights()

    def _init_weights(self):
        """Better weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, seq_probs, cond_vec):
        x = self.input_fc(seq_probs)
        cond_proj = self.cond_fc(cond_vec).unsqueeze(1)
        x = x + cond_proj
        x = x + self.pos_enc(x)
        x = self.transformer(x)
        
        # Improved pooling: mean + max
        x_mean = x.mean(dim=1)
        x_max, _ = x.max(dim=1)
        x = x_mean + x_max  # Combination of both
        
        x = self.pool_fc(x)
        score = self.out_fc(x)
        return score.squeeze(1)

# -------------------------
# Condition Encoder
# -------------------------
class ConditionEncoder(nn.Module):
    def __init__(self, num_buckets, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(num_buckets, emb_dim)
    def forward(self, cond_ids):
        return self.emb(cond_ids)

# -------------------------
# WGAN-GP helpers
# -------------------------
def wgan_critic_loss(real_scores, fake_scores):
    return torch.mean(fake_scores) - torch.mean(real_scores)

def wgan_generator_loss(fake_scores):
    return -torch.mean(fake_scores)

def gradient_penalty(critic, real_data, fake_data, cond_vec, device='cpu', gp_weight=10.0):
    batch_size = real_data.size(0)
    eps = torch.rand(batch_size, 1, 1, device=device).expand_as(real_data)
    interp = eps * real_data + (1 - eps) * fake_data
    interp.requires_grad_(True)
    scores = critic(interp, cond_vec)
    grads = autograd.grad(outputs=scores, inputs=interp,
                          grad_outputs=torch.ones_like(scores),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)
    gp = gp_weight * ((grad_norm - 1) ** 2).mean()
    return gp

# -------------------------
# StackedGAN
# -------------------------
class StackedGAN:
    def __init__(self, seq_len, vocab_size, hidden_dim=128, lr=1e-4, target_gc=0.42,
                 device=None, cond_buckets=10, cond_dim=32, noise_dim=128, enable_early_stopping=False):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.target_gc = target_gc
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(get_device_info())

        self.cond_encoder = ConditionEncoder(cond_buckets, cond_dim).to(self.device)
        self.generator = Generator(seq_len, vocab_size, noise_dim=noise_dim, cond_dim=cond_dim,
                                   d_model=hidden_dim).to(self.device)
        self.discriminator = Discriminator(seq_len, vocab_size, cond_dim=cond_dim,
                                           d_model=hidden_dim).to(self.device)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr*2, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Learning rate schedulers for better convergence
        self.g_scheduler = optim.lr_scheduler.ExponentialLR(self.g_optimizer, gamma=0.95)
        self.d_scheduler = optim.lr_scheduler.ExponentialLR(self.d_optimizer, gamma=0.95)

        self.tokenizer = None
        self.n_critic = 5
        self.gp_weight = 10.0
        self.gc_penalty_requires_grad = False
        self.noise_dim = noise_dim
        self.max_grad_norm = 1.0  # Gradient clipping
        self.temperature = 1.0  # Temperature for generation diversity
        
        # Early stopping parameters
        self.enable_early_stopping = enable_early_stopping
        self.best_d_loss = float('inf')
        self.best_g_loss = float('inf')
        self.patience = 10  # Stop if no improvement for 10 epochs
        self.wait = 0  # Counter for epochs without improvement
        self.min_delta = 0.001  # Minimum change to qualify as improvement
        
        # Training history for monitoring
        self.history = {
            'd_loss': [],
            'g_loss': [],
            'epoch': []
        }

    # -------------------------
    # Helpers
    # -------------------------
    def _to_onehot(self, x):
        if x.dim() == 2:
            B, L = x.size()
            oh = torch.zeros(B, L, self.vocab_size, device=x.device)
            oh.scatter_(2, x.unsqueeze(-1), 1.0)
            return oh
        return x

    # -------------------------
    # Training step
    # -------------------------
    def train_step(self, real_batch, cond_ids=None):
        device = self.device
        real_batch = real_batch.to(device)
        if cond_ids is None:
            cond_ids = torch.zeros(real_batch.size(0), dtype=torch.long, device=device)
        else:
            cond_ids = cond_ids.to(device)

        cond_vec = self.cond_encoder(cond_ids)
        real_onehot = self._to_onehot(real_batch).float().to(device)
        batch_size = real_onehot.size(0)

        # Train Discriminator
        d_loss_accum = 0.0
        for _ in range(self.n_critic):
            z = torch.randn(batch_size, self.noise_dim, device=device)
            with torch.no_grad():
                fake_logits = self.generator(z, cond_vec)
                fake_probs = F.softmax(fake_logits, dim=-1)
            cond_for_d = cond_vec.detach()
            real_scores = self.discriminator(real_onehot, cond_for_d)
            fake_scores = self.discriminator(fake_probs, cond_for_d)
            d_loss = wgan_critic_loss(real_scores, fake_scores)
            gp = gradient_penalty(self.discriminator, real_onehot, fake_probs, cond_for_d,
                                  device=device, gp_weight=self.gp_weight)
            d_total = d_loss + gp
            self.d_optimizer.zero_grad(set_to_none=True)
            d_total.backward()
            # Gradient clipping for discriminator
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.d_optimizer.step()
            d_loss_accum += d_total.item()

        d_loss_avg = d_loss_accum / max(1, self.n_critic)

        # Train Generator
        z = torch.randn(batch_size, self.noise_dim, device=device)
        self.g_optimizer.zero_grad(set_to_none=True)
        fake_logits = self.generator(z, cond_vec, temperature=self.temperature)
        fake_probs = F.softmax(fake_logits, dim=-1)
        fake_scores_for_g = self.discriminator(fake_probs, cond_vec)
        g_loss = wgan_generator_loss(fake_scores_for_g)

        # Optional GC penalty
        if self.tokenizer is not None and ('G' in self.tokenizer.char2idx and 'C' in self.tokenizer.char2idx):
            g_idx = self.tokenizer.char2idx['G']
            c_idx = self.tokenizer.char2idx['C']
            if self.gc_penalty_requires_grad:
                gc_prob_per_pos = fake_probs[..., g_idx] + fake_probs[..., c_idx]
                gc_ratio = gc_prob_per_pos.mean(dim=1)
                gc_loss = ((gc_ratio - self.target_gc) ** 2).mean()
                g_loss = g_loss + gc_loss
            else:
                with torch.no_grad():
                    fake_indices = torch.argmax(fake_probs, dim=-1)
                    gc_mask = (fake_indices == g_idx) | (fake_indices == c_idx)
                    gc_ratio = gc_mask.float().mean(dim=1)
                    gc_loss_val = ((gc_ratio - self.target_gc) ** 2).mean().item()
                g_loss = g_loss + gc_loss_val

        g_loss.backward()
        # Gradient clipping for generator
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)
        self.g_optimizer.step()

        return d_loss_avg, g_loss.item()
    
    def _validation_step(self, real_batch):
        """Validation step without gradient computation for efficiency."""
        device = self.device
        real_batch = real_batch.to(device)
        cond_ids = torch.zeros(real_batch.size(0), dtype=torch.long, device=device)

        cond_vec = self.cond_encoder(cond_ids)
        real_onehot = self._to_onehot(real_batch).float().to(device)
        batch_size = real_onehot.size(0)

        # Generate fake sequences
        z = torch.randn(batch_size, self.noise_dim, device=device)
        fake_logits = self.generator(z, cond_vec, temperature=self.temperature)
        fake_probs = F.softmax(fake_logits, dim=-1)
        
        # Get scores
        real_scores = self.discriminator(real_onehot, cond_vec)
        fake_scores = self.discriminator(fake_probs, cond_vec)
        
        d_loss = torch.mean(fake_scores) - torch.mean(real_scores)
        g_loss = -torch.mean(fake_scores)
        
        return d_loss.item(), g_loss.item()
    
    def update_schedulers(self):
        """Update learning rate schedulers. Call this after each epoch."""
        self.g_scheduler.step()
        self.d_scheduler.step()
    
    def should_stop_early(self, d_loss, g_loss):
        """
        Check if training should stop early based on loss improvement.
        Returns True if no improvement for 'patience' epochs.
        """
        if not self.enable_early_stopping:
            return False
        
        # Check if discriminator loss improved
        if d_loss < (self.best_d_loss - self.min_delta):
            self.best_d_loss = d_loss
            self.wait = 0
            return False
        elif g_loss < (self.best_g_loss - self.min_delta):
            self.best_g_loss = g_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
            return False
    
    def reset_early_stopping(self):
        """Reset early stopping counters (for new training run)."""
        self.best_d_loss = float('inf')
        self.best_g_loss = float('inf')
        self.wait = 0
        self.history = {'d_loss': [], 'g_loss': [], 'epoch': []}

    # -------------------------
    # Sequence Generation
    # -------------------------
    def generate(self, n_samples=10, cond_ids=None, sampling='argmax', temperature=None):
        # Auto-detect device from generator
        device = next(self.generator.parameters()).device
        self.generator.eval()

        # Create or convert cond_ids
        if cond_ids is None:
            cond_ids = torch.zeros(n_samples, dtype=torch.long, device=device)
        else:
            cond_ids = torch.tensor(cond_ids, dtype=torch.long, device=device)

        cond_vec = self.cond_encoder(cond_ids)
        z = torch.randn(n_samples, self.noise_dim, device=device)

        # Use provided temperature or default
        gen_temp = temperature if temperature is not None else self.temperature

        with torch.no_grad():
            logits = self.generator(z, cond_vec, temperature=gen_temp)
            if sampling == 'argmax':
                indices = torch.argmax(logits, dim=-1)
            elif sampling == 'sample':
                # Sample with temperature
                if gen_temp != 1.0:
                    logits = logits / gen_temp
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                indices = dist.sample()
            else:
                indices = torch.argmax(logits, dim=-1)

        sequences = [idx.tolist() for idx in indices.cpu()]
        self.generator.train()
        return sequences
