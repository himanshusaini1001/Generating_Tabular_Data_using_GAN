"""
Bagging/Ensemble implementation for StackedGAN
Creates multiple models and combines predictions for better generalization
"""
import torch
import numpy as np
import time
from train.training import StackedGAN


def _train_model_direct(gan, dataset, batch_size, epochs, device):
    """Direct training function to avoid circular imports."""
    dataset = dataset.to(device)
    gan.generator.to(device)
    gan.discriminator.to(device)
    gan.cond_encoder.to(device)
    
    n_batches = int(np.ceil(len(dataset) / batch_size))
    gan.enable_early_stopping = True
    gan.reset_early_stopping()
    
    history = {'d_loss': [], 'g_loss': [], 'epoch': []}
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        perm = torch.randperm(len(dataset))
        total_d, total_g = 0.0, 0.0
        
        for i in range(n_batches):
            idx = perm[i*batch_size : (i+1)*batch_size]
            batch = dataset[idx]
            d_loss, g_loss = gan.train_step(batch)
            total_d += d_loss
            total_g += g_loss
        
        avg_d = total_d / n_batches
        avg_g = total_g / n_batches
        
        gan.update_schedulers()
        history['d_loss'].append(avg_d)
        history['g_loss'].append(avg_g)
        history['epoch'].append(epoch)
        
        epoch_time = time.time() - epoch_start_time
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs} | D_loss={avg_d:.4f} | G_loss={avg_g:.4f} | Time={epoch_time:.2f}s")
        
        # Early stopping
        if gan.should_stop_early(avg_d, avg_g):
            print(f"  ⏹️  Early stopping at epoch {epoch}")
            break
    
    gan.generator.eval()
    gan.discriminator.eval()
    return history

class GANBagging:
    """
    Bagging ensemble for GAN models.
    Trains multiple models on different data subsets and combines predictions.
    """
    def __init__(self, seq_len, vocab_size, n_models=3, **gan_kwargs):
        """
        Args:
            seq_len: Sequence length
            vocab_size: Vocabulary size
            n_models: Number of models in ensemble
            **gan_kwargs: Additional arguments for StackedGAN
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_models = n_models
        self.models = []
        self.gan_kwargs = gan_kwargs
        
        # Initialize models
        for i in range(n_models):
            gan = StackedGAN(seq_len=seq_len, vocab_size=vocab_size, 
                           enable_early_stopping=True, **gan_kwargs)
            self.models.append(gan)
        
        print(f"✅ Initialized {n_models} GAN models for bagging")
    
    def train_bagging(self, dataset, batch_size, epochs, device, sample_ratio=0.8):
        """
        Train multiple models with bagging (different data samples per model).
        
        Args:
            dataset: Training dataset
            batch_size: Batch size
            epochs: Number of epochs
            device: Torch device
            sample_ratio: Ratio of data to use per model (0.8 = 80% sampling)
        """
        print(f"\n🎲 Training {self.n_models} models with bagging (sample_ratio={sample_ratio})")
        
        histories = []
        
        for idx, gan in enumerate(self.models, 1):
            print(f"\n{'='*60}")
            print(f"Training Model {idx}/{self.n_models}")
            print(f"{'='*60}")
            
            # Create random subset for this model (with replacement)
            sample_size = int(len(dataset) * sample_ratio)
            indices = torch.randint(0, len(dataset), (sample_size,))
            bootstrapped_dataset = dataset[indices]
            
            print(f"  📦 Using {len(bootstrapped_dataset)} samples")
            
            # Train this model directly
            history = _train_model_direct(
                gan,
                bootstrapped_dataset, 
                batch_size=batch_size, 
                epochs=epochs, 
                device=device
            )
            histories.append(history)
            
            print(f"  ✅ Model {idx} training completed")
        
        print(f"\n✅ Bagging training completed for {self.n_models} models")
        return histories
    
    def generate_ensemble(self, n_samples, cond_ids=None, method='average'):
        """
        Generate sequences using ensemble of all models.
        
        Args:
            n_samples: Number of sequences to generate
            cond_ids: Condition IDs
            method: 'average' (vote) or 'sample' (random from ensemble)
        
        Returns:
            List of generated sequences
        """
        all_sequences = []
        
        # Generate from each model
        for model in self.models:
            seqs = model.generate(n_samples=n_samples, cond_ids=cond_ids)
            all_sequences.append(seqs)
        
        # Combine predictions
        if method == 'average':
            # For each position, vote on most common token
            combined_sequences = []
            for i in range(n_samples):
                # Get tokens at position i from all models
                tokens = [seq[i] for seq in all_sequences]
                
                # Vote: for each position in sequence, pick most common
                sequence_length = len(tokens[0])
                combined_seq = []
                
                for pos in range(sequence_length):
                    # Get tokens at this position from all models
                    pos_tokens = [tok[pos] for tok in tokens]
                    # Pick most common token
                    combined_seq.append(max(set(pos_tokens), key=pos_tokens.count))
                
                combined_sequences.append(combined_seq)
            
            return combined_sequences
        
        elif method == 'sample':
            # Pick random model for each sample
            selected_sequences = []
            for i in range(n_samples):
                model_idx = np.random.randint(0, self.n_models)
                selected_sequences.append(all_sequences[model_idx][i])
            return selected_sequences
    
    def save_all(self, base_path):
        """Save all models to separate checkpoints."""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        for idx, model in enumerate(self.models):
            path = f"{base_path}/bagging_model_{idx}.pt"
            torch.save({
                'generator_state': model.generator.state_dict(),
                'discriminator_state': model.discriminator.state_dict(),
                'cond_encoder_state': model.cond_encoder.state_dict(),
                'tokenizer': model.tokenizer,
                'model_idx': idx
            }, path)
        
        print(f"✅ Saved {self.n_models} models to {base_path}/")
    
    def load_all(self, base_path):
        """Load all models from checkpoints."""
        for idx in range(self.n_models):
            path = f"{base_path}/bagging_model_{idx}.pt"
            checkpoint = torch.load(path)
            
            self.models[idx].generator.load_state_dict(checkpoint['generator_state'])
            self.models[idx].discriminator.load_state_dict(checkpoint['discriminator_state'])
            self.models[idx].cond_encoder.load_state_dict(checkpoint['cond_encoder_state'])
            self.models[idx].tokenizer = checkpoint['tokenizer']
        
        print(f"✅ Loaded {self.n_models} models from {base_path}/")

