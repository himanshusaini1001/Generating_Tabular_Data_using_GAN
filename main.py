import argparse
import torch
import numpy as np
import os
import time
from train.training import StackedGAN
from logger_utils import GANLogger  


class CharTokenizer:
    def __init__(self, sequences):
        chars = sorted(set("".join(sequences)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, seq, seq_len):
        arr = [self.char2idx.get(ch, 0) for ch in seq]
        if len(arr) < seq_len:
            arr += [0] * (seq_len - len(arr))
        return arr[:seq_len]

    def decode(self, arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.tolist()
        if isinstance(arr, int):
            arr = [arr]
        arr = [min(max(i, 0), self.vocab_size - 1) for i in arr]
        return "".join([self.idx2char.get(i, "?") for i in arr])

# -------------------------
# Dataset Loader
# -------------------------
def load_dataset(file_path, seq_len):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt", ".fasta", ".fa"]:
        raw_data = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(">"):
                    raw_data.append(line)
        tokenizer = CharTokenizer(raw_data)
        dataset = [tokenizer.encode(seq, seq_len) for seq in raw_data]
        dataset = torch.tensor(np.array(dataset), dtype=torch.long)
        return dataset, tokenizer

    else:
        raise ValueError(f"Unsupported dataset format: {ext}")

# -------------------------
# Training function with early stopping
# -------------------------
def train_gan(gan, dataset, batch_size, epochs, device, logger=None, validation_split=0.2, use_early_stopping=True):
    dataset = dataset.to(device)
    gan.generator.to(device)
    gan.discriminator.to(device)
    gan.cond_encoder.to(device)
    
    # Split into training and validation sets
    total_samples = len(dataset)
    val_size = int(total_samples * validation_split)
    train_size = total_samples - val_size
    
    indices = torch.randperm(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    
    print(f"📊 Dataset split: {train_size} train + {val_size} validation")
    
    n_batches = int(np.ceil(len(train_dataset) / batch_size))
    best_epoch = 0
    
    # Enable early stopping
    gan.enable_early_stopping = use_early_stopping
    gan.reset_early_stopping()

    for epoch in range(1, epochs + 1):
        # Record epoch start time
        epoch_start_time = time.time()
        
        # Training phase
        gan.generator.train()
        gan.discriminator.train()
        
        perm = torch.randperm(len(train_dataset))
        total_d, total_g = 0.0, 0.0

        for i in range(n_batches):
            idx = perm[i*batch_size : (i+1)*batch_size]
            batch = train_dataset[idx]
            d_loss, g_loss = gan.train_step(batch)
            total_d += d_loss
            total_g += g_loss
        
        # Validation phase
        gan.generator.eval()
        gan.discriminator.eval()
        total_val_d, total_val_g = 0.0, 0.0
        n_val_batches = int(np.ceil(len(val_dataset) / batch_size))
        
        # Validation without gradient computation (no gradient penalty)
        for i in range(n_val_batches):
            idx = i * batch_size
            val_batch = val_dataset[idx:min(idx + batch_size, len(val_dataset))]
            val_d, val_g = gan._validation_step(val_batch)
            total_val_d += val_d
            total_val_g += val_g
        
        avg_d = total_d / n_batches
        avg_g = total_g / n_batches
        avg_val_d = total_val_d / n_val_batches
        avg_val_g = total_val_g / n_val_batches
        
        # Update learning rate schedulers
        gan.update_schedulers()
        
        # Store history
        gan.history['d_loss'].append(avg_val_d)
        gan.history['g_loss'].append(avg_val_g)
        gan.history['epoch'].append(epoch)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        if logger:
            logger.log_epoch(epoch, epochs, avg_d, avg_g, epoch_time)
        else:
            current_lr_g = gan.g_optimizer.param_groups[0]['lr']
            current_lr_d = gan.d_optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{epochs} | Train D={avg_d:.4f} G={avg_g:.4f} | Val D={avg_val_d:.4f} G={avg_val_g:.4f} | Time={epoch_time:.2f}s | LR_G={current_lr_g:.6f}")
        
        # Early stopping check
        if use_early_stopping and gan.should_stop_early(avg_val_d, avg_val_g):
            print(f"\n⏹️  Early stopping triggered at epoch {epoch}!")
            print(f"   No improvement for {gan.patience} epochs")
            print(f"   Best validation loss: D={gan.best_d_loss:.4f}, G={gan.best_g_loss:.4f}")
            break
        
        # Track best epoch
        if avg_val_d < gan.best_d_loss:
            best_epoch = epoch
    
    # Set to eval mode
    gan.generator.eval()
    gan.discriminator.eval()
    
    if best_epoch > 0:
        print(f"\n✅ Best epoch: {best_epoch} with D_loss={gan.best_d_loss:.4f}")
    
    return gan.history

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','gen'], default='train')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--data', type=str, required=True, help="Path to dataset file (TXT, FASTA)")
    parser.add_argument('--target_gc', type=float, default=0.42, help="Target GC ratio (0-1)")
    parser.add_argument('--early_stopping', action='store_true', help="Enable early stopping")
    parser.add_argument('--validation_split', type=float, default=0.2, help="Validation split ratio (0-1)")
    parser.add_argument('--bagging', action='store_true', help="Use bagging ensemble training")
    parser.add_argument('--n_models', type=int, default=3, help="Number of models for bagging")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == 'train':
        dataset, tokenizer = load_dataset(args.data, args.seq_len)
        print(f"Loaded dataset: {len(dataset)} sequences")

        # Check if using bagging
        if args.bagging:
            from train.bagging import GANBagging
            print(f"\n🎲 Using BAGGING ensemble with {args.n_models} models")
            
            bagger = GANBagging(
                seq_len=args.seq_len,
                vocab_size=tokenizer.vocab_size,
                n_models=args.n_models,
                target_gc=args.target_gc,
                device=device
            )
            
            # Set tokenizer for all models
            for model in bagger.models:
                model.tokenizer = tokenizer
            
            # Initialize Logger
            logger = GANLogger(mode="train")
            logger.start()
            logger.log_params(mode=args.mode, data=args.data, epochs=args.epochs, batch=args.batch,
                              seq_len=args.seq_len, target_gc=args.target_gc, device=device, 
                              bagging=True, n_models=args.n_models)
            
            # Train with bagging
            histories = bagger.train_bagging(dataset, args.batch, args.epochs, device, sample_ratio=0.8)
            
            logger.finish("Bagging Training")
            
            # Save all models
            os.makedirs('checkpoints/bagging', exist_ok=True)
            bagger.save_all('checkpoints/bagging')
            logger.log_message(f"Saved {args.n_models} models to checkpoints/bagging/")
        
        else:
            # Regular training
            # Initialize GAN
            gan = StackedGAN(seq_len=args.seq_len, vocab_size=tokenizer.vocab_size,
                           target_gc=args.target_gc, device=device, 
                           enable_early_stopping=args.early_stopping)
            gan.tokenizer = tokenizer  # needed for GC penalty

            # Initialize Logger
            logger = GANLogger(mode="train")
            logger.start()
            logger.log_params(mode=args.mode, data=args.data, epochs=args.epochs, batch=args.batch,
                              seq_len=args.seq_len, target_gc=args.target_gc, device=device,
                              early_stopping=args.early_stopping, validation_split=args.validation_split)

            # Train with logger
            train_gan(gan, dataset, args.batch, args.epochs, device, logger, 
                     validation_split=args.validation_split, 
                     use_early_stopping=args.early_stopping)

            logger.finish("Training")

            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            save_path = f'checkpoints/stacked_epoch{args.epochs}.pt'
            torch.save({
                "generator_state": gan.generator.state_dict(),
                "discriminator_state": gan.discriminator.state_dict(),
                "cond_encoder_state": gan.cond_encoder.state_dict(),
                "tokenizer": tokenizer
            }, save_path)
            logger.log_message(f"Checkpoint saved: {save_path}")

    elif args.mode == 'gen':
        if args.checkpoint == '':
            raise ValueError("Please provide checkpoint with --checkpoint")
        print(f"Loading checkpoint from {args.checkpoint} ...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        tokenizer = checkpoint["tokenizer"]

        # Initialize GAN structure and load weights
        gan = StackedGAN(seq_len=args.seq_len, vocab_size=tokenizer.vocab_size,
                         target_gc=args.target_gc, device=device)
        gan.tokenizer = tokenizer
        gan.generator.load_state_dict(checkpoint["generator_state"])
        gan.discriminator.load_state_dict(checkpoint["discriminator_state"])
        gan.cond_encoder.load_state_dict(checkpoint["cond_encoder_state"])

        # Initialize Logger
        logger = GANLogger(mode="gen")
        logger.start()
        logger.log_params(mode=args.mode, checkpoint=args.checkpoint, n=args.n, seq_len=args.seq_len, device=device)

        # Generate sequences
        sequences = gan.generate(n_samples=args.n)
        GANLogger.save_sequences(sequences, tokenizer)

        logger.log_message(f"Generated {args.n} sequences")
        logger.finish("Generation")


if __name__ == '__main__':
    main()
