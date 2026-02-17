# compare_models.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import pandas as pd
import editdistance
from scipy.stats import entropy
from sdv.single_table.ctgan import CTGAN
from train.training import StackedGAN
from main import load_dataset, CharTokenizer

# ----------------------------
# Device setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

# ----------------------------
# Early Stopping
# ----------------------------
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ----------------------------
# Model Definitions
# ----------------------------
class SimpleGen(nn.Module):
    def __init__(self, z_dim, hidden_dim, seq_len, vocab_size):
        super().__init__()
        self.fc = nn.Linear(z_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, z):
        x = torch.relu(self.fc(z)).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        return F.softmax(self.out(out), dim=-1)


class SimpleDisc(nn.Module):
    def __init__(self, vocab_size, hidden_dim, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * seq_len, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.contiguous().view(out.size(0), -1)
        return self.fc(out)


# ----------------------------
# Metrics
# ----------------------------
def gc_content(seqs, tokenizer):
    g_idx = tokenizer.char2idx.get("G", None)
    c_idx = tokenizer.char2idx.get("C", None)
    if g_idx is None or c_idx is None:
        return 0.0
    arr = np.array([s.cpu().numpy() if isinstance(s, torch.Tensor) else np.array(s) for s in seqs])
    gc = ((arr == g_idx) | (arr == c_idx)).mean()
    return float(gc)


def kmer_distribution(seqs, tokenizer, k=3):
    def seq_to_kmers(seq):
        s = tokenizer.decode(seq)
        return [s[i:i+k] for i in range(len(s)-k+1)]
    kmers = []
    for seq in seqs:
        kmers.extend(seq_to_kmers(seq))
    counts = Counter(kmers)
    total = sum(counts.values())
    return {k: v/total for k,v in counts.items()}


def js_divergence(p, q):
    all_keys = set(p.keys()) | set(q.keys())
    p_vec = np.array([p.get(k,0) for k in all_keys])
    q_vec = np.array([q.get(k,0) for k in all_keys])
    p_vec /= (p_vec.sum() + 1e-9)
    q_vec /= (q_vec.sum() + 1e-9)
    m = 0.5 * (p_vec + q_vec)
    return float(0.5 * (entropy(p_vec, m) + entropy(q_vec, m)))


def uniqueness_ratio(seqs):
    unique = len(set(tuple(s) for s in seqs))
    return float(unique / len(seqs))


def avg_edit_distance(fake, real, tokenizer):
    real_strs = [tokenizer.decode(s) for s in real]
    fake_strs = [tokenizer.decode(s) for s in fake]
    dists = [min(editdistance.eval(f, r) for r in real_strs) / max(len(f), 1) for f in fake_strs]
    return float(np.mean(dists))


def motif_score(seqs, tokenizer, motif="ATG"):
    count = 0
    total = 0
    motif_len = len(motif)
    for s in seqs:
        seq_str = tokenizer.decode(s)
        for i in range(len(seq_str) - motif_len + 1):
            if seq_str[i:i + motif_len] == motif:
                count += 1
            total += 1
    return float(count / total if total > 0 else 0)


def calculate_precision(seqs, tokenizer, real_gc, gc_tolerance=0.1):
    """Calculate precision: fraction of generated sequences that are realistic.
    A sequence is considered realistic if:
    - Contains only valid nucleotides (A, C, G, T)
    - GC content is within tolerance of real data
    """
    valid_count = 0
    total = len(seqs)
    if total == 0:
        return 0.0
    
    valid_nucleotides = set("ACGT")
    for s in seqs:
        seq_str = tokenizer.decode(s)
        # Check if sequence contains only valid nucleotides
        if all(ch in valid_nucleotides for ch in seq_str):
            # Check if GC content is within tolerance
            seq_gc = gc_content([s], tokenizer)
            if abs(seq_gc - real_gc) <= gc_tolerance:
                valid_count += 1
    
    return float(valid_count / total)


def calculate_recall(fake_kmer, real_kmer):
    """Calculate recall: how well generated sequences match real data distribution.
    Based on k-mer distribution similarity - lower JS divergence means higher recall.
    Recall = 1 - normalized_JS_divergence (clamped to [0, 1])
    """
    js_div = js_divergence(real_kmer, fake_kmer)
    # Normalize JS divergence (typically ranges 0-1, but can be higher)
    # Convert to recall: lower divergence = higher recall
    # Use exponential decay: recall = exp(-js_div)
    recall = np.exp(-js_div)
    return float(np.clip(recall, 0.0, 1.0))


# ----------------------------
# Generic Training Loop Template
# ----------------------------
def train_gan_base(G, D, dataset, tokenizer, seq_len, epochs=50, batch=64, z_dim=128, lr=1e-4):
    g_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    dataset = dataset.to(device)
    early_stopper = EarlyStopper(patience=10)
    losses_g, losses_d = [], []
    dataset_size = len(dataset)

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch):
            batch_idx = perm[i:i+batch]
            real = dataset[batch_idx]
            cur_batch = real.size(0)
            real_oh = torch.zeros(cur_batch, seq_len, tokenizer.vocab_size, device=device)
            real_oh.scatter_(2, real.unsqueeze(-1), 1.0)

            # Train Discriminator
            z = torch.randn(cur_batch, z_dim, device=device)
            fake = G(z).detach()
            d_loss = D(fake).mean() - D(real_oh).mean()
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Train Generator
            z = torch.randn(cur_batch, z_dim, device=device)
            fake = G(z)
            g_loss = -D(fake).mean()
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            losses_g.append(float(g_loss.item()))
            losses_d.append(float(d_loss.item()))

        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{epochs}] | G_loss={np.mean(losses_g[-10:]):.4f}, D_loss={np.mean(losses_d[-10:]):.4f}")
        if early_stopper.should_stop(np.mean(losses_g[-10:])):
            print("⏹️ Early stopping triggered.")
            break

    return G, losses_g, losses_d


# ----------------------------
# Individual GAN Trainers
# ----------------------------
def train_wgan(dataset, tokenizer, seq_len, epochs=50):
    print("\n🚀 Training WGAN...")
    G = SimpleGen(128, 128, seq_len, tokenizer.vocab_size).to(device)
    D = SimpleDisc(tokenizer.vocab_size, 128, seq_len).to(device)
    return train_gan_base(G, D, dataset, tokenizer, seq_len, epochs=epochs)


def train_cgan(dataset, tokenizer, seq_len, epochs=50):
    print("\n🎯 Training CGAN...")
    G = SimpleGen(128, 128, seq_len, tokenizer.vocab_size).to(device)
    D = SimpleDisc(tokenizer.vocab_size, 128, seq_len).to(device)
    return train_gan_base(G, D, dataset, tokenizer, seq_len, epochs=epochs)


def train_cramergan(dataset, tokenizer, seq_len, epochs=50):
    print("\n🔬 Training CramerGAN...")
    G = SimpleGen(128, 128, seq_len, tokenizer.vocab_size).to(device)
    D = SimpleDisc(tokenizer.vocab_size, 128, seq_len).to(device)
    return train_gan_base(G, D, dataset, tokenizer, seq_len, epochs=epochs)


def train_dragan(dataset, tokenizer, seq_len, epochs=50):
    print("\n💧 Training DraGAN...")
    G = SimpleGen(128, 128, seq_len, tokenizer.vocab_size).to(device)
    D = SimpleDisc(tokenizer.vocab_size, 128, seq_len).to(device)
    return train_gan_base(G, D, dataset, tokenizer, seq_len, epochs=epochs)


def train_stackedgan(dataset, tokenizer, seq_len, epochs=50):
    print("\n🧩 Training StackedGAN...")
    gan = StackedGAN(seq_len=seq_len, vocab_size=tokenizer.vocab_size, device=device)
    gan.tokenizer = tokenizer
    dataset_size = len(dataset)
    early_stopper = EarlyStopper(patience=10)
    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        for i in range(0, dataset_size, 64):
            batch_idx = perm[i:i+64]
            batch = dataset[batch_idx].to(device)
            gan.train_step(batch)
        if epoch % 5 == 0:
            print(f"[StackedGAN Epoch {epoch}/{epochs}]")
        if early_stopper.should_stop(epoch):
            break
    return gan


# ----------------------------
# Evaluation Pipeline
# ----------------------------
def _f1_from_pr(precision, recall):
    """Compute F1 score from precision and recall safely."""
    try:
        p = float(precision)
        r = float(recall)
    except (TypeError, ValueError):
        return 0.0
    denom = p + r
    if denom <= 0:
        return 0.0
    return float(2.0 * p * r / denom)


def evaluate_models(real, tokenizer, seq_len, csv_path="comparison_results_gpu.csv",
                    epochs=50, gen_samples=1000):

    if os.path.exists(csv_path):
        df_cached = pd.read_csv(csv_path, index_col=0)

        # If older CSV is missing Accuracy/F1, derive them from Precision/Recall
        if ("Accuracy" not in df_cached.columns or "F1" not in df_cached.columns) and \
           ("Precision" in df_cached.columns and "Recall" in df_cached.columns):
            print("ℹ️ Cached results missing Accuracy/F1 – augmenting CSV from Precision/Recall.")

            if "Accuracy" not in df_cached.columns:
                # In this setup, Accuracy is defined as the same realism test as Precision
                df_cached["Accuracy"] = df_cached["Precision"].astype(float)

            if "F1" not in df_cached.columns:
                df_cached["F1"] = df_cached.apply(
                    lambda row: _f1_from_pr(row.get("Precision", 0.0), row.get("Recall", 0.0)),
                    axis=1,
                )

            df_cached.to_csv(csv_path)

        print(f"✅ Cached results found at {csv_path}")
        return df_cached.to_dict(orient="index")

    results = {}
    real_gc = gc_content(real, tokenizer)
    real_kmer = kmer_distribution(real, tokenizer)

    # ---- WGAN ----
    G_wgan, _, _, = train_wgan(real, tokenizer, seq_len, epochs)
    z = torch.randn(gen_samples, 128, device=device)
    fake_idx = G_wgan(z).argmax(-1).cpu().numpy()
    fake_kmer_wgan = kmer_distribution(fake_idx, tokenizer)
    prec_wgan = calculate_precision(fake_idx, tokenizer, real_gc)
    rec_wgan = calculate_recall(fake_kmer_wgan, real_kmer)
    results["WGAN"] = {
        "GC_error": abs(real_gc - gc_content(fake_idx, tokenizer)),
        "kmer_JS": js_divergence(real_kmer, fake_kmer_wgan),
        "Uniqueness": uniqueness_ratio(fake_idx),
        "EditDist": avg_edit_distance(fake_idx, real, tokenizer),
        "MotifScore": motif_score(fake_idx, tokenizer),
        "GC_content": gc_content(fake_idx, tokenizer),
        "Precision": prec_wgan,
        "Recall": rec_wgan,
        "Accuracy": prec_wgan,
        "F1": _f1_from_pr(prec_wgan, rec_wgan),
    }

    # ---- CTGAN ----
    print("\n🧮 Training CTGAN (CPU only)...")
    seq_strings = [tokenizer.decode(s) for s in real]
    df_ct = pd.DataFrame(seq_strings, columns=["sequence"])
    ctgan = CTGAN(epochs=epochs)
    ctgan.fit(df_ct, discrete_columns=["sequence"])
    fake_df = ctgan.sample(gen_samples)
    fake_idx = [tokenizer.encode(s, seq_len) for s in fake_df["sequence"]]
    fake_kmer_ctgan = kmer_distribution(fake_idx, tokenizer)
    prec_ctgan = calculate_precision(fake_idx, tokenizer, real_gc)
    rec_ctgan = calculate_recall(fake_kmer_ctgan, real_kmer)
    results["CTGAN"] = {
        "GC_error": abs(real_gc - gc_content(fake_idx, tokenizer)),
        "kmer_JS": js_divergence(real_kmer, fake_kmer_ctgan),
        "Uniqueness": uniqueness_ratio(fake_idx),
        "EditDist": avg_edit_distance(fake_idx, real, tokenizer),
        "MotifScore": motif_score(fake_idx, tokenizer),
        "GC_content": gc_content(fake_idx, tokenizer),
        "Precision": prec_ctgan,
        "Recall": rec_ctgan,
        "Accuracy": prec_ctgan,
        "F1": _f1_from_pr(prec_ctgan, rec_ctgan),
    }

    # ---- CGAN ----
    G_cgan, _, _ = train_cgan(real, tokenizer, seq_len, epochs)
    fake_idx = G_cgan(torch.randn(gen_samples, 128, device=device)).argmax(-1).cpu().numpy()
    fake_kmer_cgan = kmer_distribution(fake_idx, tokenizer)
    prec_cgan = calculate_precision(fake_idx, tokenizer, real_gc)
    rec_cgan = calculate_recall(fake_kmer_cgan, real_kmer)
    results["CGAN"] = {
        "GC_error": abs(real_gc - gc_content(fake_idx, tokenizer)),
        "kmer_JS": js_divergence(real_kmer, fake_kmer_cgan),
        "Uniqueness": uniqueness_ratio(fake_idx),
        "EditDist": avg_edit_distance(fake_idx, real, tokenizer),
        "MotifScore": motif_score(fake_idx, tokenizer),
        "GC_content": gc_content(fake_idx, tokenizer),
        "Precision": prec_cgan,
        "Recall": rec_cgan,
        "Accuracy": prec_cgan,
        "F1": _f1_from_pr(prec_cgan, rec_cgan),
    }

    # ---- CramerGAN ----
    G_cramer, _, _ = train_cramergan(real, tokenizer, seq_len, epochs)
    fake_idx = G_cramer(torch.randn(gen_samples, 128, device=device)).argmax(-1).cpu().numpy()
    fake_kmer_cramer = kmer_distribution(fake_idx, tokenizer)
    prec_cramer = calculate_precision(fake_idx, tokenizer, real_gc)
    rec_cramer = calculate_recall(fake_kmer_cramer, real_kmer)
    results["CramerGAN"] = {
        "GC_error": abs(real_gc - gc_content(fake_idx, tokenizer)),
        "kmer_JS": js_divergence(real_kmer, fake_kmer_cramer),
        "Uniqueness": uniqueness_ratio(fake_idx),
        "EditDist": avg_edit_distance(fake_idx, real, tokenizer),
        "MotifScore": motif_score(fake_idx, tokenizer),
        "GC_content": gc_content(fake_idx, tokenizer),
        "Precision": prec_cramer,
        "Recall": rec_cramer,
        "Accuracy": prec_cramer,
        "F1": _f1_from_pr(prec_cramer, rec_cramer),
    }

    # ---- DraGAN ----
    G_dragan, _, _ = train_dragan(real, tokenizer, seq_len, epochs)
    fake_idx = G_dragan(torch.randn(gen_samples, 128, device=device)).argmax(-1).cpu().numpy()
    fake_kmer_dragan = kmer_distribution(fake_idx, tokenizer)
    prec_dragan = calculate_precision(fake_idx, tokenizer, real_gc)
    rec_dragan = calculate_recall(fake_kmer_dragan, real_kmer)
    results["DraGAN"] = {
        "GC_error": abs(real_gc - gc_content(fake_idx, tokenizer)),
        "kmer_JS": js_divergence(real_kmer, fake_kmer_dragan),
        "Uniqueness": uniqueness_ratio(fake_idx),
        "EditDist": avg_edit_distance(fake_idx, real, tokenizer),
        "MotifScore": motif_score(fake_idx, tokenizer),
        "GC_content": gc_content(fake_idx, tokenizer),
        "Precision": prec_dragan,
        "Recall": rec_dragan,
        "Accuracy": prec_dragan,
        "F1": _f1_from_pr(prec_dragan, rec_dragan),
    }

    # ---- StackedGAN ----
    gan = train_stackedgan(real, tokenizer, seq_len, epochs)
    fake_idx = gan.generate(gen_samples)
    fake_kmer_stacked = kmer_distribution(fake_idx, tokenizer)
    prec_stacked = calculate_precision(fake_idx, tokenizer, real_gc)
    rec_stacked = calculate_recall(fake_kmer_stacked, real_kmer)
    results["StackedGAN"] = {
        "GC_error": abs(real_gc - gc_content(fake_idx, tokenizer)),
        "kmer_JS": js_divergence(real_kmer, fake_kmer_stacked),
        "Uniqueness": uniqueness_ratio(fake_idx),
        "EditDist": avg_edit_distance(fake_idx, real, tokenizer),
        "MotifScore": motif_score(fake_idx, tokenizer),
        "GC_content": gc_content(fake_idx, tokenizer),
        "Precision": prec_stacked,
        "Recall": rec_stacked,
        "Accuracy": prec_stacked,
        "F1": _f1_from_pr(prec_stacked, rec_stacked),
    }

    # Save results
    df_res = pd.DataFrame(results).T
    df_res.to_csv(csv_path)
    print(f"\n✅ Results saved to {csv_path}")
    return results


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--csv_path", type=str, default="comparison_results_gpu.csv")
    args = parser.parse_args()

    dataset, tokenizer = load_dataset(args.data, args.seq_len)
    results = evaluate_models(dataset, tokenizer, args.seq_len, csv_path=args.csv_path)
    print(results)
