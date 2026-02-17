import os
import time
import torch
import matplotlib.pyplot as plt

try:
    import psutil  # optional; used for CPU/RAM stats
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

class GANLogger:
    def __init__(self, mode="train"):
        """
        mode: 'train' or 'gen'
        """
        self.mode = mode
        self.start_time = None
        self.epochs = []
        self.d_losses = []
        self.g_losses = []
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"{mode}_log.txt")

    def start(self):
        self.start_time = time.time()
        self._log(f"{self.mode.upper()} started at {time.ctime()}")

    def finish(self, name="Process"):
        elapsed = time.time() - self.start_time
        self._log(f"{name} finished at {time.ctime()} | Elapsed time: {elapsed:.2f} s")
        if self.mode == "train":
            self._plot_losses()

    def _log(self, msg):
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

    def _system_stats(self):
        """Collect lightweight system stats (CPU/RAM and optional GPU)."""
        stats = {}
        if psutil:
            stats["cpu_pct"] = psutil.cpu_percent(interval=None)
            stats["ram_pct"] = psutil.virtual_memory().percent
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            stats["gpu_mem_alloc_gb"] = torch.cuda.memory_allocated(device) / 1e9
            stats["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
        return stats

    def _format_stats(self, stats):
        parts = []
        if "cpu_pct" in stats:
            parts.append(f"CPU={stats['cpu_pct']:.1f}%")
        if "ram_pct" in stats:
            parts.append(f"RAM={stats['ram_pct']:.1f}%")
        if "gpu_mem_alloc_gb" in stats:
            parts.append(f"GPU alloc={stats['gpu_mem_alloc_gb']:.2f}GB")
        if "gpu_mem_reserved_gb" in stats:
            parts.append(f"GPU resv={stats['gpu_mem_reserved_gb']:.2f}GB")
        return " | ".join(parts) if parts else ""

    def log_epoch(self, epoch, total_epochs, d_loss, g_loss, epoch_time=None):
        self.epochs.append(epoch)
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{ts}] Epoch {epoch}/{total_epochs} | D_loss={d_loss:.4f} | G_loss={g_loss:.4f}"
        if epoch_time:
            msg += f" | Time={epoch_time:.2f}s"
        stats = self._format_stats(self._system_stats())
        if stats:
            msg += f" | {stats}"
        self._log(msg)

    def log_message(self, msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        stats = self._format_stats(self._system_stats())
        suffix = f" | {stats}" if stats else ""
        self._log(f"[{ts}] {msg}{suffix}")

    def log_params(self, **kwargs):
        self._log("Parameters:")
        for k, v in kwargs.items():
            self._log(f"  {k}: {v}")

    def _plot_losses(self):
        if len(self.epochs) == 0:
            return
        plt.figure(figsize=(8,5))
        plt.plot(self.epochs, self.d_losses, label="D_loss")
        plt.plot(self.epochs, self.g_losses, label="G_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("GAN Training Losses")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.log_dir, "loss_plot.png")
        plt.savefig(plot_path)
        plt.close()
        self._log(f"Loss plot saved to {plot_path}")

    @staticmethod
    def save_sequences(sequences, tokenizer, out_file="outputs/generated.txt"):
        os.makedirs("outputs", exist_ok=True)
        with open(out_file, "w") as f:
            for s in sequences:
                if isinstance(s, torch.Tensor):
                    s = s.tolist()
                elif isinstance(s, int):
                    s = [s]
                decoded = tokenizer.decode(s)
                f.write(decoded + "\n")
