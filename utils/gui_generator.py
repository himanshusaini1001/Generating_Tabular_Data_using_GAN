import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
import os

# -------------------------
# Import GAN class
# -------------------------
from train.training import StackedGAN  # adjust path if needed
from train.bagging import GANBagging  # for bagging support

# -------------------------
# CharTokenizer
# -------------------------
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
        return "".join([self.idx2char.get(i, "?") for i in arr])

# -------------------------
# GUI functions
# -------------------------
gan = None
bagging = None
tokenizer = None
use_bagging = False

CHECKPOINT_PATH = "checkpoints/stacked_epoch100.pt"  # default
BAGGING_PATH = "checkpoints/bagging"

def load_checkpoint_default():
    global gan, bagging, tokenizer, use_bagging
    
    # Check for bagging first
    if os.path.exists(BAGGING_PATH) and len(os.listdir(BAGGING_PATH)) > 0:
        try:
            bagging_files = [f for f in os.listdir(BAGGING_PATH) if f.startswith('bagging_model_')]
            if len(bagging_files) > 0:
                n_models = len(bagging_files)
                
                # Load first model to get tokenizer
                first_checkpoint = torch.load(os.path.join(BAGGING_PATH, bagging_files[0]), weights_only=False)
                tokenizer = first_checkpoint["tokenizer"]
                
                # Create bagging ensemble
                bagging = GANBagging(seq_len=70, vocab_size=tokenizer.vocab_size, n_models=n_models, target_gc=0.42)
                bagging.load_all(BAGGING_PATH)
                
                # Set tokenizer for all models
                for model in bagging.models:
                    model.tokenizer = tokenizer
                    # Add missing attributes for old checkpoints
                    if not hasattr(model, 'temperature'):
                        model.temperature = 1.0
                    if not hasattr(model, 'history'):
                        model.history = {'d_loss': [], 'g_loss': [], 'epoch': []}
                
                use_bagging = True
                messagebox.showinfo("Loaded", f"🎲 BAGGING ensemble loaded with {n_models} models!")
                return
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not load bagging models:\n{e}\n\nTrying single model...")
    
    # Fallback to single model
    if not os.path.exists(CHECKPOINT_PATH):
        messagebox.showerror("Error", f"No checkpoint found at:\n{CHECKPOINT_PATH}\n\nAvailable checkpoints:\n" + 
                           "\n".join(os.listdir("checkpoints")[:5]))
        return
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
        
        # Try loading as full GAN object
        if "gan" in checkpoint:
            gan = checkpoint["gan"]
            tokenizer = checkpoint["tokenizer"]
            
            # Add missing attributes for old checkpoints
            if not hasattr(gan, 'temperature'):
                gan.temperature = 1.0
            if not hasattr(gan, 'history'):
                gan.history = {'d_loss': [], 'g_loss': [], 'epoch': []}
            
            messagebox.showinfo("Loaded", f"✅ Checkpoint loaded from {CHECKPOINT_PATH}")
        elif "tokenizer" in checkpoint:
            # Load state dicts manually
            tokenizer = checkpoint["tokenizer"]
            gan = StackedGAN(seq_len=70, vocab_size=tokenizer.vocab_size, target_gc=0.42)
            gan.tokenizer = tokenizer
            
            # Try loading weights
            try:
                gan.generator.load_state_dict(checkpoint.get("generator_state", {}), strict=False)
                gan.discriminator.load_state_dict(checkpoint.get("discriminator_state", {}), strict=False)
                gan.cond_encoder.load_state_dict(checkpoint.get("cond_encoder_state", {}))
                messagebox.showinfo("Loaded", f"✅ Checkpoint loaded from {CHECKPOINT_PATH}")
            except Exception as e:
                messagebox.showwarning("Warning", f"Could not load weights:\n{e}")
                messagebox.showinfo("Loaded", f"⚠️ Using random initialization")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load checkpoint:\n{e}")

def generate_and_display():
    if tokenizer is None:
        messagebox.showwarning("Warning", "Checkpoint not loaded!")
        return
    
    if use_bagging and bagging is None:
        messagebox.showwarning("Warning", "Bagging ensemble not loaded!")
        return
    
    if not use_bagging and gan is None:
        messagebox.showwarning("Warning", "Model not loaded!")
        return
    
    try:
        n = int(num_sequences_entry.get())
        
        # Use bagging if available, otherwise use single model
        if use_bagging and bagging is not None:
            sequences_int = bagging.generate_ensemble(n_samples=n, method='average')
            model_info = f"({len(bagging.models)} models)"
        else:
            sequences_int = gan.generate(n_samples=n)
            model_info = "(single model)"
        
        sequences_str = [tokenizer.decode(seq) for seq in sequences_int]

        text_box.config(state=tk.NORMAL)
        text_box.delete("1.0", tk.END)
        for i, seq in enumerate(sequences_str, 1):
            text_box.insert(tk.END, f">Seq{i}\n{seq}\n")
        text_box.config(state=tk.DISABLED)

        # GC content
        gc_count = sum(seq.count("G") + seq.count("C") for seq in sequences_str)
        total_bases = sum(len(seq) for seq in sequences_str)
        gc_content = (gc_count / total_bases * 100) if total_bases > 0 else 0
        gc_label.config(text=f"GC Content: {gc_content:.2f}% {model_info}")
        progress_bar['value'] = gc_content
    except Exception as e:
        messagebox.showerror("Error", str(e))

def save_fasta():
    sequences_text = text_box.get("1.0", tk.END).strip()
    if not sequences_text:
        messagebox.showwarning("Warning", "No sequences to save!")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".fasta",
                                             filetypes=[("FASTA files","*.fasta")])
    if file_path:
        with open(file_path, "w") as f:
            f.write(sequences_text)
        messagebox.showinfo("Saved", f"Sequences saved to {file_path}")

# -------------------------
# Tkinter GUI
# -------------------------
root = tk.Tk()
root.title("DNA Sequence Generator")
root.geometry("1000x700")
root.configure(bg="#eef1f5")

# Configure ttk theme and styles
style = ttk.Style()
try:
    # Prefer a modern native theme on Windows
    if "vista" in style.theme_names():
        style.theme_use("vista")
    else:
        style.theme_use("clam")
except Exception:
    pass

style.configure("TButton", padding=6, font=("Segoe UI", 11, "bold"))
style.configure("Primary.TButton", foreground="#000000", background="#2563eb")
style.map("Primary.TButton", background=[("active", "#1d4ed8")])
style.configure("Success.TButton", foreground="#000000", background="#16a34a")
style.map("Success.TButton", background=[("active", "#15803d")])
style.configure("Card.TFrame", background="#ffffff")
style.configure("TLabel", background="#eef1f5", font=("Segoe UI", 11))

# Top container to center content
container = tk.Frame(root, bg="#eef1f5")
container.pack(fill=tk.BOTH, expand=True)

# Header section
header_frame = tk.Frame(container, bg="#eef1f5")
header_frame.pack(pady=18)

title = tk.Label(header_frame, text="DNA Sequence Generator", font=("Segoe UI", 22, "bold"), bg="#eef1f5", fg="#0f172a")
title.pack()
subtitle = tk.Label(header_frame, text="Generate synthetic DNA sequences and monitor GC% in real-time", font=("Segoe UI", 11), bg="#eef1f5", fg="#475569")
subtitle.pack(pady=(6, 0))

# Card: Controls
card_controls = ttk.Frame(container, style="Card.TFrame")
card_controls.pack(padx=24, pady=10, fill=tk.X)

controls_inner = tk.Frame(card_controls, bg="#ffffff")
controls_inner.pack(padx=16, pady=12, fill=tk.X)

left_controls = tk.Frame(controls_inner, bg="#ffffff")
left_controls.pack(side=tk.LEFT)

lbl_sequences = tk.Label(left_controls, text="Number of sequences", font=("Segoe UI", 11), bg="#ffffff")
lbl_sequences.pack(side=tk.LEFT, padx=(0, 8))

# Use Spinbox for better numeric UX
num_sequences_entry = ttk.Spinbox(left_controls, from_=1, to=100000, width=10, font=("Segoe UI", 11))
num_sequences_entry.pack(side=tk.LEFT)
num_sequences_entry.set("100")

right_controls = tk.Frame(controls_inner, bg="#ffffff")
right_controls.pack(side=tk.RIGHT)

generate_btn = ttk.Button(right_controls, text="Generate", style="Success.TButton", command=generate_and_display)
generate_btn.pack(side=tk.LEFT, padx=8)

download_btn = ttk.Button(right_controls, text="Download as FASTA", style="Primary.TButton", command=save_fasta)
download_btn.pack(side=tk.LEFT, padx=8)

# Card: GC meter
card_gc = ttk.Frame(container, style="Card.TFrame")
card_gc.pack(padx=24, pady=6, fill=tk.X)

gc_inner = tk.Frame(card_gc, bg="#ffffff")
gc_inner.pack(padx=16, pady=12, fill=tk.X)

gc_label = tk.Label(gc_inner, text="GC Content: 0%", font=("Segoe UI", 12, "bold"), bg="#ffffff")
gc_label.pack(side=tk.LEFT, padx=5)

progress_bar = ttk.Progressbar(gc_inner, orient="horizontal", length=350, mode="determinate")
progress_bar.pack(side=tk.LEFT, padx=10)
progress_bar['maximum'] = 100

# Card: Text output with scrollbars
card_text = ttk.Frame(container, style="Card.TFrame")
card_text.pack(expand=True, fill=tk.BOTH, padx=24, pady=12)

text_inner = tk.Frame(card_text, bg="#ffffff")
text_inner.pack(expand=True, fill=tk.BOTH, padx=12, pady=12)

y_scroll = tk.Scrollbar(text_inner, orient=tk.VERTICAL)
y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

x_scroll = tk.Scrollbar(text_inner, orient=tk.HORIZONTAL)
x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

text_box = tk.Text(text_inner, wrap=tk.NONE, yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set, font=("Consolas", 11))
text_box.pack(expand=True, fill=tk.BOTH)
text_box.config(state=tk.DISABLED)

y_scroll.config(command=text_box.yview)
x_scroll.config(command=text_box.xview)

# Status bar
status_var = tk.StringVar(value="Ready")
status_bar = tk.Label(root, textvariable=status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#e5e7eb", fg="#334155", font=("Segoe UI", 10))
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

def set_status(message):
    status_var.set(message)

# Keyboard shortcuts
root.bind_all("<Control-g>", lambda e: generate_and_display())
root.bind_all("<Control-s>", lambda e: save_fasta())

# Load checkpoint automatically
set_status("Loading checkpoint…")
load_checkpoint_default()
set_status("Loaded. Generate sequences or save as FASTA.")

root.mainloop()
