import torch
import numpy as np

# Load the new checkpoint
ckpt = torch.load(r"D:\Projects\Major Project\Deepfake Detection\models\v5\bilstm_v2_best.pth",
                  map_location="cpu", weights_only=False)

print("Config stored in checkpoint:")
for k, v in ckpt["config"].items():
    print(f"  {k}: {v}")

print(f"\nEpoch: {ckpt['epoch']}")
print(f"Val AUC: {ckpt['val_auc']:.4f}")

# Load the saved clips to check label distribution and score direction
val_data = np.load(r"D:\1_Downloads\val_clips.npz")
val_seqs  = val_data["seqs"]
val_lbls  = val_data["labels"]

print(f"\nVal set — real: {(val_lbls==0).sum()}  fake: {(val_lbls==1).sum()}")

# Run the model on a small sample of real and fake clips
import torch.nn as nn

class OptFlowBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x      = self.norm(x)
        out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn(out), dim=1)
        ctx    = (out * attn_w).sum(dim=1)
        return self.fc(self.drop(ctx)).squeeze(1)

model = OptFlowBiLSTM(12, 64, 2, 0.3)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

norm_mean = ckpt["norm_mean"]
norm_std  = ckpt["norm_std"]

# Sample 20 real and 20 fake clips
real_idx = np.where(val_lbls == 0)[0][:20]
fake_idx = np.where(val_lbls == 1)[0][:20]

def score_clips(indices):
    seqs = torch.tensor(val_seqs[indices], dtype=torch.float32)
    seqs = (seqs - norm_mean) / norm_std
    seqs = torch.clamp(seqs, -10.0, 10.0)
    with torch.no_grad():
        logits = model(seqs)
        probs  = torch.sigmoid(logits).numpy()
    return probs

real_probs = score_clips(real_idx)
fake_probs = score_clips(fake_idx)

print(f"\nReal clips — sigmoid avg: {real_probs.mean():.4f}  "
      f"min: {real_probs.min():.4f}  max: {real_probs.max():.4f}")
print(f"Fake clips — sigmoid avg: {fake_probs.mean():.4f}  "
      f"min: {fake_probs.min():.4f}  max: {fake_probs.max():.4f}")
print(f"\nGap (fake - real): {fake_probs.mean() - real_probs.mean():+.4f}")

if fake_probs.mean() > real_probs.mean():
    print("\n✅ CORRECT — sigmoid gives Fake=HIGH, Real=LOW")
    print("   Use: prob = torch.sigmoid(logit).item()   (no inversion)")
else:
    print("\n❌ FLIPPED — sigmoid gives Fake=LOW, Real=HIGH")
    print("   Use: prob = 1.0 - torch.sigmoid(logit).item()")