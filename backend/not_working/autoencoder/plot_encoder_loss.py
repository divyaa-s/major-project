import matplotlib.pyplot as plt

# Hardcoded epoch-wise losses
losses = [
    0.001338,
    0.000309,
    0.000213,
    0.000161,
    0.000138,
    0.000120,
    0.000107,
    0.000099,
    0.000094,
    0.000084
]

epochs = range(1, len(losses) + 1)

plt.figure()
plt.plot(epochs, losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss (MSE)")
plt.title("Autoencoder Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.show()
