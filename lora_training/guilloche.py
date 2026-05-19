import numpy as np
import matplotlib.pyplot as plt

def guilloche_rosette(
    R=1.0, r=0.28, d=0.75,  # klassisk spirograph parametre
    n=20000
):
    t = np.linspace(0, 2*np.pi*50, n)  
    x = (R - r) * np.cos(t) + d * np.cos(((R - r) / r) * t)
    y = (R - r) * np.sin(t) - d * np.sin(((R - r) / r) * t)
    return x, y

# Generer
x, y = guilloche_rosette(R=1.0, r=0.31, d=0.82, n=30000)

# Tegn “linjeart”
fig = plt.figure(figsize=(8, 8), dpi=300)
ax = plt.gca()
ax.plot(x, y, linewidth=0.35)
ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout(pad=0)
plt.savefig("guilloche_lineart.png", transparent=True)
plt.close(fig)

# Tegn tykkere variant
fig = plt.figure(figsize=(8, 8), dpi=300)
ax = plt.gca()
ax.plot(x, y, linewidth=0.9)
ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout(pad=0)
plt.savefig("guilloche_bold.png", transparent=True)
plt.close(fig)

print("Saved: guilloche_lineart.png, guilloche_bold.png")