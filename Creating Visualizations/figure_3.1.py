import matplotlib.pyplot as plt

def zigzag_indices(n=8):
    return sorted(((x, y) for x in range(n) for y in range(n)),
                  key=lambda s: (s[0] + s[1], -s[1] if (s[0] + s[1]) % 2 else s[1]))

zigzag = zigzag_indices()
zigzag_15 = zigzag[:15]

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.set_xticks(range(9))
ax.set_yticks(range(9))
ax.grid(True)
ax.invert_yaxis()

for i, (x, y) in enumerate(zigzag_15):
    ax.text(y + 0.5, x + 0.5, str(i + 1), ha='center', va='center',
            fontsize=10, color='black', weight='bold')

    if i < len(zigzag_15) - 1:
        next_x, next_y = zigzag_15[i + 1]
        ax.annotate("",
                    xy=(next_y + 0.5, next_x + 0.5),
                    xytext=(y + 0.5, x + 0.5),
                    arrowprops=dict(arrowstyle="->", color='blue', lw=1.5))

ax.set_title("Zigzag Pattern (First 15 DCT Coefficients)")
plt.tight_layout()
plt.show()
