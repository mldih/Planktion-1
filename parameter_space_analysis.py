import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# Parameter space analysis
# Computes criterion window [Vf/mu, rf] for each species
# Produces: (1) text output, (2) figure
# ====================================================================

S = 30
m = 3
np.random.seed(42)

r = np.random.lognormal(np.log(0.7), 0.4, S)
K = np.random.lognormal(np.log(1.5), 0.4, (m, S))
V = np.random.lognormal(np.log(0.1), 0.2, (m, S))
mu = np.random.uniform(0.2, 0.4, (m, S))

f_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])

# ---------- Text output ----------
print("=" * 80)
print("Parameter space: criterion window per species")
print("=" * 80)

for f in f_values:
    delta_lower, delta_upper, widths = [], [], []
    for i in range(S):
        V_i = V[:, i].mean()
        mu_i = mu[:, i].mean()
        r_i = r[i]
        d_low = V_i * f / mu_i
        d_up = r_i * f
        delta_lower.append(d_low)
        delta_upper.append(d_up)
        widths.append(d_up - d_low)
    widths = np.array(widths)
    print(f"\nf = {f:.2f}")
    print(f"  delta_lower median: {np.median(delta_lower):.4f}")
    print(f"  delta_upper median: {np.median(delta_upper):.4f}")
    print(f"  width median: {np.median(widths):.4f}")
    print(f"  species with width > 0: {np.sum(widths > 0)} / {S}")

V_mean = V.mean(); mu_mean = mu.mean(); r_mean = r.mean()
print("\n" + "=" * 80)
print("Community mean window")
print("=" * 80)
print(f"{'f':>6} {'delta_lower':>12} {'delta_upper':>12} {'width':>10}")
for f in f_values:
    d_low = V_mean * f / mu_mean
    d_up = r_mean * f
    print(f"{f:>6.2f} {d_low:>12.4f} {d_up:>12.4f} {d_up-d_low:>10.4f}")

print("\n" + "=" * 80)
print("f = 0.5: per-species window")
print("=" * 80)
f = 0.5
print(f"{'sp':>4} {'V_i':>8} {'mu_i':>8} {'r_i':>8} "
      f"{'delta_low':>10} {'delta_up':>10} {'width':>8} {'mu*r>V':>8}")
valid = 0
for i in range(S):
    V_i = V[:, i].mean(); mu_i = mu[:, i].mean(); r_i = r[i]
    d_low = V_i * f / mu_i; d_up = r_i * f
    w = d_up - d_low
    if w > 0: valid += 1
    print(f"{i:>4} {V_i:>8.4f} {mu_i:>8.4f} {r_i:>8.4f} "
          f"{d_low:>10.4f} {d_up:>10.4f} {w:>8.4f} "
          f"{'yes' if mu_i*r_i > V_i else 'no':>8}")
print(f"\nSpecies with width > 0: {valid} / {S}")

# ---------- Figure ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A: community-mean window vs f
d_lows = np.array([V_mean * f / mu_mean for f in f_values])
d_ups = np.array([r_mean * f for f in f_values])

axes[0].fill_between(f_values, d_lows, d_ups, alpha=0.3, color='green',
                     label='Criterion window')
axes[0].plot(f_values, d_lows, 'b--', linewidth=2,
             label='$\\delta_{lower} = Vf/\\mu$')
axes[0].plot(f_values, d_ups, 'r--', linewidth=2,
             label='$\\delta_{upper} = rf$')
axes[0].set_xlabel('Resource saturating factor $f$')
axes[0].set_ylabel('Mortality $\\delta$ (1/d)')
axes[0].set_title('A. Community-mean criterion window')
axes[0].legend()
axes[0].grid(True)

# Panel B: per-species window at f = 0.5
f = 0.5
d_low_i = np.array([V[:, i].mean() * f / mu[:, i].mean() for i in range(S)])
d_up_i = np.array([r[i] * f for i in range(S)])
width_i = d_up_i - d_low_i

species_idx = np.arange(S)
axes[1].barh(species_idx, width_i, left=d_low_i,
             color=['green' if w > 0 else 'red' for w in width_i],
             alpha=0.7)
axes[1].set_xlabel('Mortality $\\delta$ (1/d)')
axes[1].set_ylabel('Species index')
axes[1].set_title(f'B. Per-species window at $f$ = {f}')
axes[1].axvline(x=0.177, color='k', linestyle=':', alpha=0.5)
axes[1].axvline(x=0.346, color='k', linestyle=':', alpha=0.5)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('parameter_space_analysis.png', dpi=150)
plt.close()
print("\nFigure saved: parameter_space_analysis.png")