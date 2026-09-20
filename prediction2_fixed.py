import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ====================================================================
# Prediction 2: Remineralization rate vs diversity (fixed)
# Changes:
#   1. Symbol c -> V
#   2. Removed 0.5 * recycle reduction
#   3. Added OFF control
#   4. Scan range [0.0, 2.0]
#   5. Mark rho_D = 1 reference line
# ====================================================================

S = 30; m = 3
np.random.seed(42)
n_repeat = 3

r = np.random.lognormal(np.log(0.7), 0.4, S)
delta_i = np.random.lognormal(np.log(0.25), 0.15, S)
K = np.random.lognormal(np.log(1.5), 0.4, (m, S))
V = np.random.lognormal(np.log(0.1), 0.2, (m, S))
mu = np.random.uniform(0.2, 0.4, (m, S))
S_input = np.array([0.05, 0.01, 0.003])
lambda_j = np.array([0.01, 0.01, 0.008])
alpha = 0.001


def simulate(rho_D, recycle_on=True):
    def model(t, y):
        R = y[:m]; N = y[m:m+S]
        f = np.prod(R[:, None] / (K + R[:, None]), axis=0)
        g = r * f
        recycle = mu @ (delta_i * N) if recycle_on else np.zeros(m)
        dR = S_input + rho_D * recycle - V @ (f * N) - lambda_j * R
        dN = N * (g - delta_i) - alpha * N**2
        return np.concatenate([dR, dN])

    y0 = np.concatenate([np.array([5.0, 0.5, 0.3]),
                         np.random.uniform(0.01, 0.5, S)])
    sol = solve_ivp(model, [0, 300], y0, method='LSODA',
                    rtol=1e-4, atol=1e-7, max_step=5.0)
    return int(np.sum(sol.y[m:m+S, -1] > 1e-4))


rho_values = np.linspace(0.0, 2.0, 15)
S_on, S_off, S_on_std, S_off_std = [], [], [], []

for rho_D in rho_values:
    runs_on, runs_off = [], []
    for rep in range(n_repeat):
        runs_on.append(simulate(rho_D, recycle_on=True))
        runs_off.append(simulate(rho_D, recycle_on=False))
    S_on.append(np.mean(runs_on)); S_off.append(np.mean(runs_off))
    S_on_std.append(np.std(runs_on)); S_off_std.append(np.std(runs_off))
    print(f'rho_D={rho_D:.2f}: ON={np.mean(runs_on):.1f}, '
          f'OFF={np.mean(runs_off):.1f}', flush=True)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.errorbar(rho_values, S_on, yerr=S_on_std, fmt='go-', linewidth=2,
            capsize=4, label='ON: remineralization active')
ax.errorbar(rho_values, S_off, yerr=S_off_std, fmt='s--', color='gray',
            linewidth=2, capsize=4, label='OFF: remineralization off')
ax.axvline(x=1.0, color='r', linestyle='--', label='$\\rho_D=1$')
ax.set_xlabel('Remineralization rate $\\rho_D$ (1/d)')
ax.set_ylabel('Surviving species $S^*$')
ax.set_title('Prediction 2 (fixed): Lower-bound effect')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig('prediction2_fixed.png', dpi=150)
plt.close()

idx = np.where(np.array(S_on) >= 0.9 * max(S_on))[0]
print(f"\nCritical rho_D* = {rho_values[idx[0]]:.3f}, "
      f"S_max = {max(S_on):.1f}")