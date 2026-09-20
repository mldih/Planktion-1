import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ====================================================================
# Prediction 3: Leakage rate vs diversity (optimized)
# Changes:
#   1. Symbol c -> V, removed 0.5 * recycle
#   2. n_repeat 3 -> 30
#   3. Statistics: mean±std -> median + IQR
#   4. Integration time 300 -> 1000
#   5. Print distribution to detect multistability
#   6. 20 scan points
# ====================================================================

S = 30; m = 3
np.random.seed(42)
n_repeat = 30

r = np.random.lognormal(np.log(0.7), 0.4, S)
delta_i = np.random.lognormal(np.log(0.25), 0.15, S)
K = np.random.lognormal(np.log(1.5), 0.4, (m, S))
V = np.random.lognormal(np.log(0.1), 0.2, (m, S))
mu = np.random.uniform(0.2, 0.4, (m, S))
S_input = np.array([0.05, 0.01, 0.003])
rho_D = 1.0
alpha = 0.001


def simulate(lambda_vec, recycle_on=True):
    def model(t, y):
        R = y[:m]; N = y[m:m+S]
        f = np.prod(R[:, None] / (K + R[:, None]), axis=0)
        g = r * f
        recycle = mu @ (delta_i * N) if recycle_on else np.zeros(m)
        dR = S_input + rho_D * recycle - V @ (f * N) - lambda_vec * R
        dN = N * (g - delta_i) - alpha * N**2
        return np.concatenate([dR, dN])

    y0 = np.concatenate([np.array([5.0, 0.5, 0.3]),
                         np.random.uniform(0.01, 0.5, S)])
    sol = solve_ivp(model, [0, 1000], y0, method='LSODA',
                    rtol=1e-4, atol=1e-7, max_step=5.0)
    return int(np.sum(sol.y[m:m+S, -1] > 1e-4))


lambda_values = np.linspace(0.0, 0.15, 20)
S_on_med, S_on_q25, S_on_q75, S_off_med = [], [], [], []

for lam in lambda_values:
    lambda_vec = np.array([lam, lam, lam * 0.8])
    runs_on = [simulate(lambda_vec, True) for _ in range(n_repeat)]
    runs_off = [simulate(lambda_vec, False) for _ in range(n_repeat)]

    S_on_med.append(np.median(runs_on))
    S_on_q25.append(np.percentile(runs_on, 25))
    S_on_q75.append(np.percentile(runs_on, 75))
    S_off_med.append(np.median(runs_off))

    unique, counts = np.unique(runs_on, return_counts=True)
    print(f'lambda={lam:.4f}: median={np.median(runs_on):.0f}, '
          f'dist={dict(zip(unique.tolist(), counts.tolist()))}', flush=True)

S_on_med = np.array(S_on_med)
S_on_err = np.array([S_on_med - np.array(S_on_q25),
                     np.array(S_on_q75) - S_on_med])

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.errorbar(lambda_values, S_on_med, yerr=S_on_err, fmt='ro-',
            linewidth=2, capsize=4, label='ON (median + IQR)')
ax.plot(lambda_values, S_off_med, 's--', color='gray',
        linewidth=2, label='OFF (median)')
ax.set_xlabel('Leakage rate $\\lambda$ (1/d)')
ax.set_ylabel('Surviving species $S^*$')
ax.set_title('Prediction 3 (optimized): Leakage vs diversity')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig('prediction3_optimized.png', dpi=150)
plt.close()

for i, lam in enumerate(lambda_values):
    if S_on_med[i] < 0.5 * max(S_on_med):
        print(f"\nS* < 50% max at lambda = {lam:.4f}")
        break