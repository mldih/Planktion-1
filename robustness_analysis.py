import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ====================================================================
# Robustness analysis with figures
# Checks: seed sensitivity, parameter perturbation (+/-20%),
#         integration time
# Figures: P1-P4 robustness panels + evolution model parameter space
# ====================================================================

S = 30
m = 3
q = 3


def build_params(seed, param_scale=1.0):
    np.random.seed(seed)
    r = np.random.lognormal(np.log(0.7), 0.4, S) * param_scale
    K = np.random.lognormal(np.log(1.5), 0.4, (m, S))
    V = np.random.lognormal(np.log(0.1), 0.2, (m, S)) * param_scale
    mu = np.random.uniform(0.2, 0.4, (m, S)) * param_scale
    delta_i = np.random.lognormal(np.log(0.25), 0.15, S)
    return r, K, V, mu, delta_i


S_input = np.array([0.05, 0.01, 0.003])
lambda_j = np.array([0.01, 0.01, 0.008])
alpha = 0.001


# ---------- P1 ----------
def run_P1(seed, param_scale=1.0, T=300):
    r, K, V, mu, _ = build_params(seed, param_scale)

    def simulate(delta_mean, recycle_on):
        delta_vec = np.random.lognormal(np.log(delta_mean), 0.15, S)
        def model(t, y):
            R = y[:m]; N = y[m:m+S]
            f = np.prod(R[:, None] / (K + R[:, None]), axis=0)
            g = r * f
            recycle = mu @ (delta_vec * N) if recycle_on else np.zeros(m)
            dR = S_input + recycle - V @ (f * N) - lambda_j * R
            dN = N * (g - delta_vec) - alpha * N**2
            return np.concatenate([dR, dN])
        y0 = np.concatenate([np.array([5.0, 0.5, 0.3]),
                             np.random.uniform(0.01, 0.5, S)])
        sol = solve_ivp(model, [0, T], y0, method='LSODA',
                        rtol=1e-4, atol=1e-7, max_step=5.0)
        return int(np.sum(sol.y[m:m+S, -1] > 1e-4))

    dv = np.linspace(0.05, 0.55, 12)
    S_A = np.array([simulate(d, False) for d in dv])
    S_B = np.array([simulate(d, True) for d in dv])
    return dv, S_A, S_B


# ---------- P2 ----------
def run_P2(seed, param_scale=1.0, T=300):
    r, K, V, mu, delta_i = build_params(seed, param_scale)

    def simulate(rho_D):
        def model(t, y):
            R = y[:m]; N = y[m:m+S]
            f = np.prod(R[:, None] / (K + R[:, None]), axis=0)
            g = r * f
            recycle = mu @ (delta_i * N)
            dR = S_input + rho_D * recycle - V @ (f * N) - lambda_j * R
            dN = N * (g - delta_i) - alpha * N**2
            return np.concatenate([dR, dN])
        y0 = np.concatenate([np.array([5.0, 0.5, 0.3]),
                             np.random.uniform(0.01, 0.5, S)])
        sol = solve_ivp(model, [0, T], y0, method='LSODA',
                        rtol=1e-4, atol=1e-7, max_step=5.0)
        return int(np.sum(sol.y[m:m+S, -1] > 1e-4))

    rv = np.linspace(0.0, 2.0, 12)
    return rv, np.array([simulate(r) for r in rv])


# ---------- P3 ----------
def run_P3(seed, param_scale=1.0, T=1000, n_rep=15):
    r, K, V, mu, delta_i = build_params(seed, param_scale)

    def simulate(lam):
        lambda_vec = np.array([lam, lam, lam * 0.8])
        def model(t, y):
            R = y[:m]; N = y[m:m+S]
            f = np.prod(R[:, None] / (K + R[:, None]), axis=0)
            g = r * f
            recycle = mu @ (delta_i * N)
            dR = S_input + recycle - V @ (f * N) - lambda_vec * R
            dN = N * (g - delta_i) - alpha * N**2
            return np.concatenate([dR, dN])
        y0 = np.concatenate([np.array([5.0, 0.5, 0.3]),
                             np.random.uniform(0.01, 0.5, S)])
        sol = solve_ivp(model, [0, T], y0, method='LSODA',
                        rtol=1e-4, atol=1e-7, max_step=5.0)
        return int(np.sum(sol.y[m:m+S, -1] > 1e-4))

    lv = np.linspace(0.0, 0.15, 10)
    med = []
    for lam in lv:
        runs = [simulate(lam) for _ in range(n_rep)]
        med.append(np.median(runs))
    return lv, np.array(med)


# ---------- P4 ----------
def run_P4(seed, param_scale=1.0, T=400):
    r, K, V, mu, delta_i = build_params(seed, param_scale)
    g_max = np.array([1.0, 0.8, 0.7])
    K_Z = np.array([0.5, 0.6, 0.7])
    m_k = np.array([0.1, 0.1, 0.12])

    def model(t, y, graze_on):
        R = y[:m]; N = y[m:m+S]; Z = y[m+S:m+S+q]
        f = np.prod(R[:, None] / (K + R[:, None]), axis=0)
        g = r * f
        total_N = np.sum(N) + 1e-10
        h = g_max[:, None] * N[None, :] / (K_Z[:, None] + total_N)
        graze = h.T @ Z if graze_on else np.zeros(S)
        recycle = mu @ ((delta_i + graze) * N)
        dR = S_input + recycle - V @ (f * N) - lambda_j * R
        dN = N * (g - delta_i - graze) - alpha * N**2
        dZ = (Z * (0.7 * (h * N[None, :]).sum(axis=1) - m_k)
              if graze_on else -m_k * Z)
        return np.concatenate([dR, dN, dZ])

    def shannon(N):
        p = N / (np.sum(N) + 1e-10)
        p = p[p > 1e-10]
        return -np.sum(p * np.log(p))

    def run(graze_on):
        Z0 = np.array([0.1, 0.1, 0.1]) if graze_on else np.zeros(3)
        y0 = np.concatenate([np.array([5.0, 0.5, 0.3]),
                             np.random.uniform(0.01, 0.5, S), Z0])
        sol = solve_ivp(model, [0, T], y0, args=(graze_on,),
                        method='LSODA', rtol=1e-4, atol=1e-7, max_step=5.0)
        return shannon(sol.y[m:m+S, -1])

    return run(True), run(False)


# ---------- Evolution model ----------
def run_evo(r0, alpha_max, c_cost, T=400):
    delta0 = 0.1; beta_max = 1.5; b_cost = 0.3
    v_H = 0.1; v_V = 0.1; K_N = 50.0; K_V = 10.0

    def evolution(t, state):
        x, y, N, V = state
        dw_H_dx = -2 * c_cost * x + alpha_max
        dw_V_dy = beta_max * N / (K_N + N) - 2 * b_cost * y
        dxdt = v_H * dw_H_dx
        dydt = v_V * dw_V_dy
        r = r0 - c_cost * x**2
        delta = delta0 + beta_max * y - alpha_max * x
        dNdt = N * (r - delta) * (1 - N / K_N)
        dVdt = V * (beta_max * y * N / (K_N + N) - b_cost * y**2) * (1 - V / K_V)
        return [dxdt, dydt, dNdt, dVdt]

    sol = solve_ivp(evolution, (0, T), [0.1, 0.1, 10.0, 1.0],
                    t_eval=np.linspace(0, T, 2000), method='LSODA')
    delta_t = delta0 + beta_max * sol.y[1] - alpha_max * sol.y[0]
    return delta_t[-1]


# ====================================================================
# Collect results
# ====================================================================

print("Collecting robustness results...")

seeds = [1, 2, 3, 42, 100]
scales = [0.8, 1.0, 1.2]
scale_labels = ['-20%', 'base', '+20%']
Ts_P1 = [300, 800, 1500]
Ts_P3 = [500, 1000, 2000]

# P1
P1_seed_B = [run_P1(s)[2].max() for s in seeds]
P1_scale_B = [run_P1(42, param_scale=sc)[2].max() for sc in scales]
P1_time_B = [run_P1(42, T=T)[2].max() for T in Ts_P1]

# P2
P2_seed_star = []
for s in seeds:
    rv, S_star = run_P2(s)
    idx = np.where(S_star >= 0.9 * S_star.max())[0]
    P2_seed_star.append(rv[idx[0]] if len(idx) > 0 else np.nan)
P2_scale_star = []
for sc in scales:
    rv, S_star = run_P2(42, param_scale=sc)
    idx = np.where(S_star >= 0.9 * S_star.max())[0]
    P2_scale_star.append(rv[idx[0]] if len(idx) > 0 else np.nan)

# P3
P3_seed_low, P3_seed_high = [], []
for s in seeds:
    lv, med = run_P3(s)
    P3_seed_low.append(med[np.argmin(np.abs(lv - 0.05))])
    P3_seed_high.append(med[np.argmin(np.abs(lv - 0.13))])
P3_scale_low, P3_scale_high = [], []
for sc in scales:
    lv, med = run_P3(42, param_scale=sc)
    P3_scale_low.append(med[np.argmin(np.abs(lv - 0.05))])
    P3_scale_high.append(med[np.argmin(np.abs(lv - 0.13))])

# P4
P4_seed_dH = []
for s in seeds:
    Hw, Hwo = run_P4(s)
    P4_seed_dH.append(Hw - Hwo)
P4_scale_dH = []
for sc in scales:
    Hw, Hwo = run_P4(42, param_scale=sc)
    P4_scale_dH.append(Hw - Hwo)

# Evolution model parameter space
r0_vals = [0.6, 0.7, 0.8]
alpha_vals = [0.8, 0.9, 1.0]
c_vals = [0.4, 0.5, 0.6]
evo_delta = np.zeros((len(r0_vals), len(alpha_vals), len(c_vals)))
for i, r0 in enumerate(r0_vals):
    for j, a in enumerate(alpha_vals):
        for k, c in enumerate(c_vals):
            evo_delta[i, j, k] = run_evo(r0, a, c)


# ====================================================================
# Figures
# ====================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- P1 ---
axes[0, 0].bar(range(len(seeds)), P1_seed_B, color='steelblue')
axes[0, 0].set_xticks(range(len(seeds)))
axes[0, 0].set_xticklabels(seeds)
axes[0, 0].set_xlabel('Seed'); axes[0, 0].set_ylabel('B peak $S^*$')
axes[0, 0].set_title('P1: Seed sensitivity'); axes[0, 0].grid(True, axis='y')

# --- P2 ---
axes[0, 1].bar(range(len(seeds)), P2_seed_star, color='seagreen')
axes[0, 1].set_xticks(range(len(seeds)))
axes[0, 1].set_xticklabels(seeds)
axes[0, 1].set_xlabel('Seed'); axes[0, 1].set_ylabel('Critical $\\rho_D^*$')
axes[0, 1].set_title('P2: Seed sensitivity'); axes[0, 1].grid(True, axis='y')

# --- P3 ---
x = np.arange(len(seeds)); w = 0.35
axes[0, 2].bar(x - w/2, P3_seed_low, w, label='$S^*(\\lambda=0.05)$', color='steelblue')
axes[0, 2].bar(x + w/2, P3_seed_high, w, label='$S^*(\\lambda=0.13)$', color='indianred')
axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(seeds)
axes[0, 2].set_xlabel('Seed'); axes[0, 2].set_ylabel('$S^*$ (median)')
axes[0, 2].set_title('P3: Seed sensitivity'); axes[0, 2].legend(); axes[0, 2].grid(True, axis='y')

# --- P4 ---
axes[1, 0].bar(range(len(seeds)), P4_seed_dH, color='purple')
axes[1, 0].axhline(0, color='k', linewidth=0.8)
axes[1, 0].set_xticks(range(len(seeds)))
axes[1, 0].set_xticklabels(seeds)
axes[1, 0].set_xlabel('Seed'); axes[1, 0].set_ylabel('$\\Delta H$')
axes[1, 0].set_title('P4: Seed sensitivity'); axes[1, 0].grid(True, axis='y')

# --- Parameter perturbation summary ---
metrics = ['P1 B peak', 'P2 rho_D*', 'P3 S*(low)', 'P3 S*(high)', 'P4 dH']
base_vals = [P1_scale_B[1], P2_scale_star[1],
             P3_scale_low[1], P3_scale_high[1], P4_scale_dH[1]]
low_vals = [P1_scale_B[0], P2_scale_star[0],
            P3_scale_low[0], P3_scale_high[0], P4_scale_dH[0]]
high_vals = [P1_scale_B[2], P2_scale_star[2],
             P3_scale_low[2], P3_scale_high[2], P4_scale_dH[2]]

x = np.arange(len(metrics)); w = 0.25
axes[1, 1].bar(x - w, low_vals, w, label='-20%', color='lightcoral')
axes[1, 1].bar(x, base_vals, w, label='base', color='steelblue')
axes[1, 1].bar(x + w, high_vals, w, label='+20%', color='seagreen')
axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(metrics, rotation=20)
axes[1, 1].set_ylabel('Metric value')
axes[1, 1].set_title('Parameter perturbation (+/-20%)')
axes[1, 1].legend(); axes[1, 1].grid(True, axis='y')

# --- Integration time ---
axes[1, 2].plot(Ts_P1, P1_time_B, 'bo-', linewidth=2, label='P1 B peak')
axes[1, 2].set_xlabel('Integration time $T$')
axes[1, 2].set_ylabel('P1 B peak $S^*$')
axes[1, 2].set_title('P1: Integration time')
axes[1, 2].legend(); axes[1, 2].grid(True)

plt.tight_layout()
plt.savefig('robustness_analysis.png', dpi=150)
plt.close()


# --- Evolution model parameter space (separate figure) ---
fig, ax = plt.subplots(figsize=(10, 6))
for i, r0 in enumerate(r0_vals):
    for j, a in enumerate(alpha_vals):
        for k, c in enumerate(c_vals):
            d = evo_delta[i, j, k]
            color = 'green' if 0.177 <= d <= 0.346 else 'red'
            ax.scatter(r0, d, color=color, s=80, alpha=0.7,
                       edgecolor='k', linewidth=0.5)
            ax.annotate(f"a={a},c={c}", (r0, d), fontsize=6,
                        xytext=(3, 3), textcoords='offset points')
ax.axhspan(0.177, 0.346, alpha=0.2, color='green',
           label='Criterion window')
ax.set_xlabel('$r_0$')
ax.set_ylabel('Equilibrium mortality $\\delta^*$')
ax.set_title('Evolution model: parameter space (27 combinations)')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig('evolution_model_parameter_space.png', dpi=150)
plt.close()

print("\nFigures saved:")
print("  - robustness_analysis.png")
print("  - evolution_model_parameter_space.png")

# ---------- Text summary ----------
print("\n" + "=" * 70)
print("Robustness summary")
print("=" * 70)
print(f"P1 seed B-peak: {[f'{x:.0f}' for x in P1_seed_B]}")
print(f"P1 param B-peak: {[f'{x:.0f}' for x in P1_scale_B]}")
print(f"P1 time B-peak: {[f'{x:.0f}' for x in P1_time_B]}")
print(f"P2 seed rho_D*: {[f'{x:.2f}' for x in P2_seed_star]}")
print(f"P2 param rho_D*: {[f'{x:.2f}' for x in P2_scale_star]}")
print(f"P3 seed S*(0.05): {[f'{x:.0f}' for x in P3_seed_low]}")
print(f"P3 seed S*(0.13): {[f'{x:.0f}' for x in P3_seed_high]}")
print(f"P4 seed dH: {[f'{x:+.3f}' for x in P4_seed_dH]}")
print(f"P4 param dH: {[f'{x:+.3f}' for x in P4_scale_dH]}")

n_in = np.sum((evo_delta >= 0.177) & (evo_delta <= 0.346))
print(f"Evolution model: {n_in}/27 combinations inside window")