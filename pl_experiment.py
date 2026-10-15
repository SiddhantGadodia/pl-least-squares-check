"""
Small numerical check related to Boumal, Criscitiello & Rebjock (2026),
"Smooth, globally Polyak-Lojasiewicz functions are nonlinear least-squares".

Theorem: smooth, globally PL f on R^n  =>  f = f* + ||phi(x)||^2, phi: R^n -> R^k a submersion,
k = codim of the minimiser set S.

Question tested here: when phi is available, does Gauss-Newton on phi beat gradient descent on f,
and how does the gap depend on the conditioning of D(phi)?

Part A: 1-D example f(x) = x^2 + 3 sin^2 x (nonconvex but PL, Karimi et al. 2016).
        Here phi can be written explicitly: phi(x) = sign(x) * sqrt(f(x)).
Part B: phi(x) = A x + eps * sin(B x), A in R^{k x n} with condition number kappa.
        Minimiser set has codimension k. Compare GD on f and Gauss-Newton on phi as kappa grows.
Run:  python pl_experiment.py      (needs numpy, matplotlib; runs in well under a minute)
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

TOL, MAXIT = 1e-10, 200000

# ---------------- Part A: 1-D ----------------
fA  = lambda x: x**2 + 3*np.sin(x)**2
dfA = lambda x: 2*x + 3*np.sin(2*x)
phiA = lambda x: np.sign(x)*np.sqrt(fA(x))
def dphiA(x):                       # phi' = f' / (2 phi); limit at 0 is 2
    p = phiA(x); return 2.0 if abs(p) < 1e-12 else dfA(x)/(2*p)

xs = np.linspace(-10, 10, 20001); xs = xs[np.abs(xs) > 1e-9]
pl_ratio = dfA(xs)**2 / fA(xs)                       # PL: |f'|^2 >= 2 mu f
dphi_vals = np.array([dphiA(x) for x in xs])
print("Part A: f(x) = x^2 + 3 sin^2 x")
print(f"  min |f'|^2 / (2 f) on [-10,10]   (PL constant mu)   = {pl_ratio.min()/2:.4f}")
print(f"  min |phi'(x)| on [-10,10]        (submersion check) = {np.abs(dphi_vals).min():.4f}")

def gd_1d(x, L=8.0):
    for t in range(MAXIT):
        if fA(x) < TOL: return t
        x -= dfA(x)/L
    return MAXIT
def gn_1d(x):
    for t in range(MAXIT):
        if fA(x) < TOL: return t
        x -= phiA(x)/dphiA(x)
    return MAXIT
starts = [-7.3, -2.1, 0.9, 3.3, 8.8]
print("  start   GD-iters   GN-on-phi-iters")
for x0 in starts: print(f"  {x0:5.1f}   {gd_1d(x0):8d}   {gn_1d(x0):8d}")

# ---------------- Part B: n-D, varying conditioning ----------------
def make_problem(n, k, kappa, eps, rng):
    U,_ = np.linalg.qr(rng.standard_normal((k,k))); V,_ = np.linalg.qr(rng.standard_normal((n,n)))
    s = np.geomspace(1, 1/kappa, k)
    A = U @ np.diag(s) @ V[:k]
    B = rng.standard_normal((k,n))/np.sqrt(n)
    phi = lambda x: A@x + eps*np.sin(B@x)
    J   = lambda x: A + eps*(np.cos(B@x)[:,None]*B)
    return phi, J

def run_gd(phi, J, x):
    for t in range(MAXIT):
        r = phi(x); f = r@r
        if f < TOL: return t
        g = 2*J(x).T@r
        step = 1.0                                   # backtracking (Armijo)
        while True:
            xn = x - step*g; rn = phi(xn)
            if rn@rn <= f - 0.5*step*(g@g) or step < 1e-14: break
            step *= 0.5
        x = xn
    return MAXIT

def run_gn(phi, J, x):
    for t in range(MAXIT):
        r = phi(x); f = r@r
        if f < TOL: return t
        d = np.linalg.lstsq(J(x), -r, rcond=None)[0]  # minimum-norm Gauss-Newton step
        step = 1.0
        while True:
            rn = phi(x+step*d)
            if rn@rn < f or step < 1e-14: break
            step *= 0.5
        x = x + step*d
    return MAXIT

n, k, eps = 20, 5, 0.3
kappas = [1, 3, 10, 30, 100]
rng = np.random.default_rng(0)
print(f"\nPart B: n={n}, k={k}, eps={eps}  (median over 5 random starts)")
print("  kappa   GD-iters   GN-iters")
gd_med, gn_med = [], []
for kap in kappas:
    phi, J = make_problem(n, k, kap, eps, rng)
    starts = [rng.standard_normal(n)*3 for _ in range(5)]
    g = np.median([run_gd(phi, J, x.copy()) for x in starts])
    h = np.median([run_gn(phi, J, x.copy()) for x in starts])
    gd_med.append(g); gn_med.append(h)
    print(f"  {kap:5d}   {g:8.0f}   {h:8.0f}")

plt.figure(figsize=(5,3.4))
plt.loglog(kappas, gd_med, "o-", label="GD on f (Armijo)")
plt.loglog(kappas, gn_med, "s-", label="Gauss-Newton on phi")
plt.xlabel("condition number of A  (kappa)"); plt.ylabel("iterations to f < 1e-10")
plt.legend(); plt.grid(alpha=.3, which="both"); plt.tight_layout(); plt.savefig("pl_gd_vs_gn.png", dpi=200)
print("\nsaved pl_gd_vs_gn.png")
