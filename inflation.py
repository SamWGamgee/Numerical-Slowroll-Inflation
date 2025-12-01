import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# --- Global constants ---
S = 5e-5         # Dimensional scaling factor (rescaling between dimensionless and dimensional quantities)
Ai = 1e-3        # Initial scale factor
T = np.linspace(0, 1000, 100000)  # Time array for integrating ODEs


# --- Helper Functions ---

def i(f, val):
    """Return the index of the largest element in array f that is <= val."""
    return np.max(np.where(f <= val))


def solve4x(x, f_func, fval):
    """Helper function for root-finding (used in brentq)."""
    return f_func(x) - fval


# --- Main Background Solver ---

def background(model_params, model_funcs, helper_funcs, 
               Nt=77.4859, Nk=60, P_star=2.1e-9, 
               phi_limit=30, maxiter=100):
    """
    Solves the background evolution of an inflationary model.

    This function integrates the background (homogeneous) equations of motion 
    for a scalar field during inflation and computes derived quantities such 
    as the Hubble parameter, slow-roll parameters, and primordial power spectra.
    """

    # --- Unpack model and helper functions ---
    f, dfdx, d2fdx2 = model_funcs
    calc_xend, calc_N = helper_funcs 

    # --- Determine key field values ---
    x_end = calc_xend(model_params)

    # Field value Nk e-folds before end of inflation
    try:
        x_star = brentq(lambda x: solve4x(x, lambda x: calc_N(x, x_end, model_params), Nk),
                        0, phi_limit, xtol=1e-15, rtol=1e-15, maxiter=maxiter)
    except ValueError as e:
        print("\n[ERROR] Root finding failed for x_star (Nk e-folds before end of inflation).")
        print(f"Reason: {e}")
        print(f"Try adjusting 'phi_limit' (current={phi_limit}) or checking 'calc_N' validity.")
        raise SystemExit("Computation stopped due to brentq failure for x_star.\n")

    # Potential normalization using COBE normalization
    V0 = P_star * (12 * np.pi**2) * (dfdx(x_star, model_params) ** 2) / f(x_star, model_params) ** 3

    # Field value corresponding to Nt total e-folds
    try:
        xi = brentq(lambda x: solve4x(x, lambda x: calc_N(x, x_end, model_params), Nt),
                    0, phi_limit, xtol=1e-15, rtol=1e-15, maxiter=maxiter)
    except ValueError as e:
        print("\n[ERROR] Root finding failed for xi (field value at Nt total e-folds).")
        print(f"Reason: {e}")
        print(f"Try increasing 'phi_limit' or checking the consistency of 'Nt' and 'calc_N'.")
        raise SystemExit("Computation stopped due to brentq failure for xi.\n")

    # --- Initial conditions for integration ---
    yi = 0  # initial field velocity
    zi = np.sqrt(yi**2 / 6 + (V0 * f(xi, model_params) / (3 * S**2)))  # Hubble parameter
    init_cond = [xi, yi, zi, Ai]

    # --- Define background system of ODEs ---
    def sys(T, var):
        x, y, z, A = var
        dxdT = y
        dydT = -3 * z * y - V0 * dfdx(x, model_params) / S**2
        dzdT = -0.5 * y**2
        dAdT = A * z
        return [dxdT, dydT, dzdT, dAdT]

    # --- Integrate system ---
    sol = solve_ivp(sys, [T[0], T[-1]], init_cond, t_eval=T, method='LSODA',
                rtol=3e-14, atol=2e-35, max_step=np.inf)
    
    # --- Safety checks ---
    if not sol.success:
        print(f"❌ Integration failed: {sol.message}")
        raise RuntimeError("ODE integration did not converge.")

    if not np.all(np.isfinite(sol.y)):
        print("❌ ODE integration failed: NaN or Inf encountered in solution.")
        raise RuntimeError("Non-finite values detected in ODE integration.")
        
    x, y, z, A = sol.y

    # --- Convert back to dimensional quantities ---
    phi, phi_t, H = x, y * S, z * S
    N = np.log(A / Ai)
    Ne = Nt - N

    # --- Compute slow-roll parameters ---
    eps = -(-z**2 + ((V0 * f(x, model_params) / S**2 - y**2)) / 3) / z**2
    eta = -(-3 * z * y - V0 * dfdx(x, model_params) / S**2) / (y * z)

    # --- Inflationary observables ---
    ns = 1 + 2 * eta - 4 * eps
    nT = -2 * eps
    r = 16 * eps
    Ps = (S * z) ** 2 / (8 * np.pi**2 * eps)
    Pt = 2 * (S * z) ** 2 / (np.pi**2)

    return phi, phi_t, H, N, Ne, eps, eta, ns, nT, r, Ps, Pt, V0


# --- Mukhanov–Sasaki Solver ---

def mukhanov_sasaki(model_params, model_var, model_funcs, Nt=77.4859, Nk=60):
    """
    Solves the Mukhanov–Sasaki equations for scalar and tensor perturbations.

    This function evolves scalar and tensor modes on a dynamically evolving 
    inflationary background to compute the corresponding power spectra.
    """

    # --- Unpack model components ---
    f, dfdx, d2fdx2 = model_funcs
    phi, phi_t, H, N, V0 = model_var
    A = Ai * np.exp(N)

    # Convert to dimensionless variables
    x, y, z = phi, phi_t / S, H / S
    aH = A * z

    # --- Determine horizon crossing index ---
    ind = i(N, Nt - (Nk + 5))
    k_star = aH[i(N, Nt - Nk)]

    # --- Initial field conditions at starting point ---
    xqi = x[ind]
    yqi = y[ind]
    zqi = np.sqrt(yqi**2 / 6 + (V0 * f(xqi, model_params) / (3 * S**2)))
    Aqi = 1e-3 * np.exp(Nt - (Nk + 5))

    # --- Initial conditions (Bunch–Davies vacuum) ---
    vi = 1 / np.sqrt(2 * k_star)
    ui = 0
    v_Ti = 0
    u_Ti = -k_star * vi / Aqi
    hi = 1 / np.sqrt(2 * k_star)
    gi = 0
    h_Ti = 0
    g_Ti = -k_star * hi / Aqi

    init_cond = [xqi, yqi, zqi, Aqi, vi, v_Ti, ui, u_Ti, hi, h_Ti, gi, g_Ti]

    # --- Define coupled system for background + perturbations ---
    def sys(T, var):
        x, y, z, A, v, v_T, u, u_T, h, h_T, g, g_T = var

        # Background evolution
        dxdT = y
        dydT = -3 * z * y - V0 * dfdx(x, model_params) / S**2
        dzdT = -0.5 * y**2
        dAdT = A * z

        # Scalar perturbations
        dvdT = v_T
        dv_TdT = -z * v_T + v * (
            2.5 * y**2
            + 2 * y * (-3 * z * y - V0 * dfdx(x, model_params) / S**2) / z
            + 2 * z**2
            + 0.5 * y**4 / z**2
            - V0 * d2fdx2(x, model_params) / S**2
            - k_star**2 / A**2
        )

        dudT = u_T
        du_TdT = -z * u_T + u * (
            2.5 * y**2
            + 2 * y * (-3 * z * y - V0 * dfdx(x, model_params) / S**2) / z
            + 2 * z**2
            + 0.5 * y**4 / z**2
            - V0 * d2fdx2(x, model_params) / S**2
            - k_star**2 / A**2
        )

        # Tensor perturbations
        dhdT = h_T
        dh_TdT = -z * h_T - h * (k_star**2 / A**2 - 2 * z**2 + 0.5 * y**2)
        dgdT = g_T
        dg_TdT = -z * g_T - g * (k_star**2 / A**2 - 2 * z**2 + 0.5 * y**2)

        return [dxdT, dydT, dzdT, dAdT, dvdT, dv_TdT, dudT, du_TdT, dhdT, dh_TdT, dgdT, dg_TdT]

    # --- Integrate the system ---
    sol = solve_ivp(sys, [T[0], T[-1]], init_cond, t_eval=T, method='LSODA',
                rtol=3e-14, atol=2e-35, max_step=np.inf)
    
    # --- Safety checks ---
    if not sol.success:
        print(f"❌ Integration failed: {sol.message}")
        raise RuntimeError("ODE integration did not converge.")

    if not np.all(np.isfinite(sol.y)):
        print("❌ ODE integration failed: NaN or Inf encountered in solution.")
        raise RuntimeError("Non-finite values detected in ODE integration.")
    
    xq, yq, zq, Aq, v, v_T, u, u_T, h, h_T, g, g_T = sol.y
        
    # Convert back to dimensional quantities
    phiq, phiq_t, Hq = xq, yq * S, zq * S
    aHk = (Aq * zq) / k_star

    # --- Derived quantities ---
    meff = (
        2.5 * yq**2
        + 2 * yq * (-3 * zq * yq - V0 * dfdx(xq, model_params) / S**2) / zq
        + 2 * zq**2
        + 0.5 * yq**4 / zq**2
        - V0 * d2fdx2(xq, model_params) / S**2
    )

    # Slow-roll parameters
    epsq = -(-zq**2 + ((V0 * f(xq, model_params) / S**2 - yq**2)) / 3) / zq**2
    etaq = -(-3 * zq * yq - V0 * dfdx(xq, model_params) / S**2) / (yq * zq)

    # E-fold count and remaining inflation
    Nq = np.log(Aq / Aqi)
    Nqe = Nt - Nq

    # --- Power spectra ---
    R2 = (v**2 + u**2) / (2 * epsq * (Aq / S)**2)
    P_S = (k_star**3 * R2) / (2 * np.pi**2)
    h2 = (h**2 + g**2) / ((Aq / S)**2)
    P_T = 4 * (k_star**3 * h2) / (np.pi**2)

    return phiq, phiq_t, Hq, Nq, Nqe, epsq, etaq, P_S, P_T, meff
