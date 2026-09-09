"""
Gauss-Seidel Linear System Solver
----------------------------------
An interactive Streamlit app that solves Ax = b using the Gauss-Seidel
iterative method, with convergence diagnostics and step-by-step iteration
tracking.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy:
    Push this folder to a GitHub repo and deploy for free at
    https://share.streamlit.io (Streamlit Community Cloud), or run it
    on any host that can run `streamlit run app.py` (Render, Railway,
    a VM, Docker, etc.).
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Gauss-Seidel Calculator", page_icon="🧮", layout="wide")

# --------------------------------------------------------------------------- #
# Theme: glassmorphic dark UI matching the web (Vercel) version of this app
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
      :root{
        --violet:#8B7CF6; --cyan:#5EEAD4; --pink:#F472B6;
        --green:#4ADE80; --amber:#FBBF24; --red:#FB7185;
        --ink:#F3F5FB; --muted:#93A0C4;
        --glass: rgba(255,255,255,0.055); --border: rgba(255,255,255,0.14);
      }
      html, body, [class*="css"]{ font-family:'Inter', sans-serif; }

      /* animated blurred background blobs behind the app */
      .stApp{
        background:
          radial-gradient(560px 560px at -10% -10%, rgba(139,124,246,0.35), transparent 65%),
          radial-gradient(500px 500px at 110% 10%, rgba(94,234,212,0.30), transparent 65%),
          radial-gradient(420px 420px at 50% 120%, rgba(244,114,182,0.20), transparent 65%),
          #0B0F1E;
        background-attachment: fixed;
      }

      h1, h2, h3{
        font-family:'Sora', sans-serif !important;
        letter-spacing:-0.02em;
      }
      h1{
        background: linear-gradient(135deg, #ffffff 30%, var(--cyan) 90%);
        -webkit-background-clip: text; background-clip: text; color: transparent !important;
      }

      /* glass panels: sidebar + block containers */
      section[data-testid="stSidebar"]{
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(18px);
        border-right: 1px solid var(--border);
      }
      div[data-testid="stVerticalBlockBorderWrapper"], .stDataFrame, div[data-testid="stExpander"]{
        background: var(--glass) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(16px);
      }

      /* inputs */
      input, textarea, .stNumberInput input{
        font-family:'JetBrains Mono', monospace !important;
        background: rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        color: var(--ink) !important;
      }

      /* primary button -> gradient pill, matches the web app's Solve button */
      .stButton > button, .stButton > button[kind="primary"]{
        background: linear-gradient(135deg, var(--violet), var(--cyan)) !important;
        color: #0B0F1E !important;
        border: none !important;
        border-radius: 12px !important;
        font-family:'Sora', sans-serif !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 24px rgba(139,124,246,0.35);
        transition: transform .15s ease, box-shadow .15s ease;
      }
      .stButton > button:hover{
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(139,124,246,0.45);
      }

      /* metrics (solution values) get the gradient treatment */
      div[data-testid="stMetricValue"]{
        font-family:'JetBrains Mono', monospace !important;
        background: linear-gradient(135deg, var(--violet), var(--cyan));
        -webkit-background-clip: text; background-clip: text; color: transparent !important;
        font-weight: 600 !important;
      }
      div[data-testid="stMetricLabel"]{ color: var(--muted) !important; }

      /* alerts (success/warning/error banners) */
      div[data-testid="stAlert"]{
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
        backdrop-filter: blur(12px);
      }

      caption, .stCaption{ color: var(--muted) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Core numerical routine
# --------------------------------------------------------------------------- #
def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Solve Ax = b using the Gauss-Seidel iterative method.

    Parameters
    ----------
    A : (n, n) ndarray
    b : (n,) ndarray
    x0 : (n,) ndarray or None -- initial guess (defaults to zeros)
    tol : float -- stopping tolerance on the infinity-norm of successive
          differences
    max_iter : int -- maximum number of iterations

    Returns
    -------
    x : (n,) ndarray -- final solution estimate
    history : list[dict] -- one row per iteration with the iterate,
              the error, and whether it converged
    converged : bool
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float).copy()
    history = []

    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])          # already-updated entries
            s2 = np.dot(A[i, i + 1:], x_old[i + 1:])  # not-yet-updated entries
            x[i] = (b[i] - s1 - s2) / A[i, i]

        err = np.max(np.abs(x - x_old))
        row = {"Iteration": k, "Error": err}
        for i in range(n):
            row[f"x{i + 1}"] = x[i]
        history.append(row)

        if err < tol:
            return x, history, True

    return x, history, False


def is_diagonally_dominant(A):
    A = np.abs(A)
    diag = np.diag(A)
    off_diag_sum = A.sum(axis=1) - diag
    return bool(np.all(diag >= off_diag_sum)) and bool(np.any(diag > off_diag_sum))


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("🧮 Gauss-Seidel Linear System Calculator")
st.caption("Solve Ax = b iteratively, with convergence checks and a full iteration trace.")

with st.sidebar:
    st.header("Settings")
    n = st.number_input("Number of equations / unknowns (n)", min_value=2, max_value=10, value=3, step=1)
    tol = st.number_input("Tolerance", min_value=1e-12, max_value=1.0, value=1e-6, format="%.1e")
    max_iter = st.number_input("Max iterations", min_value=1, max_value=1000, value=50, step=1)
    st.markdown("---")
    st.markdown(
        "**Convergence tip:** Gauss-Seidel is guaranteed to converge if matrix "
        "A is diagonally dominant (each diagonal entry's absolute value is "
        "≥ the sum of the absolute values of the rest of its row)."
    )

st.subheader("1. Enter the coefficient matrix A and vector b")

default_A = np.zeros((n, n))
default_b = np.zeros(n)

col_a, col_b = st.columns([3, 1])

with col_a:
    st.markdown("**Matrix A**")
    A_df = pd.DataFrame(
        default_A,
        columns=[f"col{j+1}" for j in range(n)],
        index=[f"row{i+1}" for i in range(n)],
    )
    A_edit = st.data_editor(A_df, key="A_editor", num_rows="fixed")

with col_b:
    st.markdown("**Vector b**")
    b_df = pd.DataFrame(default_b, columns=["b"], index=[f"row{i+1}" for i in range(n)])
    b_edit = st.data_editor(b_df, key="b_editor", num_rows="fixed")

st.markdown("**Initial guess x₀** (optional — defaults to all zeros)")
x0_df = pd.DataFrame(np.zeros(n), columns=["x0"], index=[f"x{i+1}" for i in range(n)])
x0_edit = st.data_editor(x0_df, key="x0_editor", num_rows="fixed")

A = A_edit.to_numpy(dtype=float)
b = b_edit.to_numpy(dtype=float).flatten()
x0 = x0_edit.to_numpy(dtype=float).flatten()

st.subheader("2. Solve")

if np.any(np.diag(A) == 0):
    st.error("A zero appears on the diagonal of A — Gauss-Seidel requires nonzero diagonal entries. "
              "Try reordering your equations.")
else:
    dominant = is_diagonally_dominant(A)
    if dominant:
        st.success("A is diagonally dominant — convergence is guaranteed.")
    else:
        st.warning("A is NOT diagonally dominant. The method may still converge, but it isn't guaranteed. "
                    "Consider reordering rows/equations to strengthen the diagonal.")

    if st.button("Run Gauss-Seidel", type="primary"):
        x, history, converged = gauss_seidel(A, b, x0=x0, tol=tol, max_iter=int(max_iter))

        if converged:
            st.success(f"Converged in {len(history)} iterations (tolerance {tol:.1e}).")
        else:
            st.error(f"Did not converge within {int(max_iter)} iterations. "
                      f"Last error: {history[-1]['Error']:.3e}")

        st.subheader("3. Solution")
        sol_cols = st.columns(n)
        for i, c in enumerate(sol_cols):
            c.metric(f"x{i + 1}", f"{x[i]:.6f}")

        st.subheader("4. Iteration history")
        hist_df = pd.DataFrame(history).set_index("Iteration")
        st.dataframe(hist_df.style.format("{:.6f}"), use_container_width=True)

        st.subheader("5. Convergence plot")
        st.line_chart(hist_df["Error"])

        st.subheader("6. Verification (residual = b - Ax)")
        residual = b - A @ x
        st.write(pd.DataFrame({"residual": residual}, index=[f"row{i+1}" for i in range(n)]).T)

st.markdown("---")
with st.expander("About the Gauss-Seidel method"):
    st.markdown(
        r"""
The Gauss-Seidel method solves $Ax = b$ by iterating

$$x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j<i} a_{ij}x_j^{(k+1)} - \sum_{j>i} a_{ij}x_j^{(k)}\right)$$

using the **most recently updated** values of each variable within the same
sweep (unlike Jacobi, which uses only values from the previous iteration).
It's guaranteed to converge when A is diagonally dominant or symmetric
positive definite, but can be tried on any square system.
"""
    )
