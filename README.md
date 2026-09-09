# Gauss-Seidel Calculator (Streamlit)

A Python web app for solving a system of linear equations, `Ax = b`, using
the **Gauss-Seidel method** — an iterative technique that refines a guess
for `x` one sweep at a time until it settles on the answer.

## What it does

You enter a square matrix `A`, a vector `b`, and (optionally) a starting
guess `x₀`. The app then:

1. Checks whether `A` is **diagonally dominant** — the condition that
   guarantees Gauss-Seidel will converge.
2. Runs the iteration until the change between sweeps drops below your
   chosen tolerance, or it hits the iteration limit.
3. Shows the solution, a table of every iteration, a convergence chart, and
   a residual check (`b − Ax`, which should be ≈ 0 for a correct solution).

## How Gauss-Seidel works

For a system `Ax = b`, each variable `x_i` can be isolated from its row:

```
x_i = (b_i − Σ(j≠i) a_ij·x_j) / a_ii
```

Gauss-Seidel starts from a guess (usually all zeros) and repeatedly sweeps
through this formula for every `i`, **always using the most recently
updated values** — unlike the related Jacobi method, which only uses values
from the previous full sweep. That immediate reuse is what makes Gauss-Seidel
converge faster in practice.

It's guaranteed to converge when `A` is diagonally dominant or symmetric
positive definite. Outside those conditions it can still converge, but
isn't guaranteed to — the app flags this either way before you run it.

## Project structure

```
gauss-seidel-app/
├── app.py                    ← Streamlit UI + the Gauss-Seidel solver
├── requirements.txt          ← Python dependencies
└── .streamlit/config.toml     ← dark, glass-style theme
```

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

## Deploying it

- **Streamlit Community Cloud** (free): push this folder to GitHub, then
  deploy at https://share.streamlit.io by pointing it at `app.py`.
- **Docker / any Python host**: run `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`.

There's also a plain HTML/JS + Python-serverless version of this same
calculator, built for Vercel, in the sibling `gauss-seidel-vercel/` project
if you'd rather deploy without Streamlit.
