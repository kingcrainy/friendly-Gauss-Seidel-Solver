"""
Vercel Python Serverless Function
----------------------------------
POST /api/solve
Body: { "A": [[...]], "b": [...], "x0": [...] | null, "tol": float, "maxIter": int }
Returns: { solution, history, converged, diagonallyDominant, residual, error? }

Vercel auto-detects any .py file in /api exposing a `handler` class that
subclasses BaseHTTPRequestHandler and turns it into a serverless endpoint.
No extra config needed beyond requirements.txt in the project root.
"""

from http.server import BaseHTTPRequestHandler
import json
import numpy as np


def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float).copy()
    history = []

    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]

        err = float(np.max(np.abs(x - x_old)))
        history.append({"iteration": k, "error": err, "x": x.tolist()})

        if err < tol:
            return x, history, True

    return x, history, False


def is_diagonally_dominant(A):
    A = np.abs(A)
    diag = np.diag(A)
    off_diag_sum = A.sum(axis=1) - diag
    return bool(np.all(diag >= off_diag_sum)) and bool(np.any(diag > off_diag_sum))


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw or b"{}")

            A_list = payload.get("A")
            b_list = payload.get("b")
            x0_list = payload.get("x0")
            tol = float(payload.get("tol", 1e-6))
            max_iter = int(payload.get("maxIter", 100))

            if not A_list or not b_list:
                self._send_json(400, {"error": "Missing A or b."})
                return

            A = np.array(A_list, dtype=float)
            b = np.array(b_list, dtype=float)
            n = len(b)

            if A.shape != (n, n):
                self._send_json(400, {"error": f"A must be {n}x{n} to match b's length ({n})."})
                return

            if np.any(np.diag(A) == 0):
                self._send_json(400, {"error": "Zero on the diagonal of A — Gauss-Seidel needs nonzero diagonal entries. Try reordering the equations."})
                return

            x0 = np.array(x0_list, dtype=float) if x0_list else None
            dominant = is_diagonally_dominant(A)

            x, history, converged = gauss_seidel(A, b, x0=x0, tol=tol, max_iter=max_iter)
            residual = (b - A @ x).tolist()

            self._send_json(200, {
                "solution": x.tolist(),
                "history": history,
                "converged": converged,
                "diagonallyDominant": dominant,
                "residual": residual,
            })
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": str(exc)})
