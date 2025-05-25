#!/usr/bin/env python3
"""
Mesure temps et mémoire pour la couche cachée entière
(passe avant + rétro-propagation + mise à jour des poids).

On fait varier le nombre d’entrées (n_inputs) ; le nombre de neurones
cachés (n_hidden) est paramétrable

Usage :
    python measure_hidden_layer.py            # n_inputs = 2..2048, n_hidden = 2
    python measure_hidden_layer.py -m 4       # même tests mais 4 neurones cachés
    python measure_hidden_layer.py -m 8 -r 20000
"""

import math, random, tracemalloc, time, argparse

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def bench(n_inputs: int, n_hidden: int, repeats: int):
    # données fixes pour la série de répétitions
    X      = [random.random() for _ in range(n_inputs)]
    W_ih   = [[random.random() for _ in range(n_hidden)] for _ in range(n_inputs)]
    B_h    = [random.random() for _ in range(n_hidden)]
    upstream_error = [1.0 for _ in range(n_hidden)]     # simplification

    tracemalloc.start()
    t0 = time.perf_counter()

    for _ in range(repeats):
        # ---------- forward ----------
        H_pot = [sum(X[i] * W_ih[i][j] for i in range(n_inputs)) + B_h[j]
                 for j in range(n_hidden)]
        H_out = [sigmoid(p) for p in H_pot]

        # ---------- backward ----------
        delta_h = [H_out[j] * (1 - H_out[j]) * upstream_error[j]
                   for j in range(n_hidden)]

        # ---------- update poids & biais ----------
        for i in range(n_inputs):
            for j in range(n_hidden):
                W_ih[i][j] += H_out[j] * delta_h[j]    # η mis à 1 pour la mesure
        for j in range(n_hidden):
            B_h[j] += delta_h[j]

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed / repeats, peak

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", "-m", type=int, default=2,
                    help="nombre de neurones cachés (défaut 2)")
    ap.add_argument("--repeats", "-r", type=int, default=10_000,
                    help="répétitions par point (défaut 10 000)")
    args = ap.parse_args()

    header = f"n_in  │ t_avg (µs) │ peak (KB)  (hidden={args.hidden})"
    print(header)
    print("-" * len(header))

    for n_in in [2,4,8,16,32,64,128,256,512,1024,2048]:
        t, mem = bench(n_in, args.hidden, args.repeats)
        print(f"{n_in:4d} │ {t*1e6:10.2f} │ {mem/1024:10.2f}")

if __name__ == "__main__":
    main()
