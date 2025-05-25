#!/usr/bin/env python3
"""
Mesure du temps moyen et du pic mémoire pour un passage
avant + perte + rétro-propagation + mise à jour des poids
d’un réseau complet :
      n_inputs  →  n_hidden  →  n_outputs (défaut 1)

Par défaut il reproduit le MLP 2-2-1 du TP, mais on peut
faire varier n_inputs et n_hidden pour vérifier la tendance
O(n_inputs × n_hidden).

Usage :
    python measure_full_network.py
    python measure_full_network.py -m 4          # 2-4-1
    python measure_full_network.py -m 8          # 4-2-1
"""

import argparse, math, random, time, tracemalloc

def relu(x: float) -> float:
    """Activation primitive-récursive : max(0, x)."""
    return x if x > 0 else 0.0

def bench(n_in: int, n_hid: int, repeats: int):
    """Renvoie (temps moyen, pic mémoire)"""
    # --- initialisation (hors chronométrage) ---
    X     = [random.random() for _ in range(n_in)]
    W_ih  = [[random.random() for _ in range(n_hid)] for _ in range(n_in)]
    B_h   = [random.random() for _ in range(n_hid)]

    W_ho  = [random.random() for _ in range(n_hid)]   # n_out = 1
    B_o   = random.random()
    t_out = 1.0                                       # étiquette cible

    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()

    for _ in range(repeats):
        # -------- passe avant --------
        H_pot = [sum(X[i] * W_ih[i][j] for i in range(n_in)) + B_h[j]
                 for j in range(n_hid)]
        H_out = [relu(p) for p in H_pot]

        O_pot = sum(H_out[j] * W_ho[j] for j in range(n_hid)) + B_o
        O_out = relu(O_pot)

        # -------- perte + gradients sortie --------
        relu_prime_o = 1.0 if O_pot > 0 else 0.0
        delta_o = relu_prime_o * (t_out - O_out)

        # -------- rétro-propagation cachée --------
        relu_prime_h = [1.0 if H_pot[j] > 0 else 0.0 for j in range(n_hid)]
        delta_h = [relu_prime_h[j] * delta_o * W_ho[j] for j in range(n_hid)]

        # -------- mise à jour des poids --------
        for j in range(n_hid):
            W_ho[j] += H_out[j] * delta_o            # lr fixé à 1 pour mesurer
        B_o += delta_o

        for i in range(n_in):
            for j in range(n_hid):
                W_ih[i][j] += X[i] * delta_h[j]
        for j in range(n_hid):
            B_h[j] += delta_h[j]

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed / repeats, peak

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--inputs", type=int, default=2,
                    help="nombre d’entrées (défaut 2)")
    ap.add_argument("-m", "--hidden", type=int, default=2,
                    help="neurones cachés (défaut 2)")
    ap.add_argument("-r", "--repeats", type=int, default=10_000,
                    help="répétitions par point (défaut 10 000)")
    args = ap.parse_args()

    header = f"n_in │ t_avg (µs) │ peak (KB)  (hidden={args.hidden})"
    print(header)
    print("-" * len(header))

    for n_in in [args.inputs, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        t, mem = bench(n_in, args.hidden, args.repeats)
        print(f"{n_in:4d} │ {t*1e6:10.2f} │ {mem/1024:10.2f}")

if __name__ == "__main__":
    main()
