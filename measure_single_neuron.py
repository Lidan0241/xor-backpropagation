#!/usr/bin/env python3
"""
Mesure empirique de la complexité (temps + mémoire) d'un seul neurone caché
dans le MLP « Perceptron-Multicouche ».

Pour chaque taille d'entrée (2, 4, 8, … , 2048) :
  1. on crée une liste d'entrées aléatoires et la même taille de poids
  2. on exécute N répétitions du couple :
        - forward  : potentiel = Σ x_i * w_i ; s = σ(potentiel)
        - backward : delta = s*(1-s)*upstream_error   (avec erreur amont = 1.0)
  3. on mesure le temps moyen par répétition et le pic mémoire
"""

import random, math, tracemalloc, time, argparse

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def bench(n_inputs: int, repeats: int = 10_000):
    """Retourne (temps_moyen_sec, pic_memoire_octets)"""
    # données fixes pour la série de répétitions
    inputs  = [random.random() for _ in range(n_inputs)]
    weights = [random.random() for _ in range(n_inputs)]
    upstream_error = 1.0

    # préparation mémoire
    tracemalloc.start()
    t0 = time.perf_counter()

    for _ in range(repeats):
        # ------------ forward ------------
        potential = sum(x*w for x, w in zip(inputs, weights))
        s         = sigmoid(potential)

        # ------------ backward -----------
        delta = s * (1 - s) * upstream_error
        # (pas de mise à jour de poids ici -> on se concentre sur le neurone)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed / repeats, peak

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", "-r", type=int, default=10_000,
                    help="répétitions par point (défaut : 10 000)")
    args = ap.parse_args()

    print(f"{'n_inputs':>8} │ {'t_avg (µs)':>10} │ {'peak (KB)':>10}")
    print("-"*34)

    for n in [2,4,8,16,32,64,128,256,512,1024,2048]:
        t_avg, peak = bench(n, args.repeats)
        print(f"{n:8d} │ {t_avg*1e6:10.2f} │ {peak/1024:10.2f}")

if __name__ == "__main__":
    main()
