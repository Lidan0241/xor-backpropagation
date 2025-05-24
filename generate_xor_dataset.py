#!/usr/bin/env python3
"""
Génère un CSV pour la fonction OU-exclusif (XOR) avec un taux d’erreur contrôlé.
Chaque ligne contient : x1, x2, y
  – x1, x2  ∈ {0,1}
  – y est le XOR de x1 et x2, éventuellement inversé selon le bruit.
"""

import csv
import random
import argparse

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", "-n", type=int, default=500,
                    help="nombre total d’exemples à générer")
    ap.add_argument("--noise", "-p", type=float, default=0.0,
                    help="proportion 0.0-1.0 d’étiquettes incorrectes")
    ap.add_argument("--out", "-o", default="xor_dataset.csv",
                    help="fichier de sortie CSV")
    args = ap.parse_args()

    if not 0.0 <= args.noise <= 1.0:
        raise ValueError("--noise doit être compris entre 0 et 1")

    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x1", "x2", "y"])
        for _ in range(args.samples):
            x1, x2 = random.randint(0, 1), random.randint(0, 1)
            y = x1 ^ x2                         # XOR parfait
            if random.random() < args.noise:    # on le retourne avec la proba souhaitée
                y = 1 - y
            wr.writerow([x1, x2, y])

if __name__ == "__main__":
    main()
