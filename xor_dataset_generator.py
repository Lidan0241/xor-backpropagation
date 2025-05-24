"""
Génère un jeu de données XOR bruité.

Usage :
    python xor_dataset_generator.py --samples 1000 --noise 0.25 --out xor_train.csv
"""
import csv
import random
import argparse

def generate(n_samples: int, noise_ratio: float, out_file: str) -> None:
    assert 0.0 <= noise_ratio <= 1.0, "noise must be in [0, 1]"
    rows = []
    for _ in range(n_samples):
        x0, x1 = random.randint(0, 1), random.randint(0, 1)
        y = x0 ^ x1            # XOR pur
        rows.append([x0, x1, y])

    # on inverse `y` pour `noise_ratio·n_samples` exemples
    n_flip = int(noise_ratio * n_samples)
    for idx in random.sample(range(n_samples), n_flip):
        rows[idx][2] ^= 1      # 0→1, 1→0

    with open(out_file, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["arg0", "arg1", "xor"])  # en-tête
        wr.writerows(rows)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--noise",   type=float, default=0.0)
    p.add_argument("--out",     type=str,   default="xor_train.csv")
    args = p.parse_args()
    generate(args.samples, args.noise, args.out)
