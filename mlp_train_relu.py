#!/usr/bin/env python3
"""
MLP 2-2-1 pour XOR (bruité ou non) avec ReLU bornée (couche cachée)
et sigmoïde (sortie). Hyper-paramètres et borne K dans config.json.
"""

import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json, csv, argparse
from random import uniform, seed
from itertools import product
from math import exp
import layer, learn
import activation      

# ------------------------------------------------------------
def σ(z: float) -> float:      # sigmoïde pour la sortie
    return 1.0 / (1.0 + exp(-z))

def load_csv(path: str):
    bank = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            ex = learn.Learn()
            ex.add_input(int(row["x1"]))
            ex.add_input(int(row["x2"]))
            ex.add_output(int(row["y"]))
            bank.append(ex)
    return bank
# ------------------------------------------------------------
def train(cfg: dict, data):
    ni, nh, no   = cfg["layers"]          # 2-2-1
    η            = cfg["learning_rate"]
    max_epochs   = cfg["epochs"]
    target_mse   = cfg["target_mse"]
    K            = cfg.get("K_relu", 10.0)

    # couches (on gère les biais séparément)
    l_in  = layer.Layer(ni)
    l_hid = layer.Layer(nh)
    l_out = layer.Layer(no)

    # poids + biais
    w_ih = [[uniform(-.5, .5) for _ in range(nh)] for _ in range(ni)]
    b_h  = [uniform(-.5, .5) for _ in range(nh)]
    w_ho = [uniform(-.5, .5) for _ in range(nh)]   # sortie unique
    b_o  = uniform(-.5, .5)

    for epoch in range(1, max_epochs + 1):
        sse = 0.0
        correct = 0
        for ex in data:
            # ------------ forward ------------
            X = [ex.get_input(i) for i in range(ni)]

            h_pot = [sum(X[i] * w_ih[i][j] for i in range(ni)) + b_h[j]
                     for j in range(nh)]
            h_out = [activation.relu_k(p, K) for p in h_pot]

            o_pot = sum(h_out[j] * w_ho[j] for j in range(nh)) + b_o
            o_out = σ(o_pot)

            # ----------- erreur --------------
            t = ex.get_output(0)
            delta_o = (t - o_out) * o_out * (1 - o_out)
            sse += (t - o_out) ** 2
            if round(o_out) == t:
                correct += 1

            # ---------- back-prop cachée -----
            delta_h = [activation.d_relu_k(h_pot[j], K) * delta_o * w_ho[j]
                       for j in range(nh)]

            # ---------- mise à jour ----------
            for j in range(nh):
                w_ho[j] += η * h_out[j] * delta_o
            b_o += η * delta_o

            for i, j in product(range(ni), range(nh)):
                w_ih[i][j] += η * X[i] * delta_h[j]
            for j in range(nh):
                b_h[j] += η * delta_h[j]

        mse = 0.5 * sse / len(data)
        acc = correct / len(data)
        if mse <= target_mse:
            print(f"Époque {epoch:5d} | MSE {mse:.6f} | Acc {acc:.3f}  - seuil atteint")
            break
        if epoch % 500 == 0 or epoch == 1:
            print(f"Époque {epoch:5d} | MSE {mse:.6f} | Acc {acc:.3f}")

# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("dataset")
    ap.add_argument("--seed", type=int, help="graine aléatoire")
    args = ap.parse_args()

    if args.seed is not None:
        seed(args.seed)

    with open(args.config) as fh:
        cfg = json.load(fh)

    data = load_csv(args.dataset)
    train(cfg, data)

if __name__ == "__main__":
    main()
