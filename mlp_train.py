"""
Entraînement d’un MLP 2-2-1 pour le XOR (bruité ou non)
– lecture des hyper-paramètres dans config.json
– lecture des données dans un CSV généré par generate_xor_dataset.py
Le code ajoute des biais sur la couche cachée et la couche de sortie.
"""

import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json, csv, argparse
from random import uniform, seed
from itertools import product
from math import exp

import layer, learn

# ------------------------------------------------------------
def σ(x: float) -> float:            # sigmoid
    return 1 / (1 + exp(-x))

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
    ni, nh, no       = cfg["layers"]          # 2-2-1
    η                = cfg["learning_rate"]
    max_epochs       = cfg["epochs"]
    target_mse       = cfg["target_mse"]

    # couches (sans biais internes : on gère les biais séparément)
    l_in   = layer.Layer(ni)
    l_hid  = layer.Layer(nh)
    l_out  = layer.Layer(no)

    # poids entrée→cachée et biais cachés
    w_ih   = [[uniform(-.5, .5) for _ in range(nh)] for _ in range(ni)]
    b_h    = [uniform(-.5, .5) for _ in range(nh)]

    # poids cachée→sortie et biais sortie
    w_ho   = [[uniform(-.5, .5) for _ in range(no)] for _ in range(nh)]
    b_o    = [uniform(-.5, .5) for _ in range(no)]

    for epoch in range(1, max_epochs + 1):
        sse = 0.0
        for ex in data:
            # ---------- forward ----------
            x = [ex.get_input(i) for i in range(ni)]

            h_pot = [sum(x[i] * w_ih[i][j] for i in range(ni)) + b_h[j]
                     for j in range(nh)]
            h_out = [σ(p) for p in h_pot]

            o_pot = [sum(h_out[j] * w_ho[j][k] for j in range(nh)) + b_o[k]
                     for k in range(no)]
            o_out = [σ(p) for p in o_pot]

            # ---------- erreur ----------
            t      = [ex.get_output(k) for k in range(no)]
            delta_o = [(t[k] - o_out[k]) * o_out[k] * (1 - o_out[k])
                       for k in range(no)]
            sse    += sum((t[k] - o_out[k])**2 for k in range(no))

            # ---------- back-prop cachée ----------
            delta_h = [h_out[j] * (1 - h_out[j]) *
                       sum(delta_o[k] * w_ho[j][k] for k in range(no))
                       for j in range(nh)]

            # ---------- mise à jour poids & biais ----------
            for j, k in product(range(nh), range(no)):
                w_ho[j][k] += η * h_out[j] * delta_o[k]
            for k in range(no):
                b_o[k]     += η * delta_o[k]

            for i, j in product(range(ni), range(nh)):
                w_ih[i][j] += η * x[i] * delta_h[j]
            for j in range(nh):
                b_h[j]     += η * delta_h[j]

        mse = 0.5 * sse / len(data)
        if mse <= target_mse:
            print(f"Époque {epoch:5d} | MSE {mse:.6f}  - seuil atteint")
            break
        if epoch % 500 == 0 or epoch == 1:
            print(f"Époque {epoch:5d} | MSE {mse:.6f}")

# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("dataset")
    parser.add_argument("--seed", type=int, help="graine aléatoire")
    args = parser.parse_args()

    if args.seed is not None:
        seed(args.seed)

    with open(args.config) as fh:
        cfg = json.load(fh)

    data = load_csv(args.dataset)
    train(cfg, data)

if __name__ == "__main__":
    main()
