"""
Entraîne un MLP 2×2×1 sur le XOR (bruité) et trace la courbe de perte.

Usage :
    python train_xor.py --config config.json
"""
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

SIGM = lambda x: 1.0 / (1.0 + np.exp(-x))

class MLP221:
    def __init__(self, in_sz, hid_sz, out_sz, lr):
        np.random.seed(42)  # pour reproductibilité
        self.lr  = lr
        self.w1  = np.random.uniform(-1, 1, (in_sz,  hid_sz))
        self.w2  = np.random.uniform(-1, 1, (hid_sz, out_sz))

    def forward(self, x):
        self.h      = SIGM(x @ self.w1)        # (batch, hid)
        self.y_hat  = SIGM(self.h @ self.w2)   # (batch, 1)
        return self.y_hat

    def backward(self, x, y_true):
        err_out     = (y_true - self.y_hat) * self.y_hat * (1 - self.y_hat)
        grad_w2     = self.h.T @ err_out
        err_hid     = (err_out @ self.w2.T) * self.h * (1 - self.h)
        grad_w1     = x.T @ err_hid

        self.w2 += self.lr * grad_w2
        self.w1 += self.lr * grad_w1

    def fit(self, x, y, epochs):
        losses = []
        for _ in range(epochs):
            self.forward(x)
            loss = np.mean((y - self.y_hat) ** 2)
            losses.append(loss)
            self.backward(x, y)
        return losses

    def predict(self, x):
        return (self.forward(x) > 0.5).astype(int)

def load_csv(path):
    with open(path) as f:
        rd = csv.DictReader(f)
        data = [(int(r["arg0"]), int(r["arg1"]), int(r["xor"])) for r in rd]
    data = np.array(data, dtype=np.float32)
    return data[:, :2], data[:, 2:3]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json", type=str)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    X, y = load_csv(cfg["dataset_path"])

    net = MLP221(cfg["nb_input"], cfg["nb_hidden"], cfg["nb_output"],
                 cfg["learning_rate"])
    losses = net.fit(X, y, cfg["epochs"])

    preds = net.predict(X)
    acc = (preds == y).mean()
    print(f"Accuracy en fin d’apprentissage : {acc*100:.2f} %")

    # courbe de perte
    plt.plot(losses, label="MSE")
    plt.xlabel("Époque")
    plt.ylabel("Loss")
    plt.title("Courbe de perte pendant l’apprentissage")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    #plt.savefig("courbe_perte.png")
    plt.show()
