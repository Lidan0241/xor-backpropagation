# activation.py
def relu_k(z: float, K: float = 10.0) -> float:
    """ReLU bornée : 0 si z<=0, z si 0<z<K, K si z>=K."""
    if z <= 0:
        return 0.0
    return z if z < K else K

def d_relu_k(z: float, K: float = 10.0) -> float:
    """Dérivée : 0 pour z<=0 ou z>=K, sinon 1."""
    return 1.0 if 0.0 < z < K else 0.0
