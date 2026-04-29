import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

sim = AerSimulator()

# ---------------- QUANTUM LAYER ----------------
def quantum_layer(x, weights):
    qc = QuantumCircuit(2)

    qc.ry(x[0], 0)
    qc.ry(x[1], 1)

    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)

    qc.cx(0, 1)

    qc.measure_all()

    result = sim.run(qc, shots=500).result()
    counts = result.get_counts()

    p00 = counts.get('00', 0) / 500
    p11 = counts.get('11', 0) / 500

    return np.array([p00, p11])


# ---------------- HYBRID MODEL ----------------
def hybrid_model(x, q_weights, W, b):
    q_out = quantum_layer(x, q_weights)
    return np.dot(q_out, W) + b


# ---------------- CLASSICAL MODEL ----------------
def classical_model(x, W, b):
    return np.dot(x, W) + b


# ---------------- LOSS ----------------
def loss(pred, y):
    return (pred - y) ** 2


# ---------------- DATA ----------------
X = np.array([
    [0.1, 0.5],
    [0.2, 0.4],
    [0.3, 0.7],
    [0.4, 0.9]
])

y = np.array([0.6, 0.5, 0.8, 1.0])


# ---------------- INITIAL PARAMETERS ----------------
q_weights = np.random.randn(2)
W_q = np.random.randn(2)
b_q = 0.0

W_c = np.random.randn(2)
b_c = 0.0


# ---------------- TRAIN + COMPARE ----------------
q_losses = []
c_losses = []

for i in range(len(X)):

    x = X[i]
    target = y[i]

    # ----- quantum model prediction -----
    q_pred = hybrid_model(x, q_weights, W_q, b_q)
    q_losses.append(loss(q_pred, target))

    # ----- classical model prediction -----
    c_pred = classical_model(x, W_c, b_c)
    c_losses.append(loss(c_pred, target))


# ---------------- RESULTS ----------------
print("Quantum model loss:", np.mean(q_losses))
print("Classical model loss:", np.mean(c_losses))