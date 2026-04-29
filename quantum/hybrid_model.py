import numpy as np  # numerical operations (arrays, math)
from qiskit import QuantumCircuit  # builds quantum circuits
from qiskit_aer import AerSimulator  # simulator for running quantum circuits

sim = AerSimulator()  # create quantum simulator backend

# ---------------- QUANTUM LAYER ----------------
def quantum_layer(x, weights):  # function that acts like a neural network layer but quantum
    
    qc = QuantumCircuit(2)  # create a 2-qubit quantum circuit
    
    qc.ry(x[0], 0)  # encode first input feature into qubit 0 using rotation
    qc.ry(x[1], 1)  # encode second input feature into qubit 1 using rotation
    
    qc.ry(weights[0], 0)  # apply trainable rotation on qubit 0 (learnable parameter)
    qc.ry(weights[1], 1)  # apply trainable rotation on qubit 1 (learnable parameter)
    
    qc.cx(0, 1)  # entangle qubit 0 and 1 (this creates quantum correlation)
    
    qc.measure_all()  # measure all qubits (convert quantum state → classical bits)
    
    result = sim.run(qc, shots=1000).result()  # run circuit 1000 times (sampling quantum system)
    counts = result.get_counts()  # get frequency of measurement results
    
    prob_00 = counts.get('00', 0) / 1000  # probability of measuring state |00>
    prob_11 = counts.get('11', 0) / 1000  # probability of measuring state |11>
    
    return np.array([prob_00, prob_11])  # return quantum features as vector


# ---------------- CLASSICAL MODEL ----------------
def model(x, weights, W, b):  # full hybrid model (quantum + classical)
    
    q_out = quantum_layer(x, weights)  # pass input through quantum layer
    
    return np.dot(q_out, W) + b  # classical linear layer (prediction)


# ---------------- LOSS FUNCTION ----------------
def loss(x, y, weights, W, b):  # measures error between prediction and true value
    
    pred = model(x, weights, W, b)  # get model prediction
    
    return (pred - y) ** 2  # squared error (standard regression loss)


# ---------------- DATA ----------------
x = np.array([0.1, 0.5])  # input features (like a data sample)
y = 1.0  # true label (target output)


# ---------------- PARAMETERS ----------------
weights = np.random.randn(2)  # quantum circuit parameters (trainable)
W = np.random.randn(2)  # classical layer weights
b = 0.0  # bias term


# ---------------- TRAINING ----------------
lr = 0.1  # learning rate

for i in range(20):  # training loop (20 steps)
    
    l = loss(x, y, weights, W, b)  # compute current loss
    
    grad = np.zeros_like(weights)  # initialize gradient array
    
    eps = 1e-3  # small value for numerical gradient approximation
    
    for j in range(len(weights)):  # loop over each quantum weight
        
        w_temp = weights.copy()  # copy current weights
        
        w_temp[j] += eps  # slightly increase one weight
        
        grad[j] = (loss(x, y, w_temp, W, b) - l) / eps  # estimate gradient (finite difference)
    
    weights -= lr * grad  # update weights using gradient descent
    
    print(f"Step {i}, Loss: {l}")  # print training progress


# ---------------- FINAL OUTPUT ----------------
print("Final prediction:", model(x, weights, W, b))  # show final model output