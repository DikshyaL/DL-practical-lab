import pennylane as qml
from pennylane import numpy as np

# create a quantum device with 2 qubits
dev = qml.device("default.qubit", wires=2)

# define a quantum circuit (this is your "quantum layer")
@qml.qnode(dev)
def circuit(x, weights):
    
    # encode classical data into qubits using rotations
    qml.AngleEmbedding(x, wires=[0, 1])
    
    # apply trainable quantum layers (entanglement + rotations)
    qml.BasicEntanglerLayers(weights, wires=[0, 1])
    
    # measure both qubits → gives 2 outputs
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]


# define full hybrid model
def model(x, weights, W, b):
    
    # pass input through quantum circuit
    q_out = circuit(x, weights)
    
    # convert quantum output to classical vector
    q_out = np.array(q_out)
    
    # apply classical linear layer
    return np.dot(q_out, W) + b


# mean squared error loss
def loss(x, y, weights, W, b):
    
    # get prediction
    pred = model(x, weights, W, b)
    
    # compute squared error
    return (pred - y) ** 2


# sample input (2 features)
x = np.array([0.1, 0.5])

# target output (dummy label)
y = 1.0

# initialize quantum weights (1 layer, 2 qubits)
weights = np.random.randn(1, 2)

# initialize classical weights
W = np.random.randn(2)

# bias term
b = 0.0

# optimizer for quantum weights
opt = qml.GradientDescentOptimizer(stepsize=0.1)

# training loop
for i in range(50):
    
    # update quantum weights using gradient descent
    weights = opt.step(lambda w: loss(x, y, w, W, b), weights)
    
    # print progress every 10 steps
    if i % 10 == 0:
        print(f"Step {i}, Loss: {loss(x, y, weights, W, b)}")


# final prediction
print("Final prediction:", model(x, weights, W, b))