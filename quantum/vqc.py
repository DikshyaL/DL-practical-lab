import pennylane as qml
from pennylane import numpy as np

# defining 2 qubits
dev = qml.device("default.qubit", wires = 2)

@qml.qnode(dev)
def circuit(x, weights):
    # encode data
    qml.AngleEmbedding(x, wires=[0,1])
    
    #tarinable layer
    qml.BasicEntanglerLayers(weights, wires=[0,1])
    
    return qml.expval(qml.PauliZ(0))

# random weights

weights = np.random.randn(1,2)

x = np.array([0.1,0.5])

print(circuit(x,weights))