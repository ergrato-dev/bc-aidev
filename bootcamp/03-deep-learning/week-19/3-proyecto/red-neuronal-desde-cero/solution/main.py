"""
Proyecto: Red Neuronal desde Cero - SOLUCIÓN
=============================================
MLP completo usando solo NumPy.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ============================================
# FUNCIONES DE ACTIVACIÓN
# ============================================

def sigmoid(z):
    """Función sigmoid."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z):
    """Derivada de sigmoid."""
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """Función ReLU."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivada de ReLU."""
    return (z > 0).astype(float)


# ============================================
# INICIALIZACIÓN
# ============================================

def initialize_parameters(layer_dims):
    """Inicializa pesos con He initialization."""
    parameters = {}
    L = len(layer_dims) - 1
    
    for l in range(1, L + 1):
        parameters[f'W{l}'] = np.random.randn(
            layer_dims[l], layer_dims[l-1]
        ) * np.sqrt(2.0 / layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
    return parameters


# ============================================
# FORWARD PROPAGATION
# ============================================

def forward_propagation(X, parameters):
    """Forward pass completo."""
    cache = {'A0': X}
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        A_prev = A
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        
        Z = np.dot(W, A_prev) + b
        cache[f'Z{l}'] = Z
        
        # ReLU para hidden layers, sigmoid para output
        if l == L:
            A = sigmoid(Z)
        else:
            A = relu(Z)
        
        cache[f'A{l}'] = A
    
    return A, cache


# ============================================
# FUNCIÓN DE PÉRDIDA
# ============================================

def compute_loss(Y_hat, Y):
    """Binary Cross-Entropy Loss."""
    m = Y.shape[1]
    epsilon = 1e-8
    
    loss = -1/m * np.sum(
        Y * np.log(Y_hat + epsilon) + 
        (1 - Y) * np.log(1 - Y_hat + epsilon)
    )
    
    return np.squeeze(loss)


# ============================================
# BACKWARD PROPAGATION
# ============================================

def backward_propagation(Y_hat, Y, cache, parameters):
    """Backward pass - calcula gradientes."""
    gradients = {}
    m = Y.shape[1]
    L = len(parameters) // 2
    
    # Capa de salida
    dZ = Y_hat - Y  # derivada de BCE + sigmoid combinada
    
    for l in reversed(range(1, L + 1)):
        A_prev = cache[f'A{l-1}']
        
        gradients[f'dW{l}'] = (1/m) * np.dot(dZ, A_prev.T)
        gradients[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        if l > 1:
            W = parameters[f'W{l}']
            Z_prev = cache[f'Z{l-1}']
            dA_prev = np.dot(W.T, dZ)
            dZ = dA_prev * relu_derivative(Z_prev)
    
    return gradients


# ============================================
# ACTUALIZACIÓN DE PESOS
# ============================================

def update_parameters(parameters, gradients, learning_rate):
    """Actualiza pesos con gradient descent."""
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}']
    
    return parameters


# ============================================
# ENTRENAMIENTO
# ============================================

def train(X, Y, layer_dims, epochs=5000, learning_rate=0.1, print_every=500):
    """Entrena la red neuronal."""
    parameters = initialize_parameters(layer_dims)
    losses = []
    
    for epoch in range(epochs):
        # Forward
        Y_hat, cache = forward_propagation(X, parameters)
        
        # Loss
        loss = compute_loss(Y_hat, Y)
        losses.append(loss)
        
        # Backward
        gradients = backward_propagation(Y_hat, Y, cache, parameters)
        
        # Update
        parameters = update_parameters(parameters, gradients, learning_rate)
        
        # Print
        if epoch % print_every == 0:
            print(f"Época {epoch:5d} | Loss: {loss:.6f}")
    
    return parameters, losses


# ============================================
# PREDICCIÓN
# ============================================

def predict(X, parameters):
    """Predice clases para X."""
    Y_hat, _ = forward_propagation(X, parameters)
    predictions = (Y_hat > 0.5).astype(int)
    return predictions


# ============================================
# VISUALIZACIÓN
# ============================================

def plot_loss(losses):
    """Grafica la curva de pérdida."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Época')
    plt.ylabel('Loss (BCE)')
    plt.title('Curva de Aprendizaje')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_decision_boundary(X, Y, parameters):
    """Grafica la frontera de decisión."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T, parameters)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X[0, Y[0]==0], X[1, Y[0]==0], c='red', s=100, label='Clase 0')
    ax.scatter(X[0, Y[0]==1], X[1, Y[0]==1], c='blue', s=100, label='Clase 1')
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Frontera de Decisión - XOR Resuelto')
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("RED NEURONAL DESDE CERO - SOLUCIÓN")
    print("=" * 50)
    
    # Dataset XOR
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])
    
    print(f"\nDataset XOR:")
    print(f"X:\n{X}")
    print(f"Y: {Y}")
    
    # Arquitectura
    layer_dims = [2, 4, 1]
    print(f"\nArquitectura: {layer_dims}")
    
    # Entrenar
    print("\nEntrenando...")
    parameters, losses = train(X, Y, layer_dims, epochs=5000, learning_rate=0.5)
    
    # Evaluar
    predictions = predict(X, parameters)
    accuracy = np.mean(predictions == Y) * 100
    
    print(f"\n{'=' * 50}")
    print("RESULTADOS")
    print(f"{'=' * 50}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Predicciones: {predictions.flatten()}")
    print(f"Esperado:     {Y.flatten()}")
    
    # Pesos finales
    print(f"\nPesos finales:")
    for key, value in parameters.items():
        print(f"  {key}: shape {value.shape}")
    
    # Visualizar
    plot_loss(losses)
    plot_decision_boundary(X, Y, parameters)
    
    # Bonus: Probar con más datos
    print("\n" + "=" * 50)
    print("BONUS: Dataset Moons")
    print("=" * 50)
    
    try:
        from sklearn.datasets import make_moons
        
        X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
        X_moons = X_moons.T
        Y_moons = y_moons.reshape(1, -1)
        
        print(f"X shape: {X_moons.shape}")
        print(f"Y shape: {Y_moons.shape}")
        
        # Red más grande para datos más complejos
        layer_dims_moons = [2, 8, 4, 1]
        
        print("\nEntrenando en moons...")
        params_moons, losses_moons = train(
            X_moons, Y_moons, layer_dims_moons, 
            epochs=10000, learning_rate=0.5, print_every=2000
        )
        
        preds_moons = predict(X_moons, params_moons)
        acc_moons = np.mean(preds_moons == Y_moons) * 100
        print(f"\nAccuracy en Moons: {acc_moons:.1f}%")
        
        plot_decision_boundary(X_moons, Y_moons, params_moons)
        
    except ImportError:
        print("sklearn no disponible. Instala con: pip install scikit-learn")
