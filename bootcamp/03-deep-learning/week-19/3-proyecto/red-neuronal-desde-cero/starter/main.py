"""
Proyecto: Red Neuronal desde Cero
=================================
Implementa un MLP completo usando solo NumPy.

⚠️ NO usar TensorFlow, PyTorch, Keras ni ningún framework de DL.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ============================================
# FUNCIONES DE ACTIVACIÓN
# ============================================

def sigmoid(z):
    """
    Función sigmoid.
    
    Args:
        z: array de cualquier shape
    
    Returns:
        activación sigmoid de z
    """
    # TODO: Implementar sigmoid
    # Tip: 1 / (1 + exp(-z)), usar clip para estabilidad
    pass


def sigmoid_derivative(z):
    """
    Derivada de sigmoid.
    
    Args:
        z: pre-activación
    
    Returns:
        derivada de sigmoid evaluada en z
    """
    # TODO: Implementar derivada
    # Tip: sigmoid(z) * (1 - sigmoid(z))
    pass


def relu(z):
    """
    Función ReLU.
    
    Args:
        z: array de cualquier shape
    
    Returns:
        max(0, z)
    """
    # TODO: Implementar ReLU
    pass


def relu_derivative(z):
    """
    Derivada de ReLU.
    
    Args:
        z: pre-activación
    
    Returns:
        1 si z > 0, 0 si z <= 0
    """
    # TODO: Implementar derivada
    pass


# ============================================
# INICIALIZACIÓN
# ============================================

def initialize_parameters(layer_dims):
    """
    Inicializa pesos con He initialization.
    
    Args:
        layer_dims: lista con dimensiones [n_input, n_hidden, ..., n_output]
    
    Returns:
        parameters: dict con W1, b1, W2, b2, etc.
    """
    parameters = {}
    L = len(layer_dims) - 1
    
    for l in range(1, L + 1):
        # TODO: Inicializar W_l y b_l
        # W_l shape: (layer_dims[l], layer_dims[l-1])
        # b_l shape: (layer_dims[l], 1)
        # Usar He: np.random.randn(...) * np.sqrt(2.0 / layer_dims[l-1])
        pass
    
    return parameters


# ============================================
# FORWARD PROPAGATION
# ============================================

def forward_propagation(X, parameters):
    """
    Forward pass completo.
    
    Args:
        X: input (n_features, n_samples)
        parameters: dict con W y b de cada capa
    
    Returns:
        A_L: output final (predicciones)
        cache: dict con Z y A de cada capa (para backprop)
    """
    cache = {'A0': X}
    A = X
    L = len(parameters) // 2  # número de capas
    
    # TODO: Implementar forward pass
    # Para cada capa l:
    #   Z_l = W_l @ A_{l-1} + b_l
    #   A_l = activación(Z_l)  # ReLU para hidden, sigmoid para output
    #   Guardar en cache
    
    # Tip: Para capa final usar sigmoid, para hidden usar relu
    
    return A, cache


# ============================================
# FUNCIÓN DE PÉRDIDA
# ============================================

def compute_loss(Y_hat, Y):
    """
    Binary Cross-Entropy Loss.
    
    Args:
        Y_hat: predicciones (1, n_samples)
        Y: labels reales (1, n_samples)
    
    Returns:
        loss: scalar
    """
    m = Y.shape[1]
    epsilon = 1e-8  # evitar log(0)
    
    # TODO: Implementar BCE
    # loss = -1/m * sum(Y*log(Y_hat) + (1-Y)*log(1-Y_hat))
    
    return None


# ============================================
# BACKWARD PROPAGATION
# ============================================

def backward_propagation(Y_hat, Y, cache, parameters):
    """
    Backward pass - calcula gradientes.
    
    Args:
        Y_hat: predicciones
        Y: labels
        cache: valores de forward pass
        parameters: pesos actuales
    
    Returns:
        gradients: dict con dW1, db1, dW2, db2, etc.
    """
    gradients = {}
    m = Y.shape[1]
    L = len(parameters) // 2
    
    # TODO: Implementar backpropagation
    # 
    # Capa de salida (L):
    #   dZ_L = Y_hat - Y  (derivada de BCE + sigmoid)
    #   dW_L = (1/m) * dZ_L @ A_{L-1}.T
    #   db_L = (1/m) * sum(dZ_L, axis=1, keepdims=True)
    #
    # Capas hidden (l = L-1, ..., 1):
    #   dA_l = W_{l+1}.T @ dZ_{l+1}
    #   dZ_l = dA_l * relu_derivative(Z_l)
    #   dW_l = (1/m) * dZ_l @ A_{l-1}.T
    #   db_l = (1/m) * sum(dZ_l, axis=1, keepdims=True)
    
    return gradients


# ============================================
# ACTUALIZACIÓN DE PESOS
# ============================================

def update_parameters(parameters, gradients, learning_rate):
    """
    Actualiza pesos con gradient descent.
    
    Args:
        parameters: pesos actuales
        gradients: gradientes calculados
        learning_rate: tasa de aprendizaje
    
    Returns:
        parameters: pesos actualizados
    """
    L = len(parameters) // 2
    
    # TODO: Implementar actualización
    # W_l = W_l - learning_rate * dW_l
    # b_l = b_l - learning_rate * db_l
    
    return parameters


# ============================================
# ENTRENAMIENTO
# ============================================

def train(X, Y, layer_dims, epochs=5000, learning_rate=0.1, print_every=500):
    """
    Entrena la red neuronal.
    
    Args:
        X: input data
        Y: labels
        layer_dims: arquitectura
        epochs: número de iteraciones
        learning_rate: tasa de aprendizaje
        print_every: frecuencia de impresión
    
    Returns:
        parameters: pesos entrenados
        losses: historial de pérdida
    """
    # TODO: Implementar loop de entrenamiento
    # 1. Inicializar parámetros
    # 2. Para cada época:
    #    - Forward propagation
    #    - Calcular pérdida
    #    - Backward propagation
    #    - Actualizar parámetros
    #    - Guardar pérdida
    
    parameters = None
    losses = []
    
    return parameters, losses


# ============================================
# PREDICCIÓN
# ============================================

def predict(X, parameters):
    """
    Predice clases para X.
    
    Args:
        X: input data
        parameters: pesos entrenados
    
    Returns:
        predictions: 0 o 1
    """
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
    
    # Crear mesh
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predecir para cada punto del mesh
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T, parameters)
    Z = Z.reshape(xx.shape)
    
    # Graficar
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X[0, Y[0]==0], X[1, Y[0]==0], c='red', s=100, label='Clase 0')
    ax.scatter(X[0, Y[0]==1], X[1, Y[0]==1], c='blue', s=100, label='Clase 1')
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Frontera de Decisión')
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("RED NEURONAL DESDE CERO - PROBLEMA XOR")
    print("=" * 50)
    
    # Dataset XOR
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])
    
    print(f"\nDataset XOR:")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"X:\n{X}")
    print(f"Y: {Y}")
    
    # Arquitectura: 2 -> 4 -> 1
    layer_dims = [2, 4, 1]
    print(f"\nArquitectura: {layer_dims}")
    
    # TODO: Descomentar cuando hayas implementado las funciones
    # print("\nEntrenando...")
    # parameters, losses = train(X, Y, layer_dims, epochs=5000, learning_rate=0.5)
    # 
    # # Evaluar
    # predictions = predict(X, parameters)
    # accuracy = np.mean(predictions == Y) * 100
    # print(f"\nAccuracy: {accuracy:.1f}%")
    # print(f"Predicciones: {predictions.flatten()}")
    # print(f"Esperado:     {Y.flatten()}")
    # 
    # # Visualizar
    # plot_loss(losses)
    # plot_decision_boundary(X, Y, parameters)
    
    print("\n¡Implementa las funciones TODO para entrenar la red!")
