"""
Proyecto: Entrenador Optimizado
===============================

Implementa un pipeline de entrenamiento completo con todas las técnicas
de optimización: AdamW, OneCycleLR, inicialización, callbacks, checkpoints.

Meta: >80% accuracy en CIFAR-10
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ============================================
# CONFIGURACIÓN
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

config = {
    "lr": 0.001,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "scheduler": "onecycle",
    "max_lr": 0.01,
    "patience": 7,
    "grad_clip": 1.0,
    "epochs": 30,
    "batch_size": 64,
}


# ============================================
# MODELO CNN
# ============================================
class SimpleCNN(nn.Module):
    """CNN simple para CIFAR-10."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ============================================
# CALLBACKS
# ============================================
class EarlyStopping:
    """Detiene entrenamiento si no hay mejora."""

    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class MetricsLogger:
    """Registra métricas de entrenamiento."""

    def __init__(self):
        self.history = {}

    def log(self, metrics: dict):
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)

    def plot(self, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(self.history.get("train_loss", []), label="Train")
        axes[0].plot(self.history.get("val_loss", []), label="Val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(self.history.get("train_acc", []), label="Train")
        axes[1].plot(self.history.get("val_acc", []), label="Val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning Rate
        axes[2].plot(self.history.get("lr", []))
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


# ============================================
# OPTIMIZED TRAINER - TODO: IMPLEMENTAR
# ============================================
class OptimizedTrainer:
    """
    Pipeline de entrenamiento optimizado.

    Integra: optimizador, scheduler, inicialización,
    early stopping, checkpoints, gradient clipping.
    """

    def __init__(self, model, config, train_loader):
        """
        Inicializa el trainer.

        Args:
            model: Red neuronal
            config: Diccionario de configuración
            train_loader: DataLoader de entrenamiento (para OneCycleLR)
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # TODO: Inicializar pesos del modelo
        # Hint: self.init_weights()

        # TODO: Crear optimizador (AdamW)
        # Hint: self.optimizer = optim.AdamW(...)
        self.optimizer = None  # TODO: Implementar

        # TODO: Crear scheduler (OneCycleLR o CosineAnnealing)
        # Hint: Para OneCycleLR necesitas steps_per_epoch=len(train_loader)
        self.scheduler = None  # TODO: Implementar
        self.step_scheduler_per_batch = False  # True para OneCycleLR

        # TODO: Crear criterion
        self.criterion = None  # TODO: nn.CrossEntropyLoss()

        # TODO: Inicializar callbacks
        self.early_stopping = None  # TODO: EarlyStopping(patience=config['patience'])
        self.logger = MetricsLogger()

        # Estado
        self.best_val_acc = 0.0
        self.current_epoch = 0

    def init_weights(self):
        """
        Inicializa pesos con He (Kaiming) para ReLU.

        TODO: Implementar inicialización para:
        - Conv2d: kaiming_normal_, mode='fan_out'
        - BatchNorm: weight=1, bias=0
        - Linear: kaiming_normal_
        """
        # TODO: Implementar
        # for module in self.model.modules():
        #     if isinstance(module, nn.Conv2d):
        #         ...
        pass

    def train_epoch(self, train_loader):
        """
        Entrena una época.

        TODO: Implementar:
        1. model.train()
        2. Loop por batches
        3. Forward pass
        4. Backward pass
        5. Gradient clipping
        6. Optimizer step
        7. Scheduler step (si step_per_batch)

        Returns:
            tuple: (loss_promedio, accuracy)
        """
        # TODO: Implementar
        train_loss = 0.0
        correct = 0
        total = 0

        # self.model.train()
        # for x, y in train_loader:
        #     ...

        return train_loss, correct / max(total, 1)

    def validate(self, val_loader):
        """
        Evalúa en conjunto de validación.

        TODO: Implementar:
        1. model.eval()
        2. torch.no_grad()
        3. Loop por batches
        4. Calcular loss y accuracy

        Returns:
            tuple: (loss_promedio, accuracy)
        """
        # TODO: Implementar
        val_loss = 0.0
        correct = 0
        total = 0

        # self.model.eval()
        # with torch.no_grad():
        #     ...

        return val_loss, correct / max(total, 1)

    def fit(self, train_loader, val_loader, epochs):
        """
        Loop principal de entrenamiento.

        TODO: Implementar:
        1. Loop por épocas
        2. train_epoch()
        3. validate()
        4. Scheduler step (si no es per_batch)
        5. Logger
        6. Checkpoint si mejora
        7. Early stopping

        Returns:
            dict: Historial de métricas
        """
        # TODO: Implementar
        # for epoch in range(epochs):
        #     train_loss, train_acc = self.train_epoch(train_loader)
        #     val_loss, val_acc = self.validate(val_loader)
        #     ...

        return self.logger.history

    def save_checkpoint(self, path):
        """
        Guarda checkpoint completo.

        TODO: Guardar:
        - epoch
        - model_state_dict
        - optimizer_state_dict
        - scheduler_state_dict
        - best_val_acc
        """
        # TODO: Implementar
        # torch.save({...}, path)
        pass

    def load_checkpoint(self, path):
        """
        Carga checkpoint.

        TODO: Cargar y restaurar estado de:
        - model
        - optimizer
        - scheduler
        - epoch
        - best_val_acc
        """
        # TODO: Implementar
        # checkpoint = torch.load(path)
        # ...
        pass


# ============================================
# DATOS
# ============================================
def get_dataloaders(batch_size=64):
    """Prepara dataloaders para CIFAR-10."""
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        "./data", train=False, download=True, transform=transform_test
    )

    # Split train en train/val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size * 2, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, num_workers=2)

    return train_loader, val_loader, test_loader


# ============================================
# MAIN
# ============================================
def main():
    print("=" * 60)
    print("PROYECTO: ENTRENADOR OPTIMIZADO")
    print("=" * 60)

    # Datos
    print("\nCargando datos...")
    train_loader, val_loader, test_loader = get_dataloaders(config["batch_size"])
    print(
        f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}"
    )

    # Modelo
    print("\nCreando modelo...")
    model = SimpleCNN(num_classes=10)
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    # Trainer
    print("\nInicializando trainer...")
    trainer = OptimizedTrainer(model, config, train_loader)

    # Entrenar
    print("\nIniciando entrenamiento...")
    print(
        f'Épocas: {config["epochs"]}, LR: {config["lr"]}, Scheduler: {config["scheduler"]}'
    )
    print("-" * 60)

    history = trainer.fit(train_loader, val_loader, config["epochs"])

    # Visualizar
    print("\nGenerando gráficas...")
    trainer.logger.plot(save_path="training_history.png")

    # Evaluar en test
    print("\nEvaluando en test...")
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Resultado
    print("\n" + "=" * 60)
    if test_acc >= 0.80:
        print(f"✅ ¡OBJETIVO CUMPLIDO! Test accuracy: {test_acc*100:.2f}%")
    else:
        print(
            f"❌ Objetivo no alcanzado. Test accuracy: {test_acc*100:.2f}% (meta: 80%)"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
