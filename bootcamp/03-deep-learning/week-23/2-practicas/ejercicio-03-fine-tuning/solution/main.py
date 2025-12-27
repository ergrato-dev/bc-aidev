# ============================================
# EJERCICIO 03: Estrategias de Fine-tuning
# SOLUCIÓN COMPLETA
# ============================================

print("=== Ejercicio 03: Estrategias de Fine-tuning ===\n")

# ============================================
# PASO 1: Configuración Inicial
# ============================================
print("--- Paso 1: Configuración ---")

import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# Cargar modelo
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
print(f"Modelo cargado: ResNet18")

print()

# ============================================
# PASO 2: Identificar Grupos de Capas
# ============================================
print("--- Paso 2: Grupos de Capas ---")


def get_layer_groups(model):
    """Agrupa las capas de ResNet para fine-tuning."""
    groups = {
        "stem": [model.conv1, model.bn1],
        "layer1": [model.layer1],
        "layer2": [model.layer2],
        "layer3": [model.layer3],
        "layer4": [model.layer4],
        "head": [model.fc],
    }
    return groups


layer_groups = get_layer_groups(model)
for name, layers in layer_groups.items():
    params = sum(p.numel() for layer in layers for p in layer.parameters())
    print(f"{name}: {params:,} parámetros")

print()

# ============================================
# PASO 3: Congelar por Grupos
# ============================================
print("--- Paso 3: Congelar por Grupos ---")


def freeze_group(group_layers):
    """Congela un grupo de capas."""
    for layer in group_layers:
        for param in layer.parameters():
            param.requires_grad = False


def unfreeze_group(group_layers):
    """Descongela un grupo de capas."""
    for layer in group_layers:
        for param in layer.parameters():
            param.requires_grad = True


def count_trainable(model):
    """Cuenta parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Congelar todo excepto head
for name, layers in layer_groups.items():
    if name != "head":
        freeze_group(layers)

print(f"Parámetros entrenables: {count_trainable(model):,}")

print()

# ============================================
# PASO 4: Discriminative Learning Rates
# ============================================
print("--- Paso 4: Discriminative LR ---")


def get_optimizer_with_discriminative_lr(
    model, layer_groups, base_lr=1e-3, lr_mult=0.1
):
    """
    Crea optimizer con LR discriminativos.
    """
    param_groups = []
    group_names = ["head", "layer4", "layer3", "layer2", "layer1", "stem"]

    for i, name in enumerate(group_names):
        if name not in layer_groups:
            continue

        layers = layer_groups[name]
        lr = base_lr * (lr_mult**i)

        params = []
        for layer in layers:
            params.extend([p for p in layer.parameters() if p.requires_grad])

        if params:
            param_groups.append({"params": params, "lr": lr, "name": name})
            print(f"{name}: lr = {lr:.2e}")

    return torch.optim.Adam(param_groups)


# Descongelar todo para discriminative LR
for layers in layer_groups.values():
    unfreeze_group(layers)

optimizer = get_optimizer_with_discriminative_lr(model, layer_groups)

print()

# ============================================
# PASO 5: Gradual Unfreezing
# ============================================
print("--- Paso 5: Gradual Unfreezing ---")


class GradualUnfreezer:
    """Gestiona el descongelamiento gradual de capas."""

    def __init__(self, model, layer_groups, unfreeze_schedule):
        self.model = model
        self.layer_groups = layer_groups
        self.schedule = unfreeze_schedule

        # Inicialmente congelar todo excepto head
        for name, layers in layer_groups.items():
            if name != "head":
                freeze_group(layers)

    def step(self, epoch):
        """Ejecuta descongelamiento según schedule."""
        if epoch in self.schedule:
            groups_to_unfreeze = self.schedule[epoch]
            for group_name in groups_to_unfreeze:
                if group_name in self.layer_groups:
                    unfreeze_group(self.layer_groups[group_name])
                    print(f"Epoch {epoch}: Descongelado {group_name}")


# Ejemplo de schedule
schedule = {
    0: ["head"],
    2: ["layer4"],
    4: ["layer3"],
    6: ["layer2"],
}

# Recargar modelo para demo
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
layer_groups = get_layer_groups(model)
unfreezer = GradualUnfreezer(model, layer_groups, schedule)

# Simular epochs
for epoch in range(8):
    unfreezer.step(epoch)
    print(f"  Trainable: {count_trainable(model):,}")

print()

# ============================================
# PASO 6: Warmup + Cosine Annealing
# ============================================
print("--- Paso 6: Warmup + Cosine Annealing ---")

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def create_scheduler_with_warmup(optimizer, warmup_epochs, total_epochs):
    """Crea scheduler con warmup + cosine annealing."""
    warmup = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )

    cosine = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    return scheduler


# Crear optimizer simple para demo
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = create_scheduler_with_warmup(optimizer, warmup_epochs=3, total_epochs=20)

# Visualizar LR schedule
print("LR Schedule:")
for epoch in range(20):
    lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch:2d}: lr = {lr:.2e}")
    scheduler.step()

print()

# ============================================
# RESUMEN
# ============================================
print("=== Resumen ===")
print("- Discriminative LR: Capas profundas aprenden más lento")
print("- Gradual Unfreezing: Descongelar progresivamente")
print("- Warmup: Comenzar con LR bajo, subir gradualmente")
print("- Cosine Annealing: Decaimiento suave del LR")
