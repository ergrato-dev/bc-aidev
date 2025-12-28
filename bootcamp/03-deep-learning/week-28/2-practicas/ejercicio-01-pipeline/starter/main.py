"""
Pipeline End-to-End para Deep Learning
======================================

Este script implementa un pipeline completo y modular
para entrenar modelos de clasificación de imágenes.

Ejecutar:
    python main.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# ============================================
# PASO 1: Configuración del Proyecto
# ============================================
print("=" * 60)
print("PASO 1: Configuración del Proyecto")
print("=" * 60)

# Configuración centralizada del pipeline
# Todas las opciones en un solo lugar para fácil modificación
config = {
    "data": {
        "dataset": "CIFAR10",
        "data_dir": "./data",
        "batch_size": 32,
        "num_workers": 2,
        "val_split": 0.2,
    },
    "model": {
        "architecture": "resnet18",
        "num_classes": 10,
        "pretrained": True,
        "freeze_backbone": False,
    },
    "training": {
        "epochs": 5,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "device": "auto",  # 'auto', 'cuda', 'cpu'
    },
    "output": {"model_path": "./best_model.pth", "log_interval": 50},
}

print(f"Configuración cargada:")
print(f"  - Dataset: {config['data']['dataset']}")
print(f"  - Batch size: {config['data']['batch_size']}")
print(f"  - Epochs: {config['training']['epochs']}")
print(f"  - Learning rate: {config['training']['lr']}")
print()


# ============================================
# PASO 2: Módulo de Datos
# ============================================
print("=" * 60)
print("PASO 2: Módulo de Datos")
print("=" * 60)

# Descomenta las siguientes líneas:

# def get_transforms(train: bool = True) -> transforms.Compose:
#     """
#     Retorna las transformaciones según el modo.
#
#     Args:
#         train: Si True, incluye data augmentation
#
#     Returns:
#         Compose de transformaciones
#     """
#     if train:
#         return transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.4914, 0.4822, 0.4465],
#                 std=[0.2470, 0.2435, 0.2616]
#             )
#         ])
#     else:
#         return transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.4914, 0.4822, 0.4465],
#                 std=[0.2470, 0.2435, 0.2616]
#             )
#         ])


# def get_dataloaders(data_config: dict) -> tuple[DataLoader, DataLoader]:
#     """
#     Crea DataLoaders para entrenamiento y validación.
#
#     Args:
#         data_config: Configuración de datos
#
#     Returns:
#         Tuple de (train_loader, val_loader)
#     """
#     # Descargar y cargar dataset
#     full_dataset = datasets.CIFAR10(
#         root=data_config['data_dir'],
#         train=True,
#         download=True,
#         transform=get_transforms(train=True)
#     )
#
#     # Split train/val
#     val_size = int(len(full_dataset) * data_config['val_split'])
#     train_size = len(full_dataset) - val_size
#
#     train_dataset, val_dataset = random_split(
#         full_dataset, [train_size, val_size],
#         generator=torch.Generator().manual_seed(42)
#     )
#
#     # Crear DataLoaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=data_config['batch_size'],
#         shuffle=True,
#         num_workers=data_config['num_workers'],
#         pin_memory=True
#     )
#
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=data_config['batch_size'],
#         shuffle=False,
#         num_workers=data_config['num_workers'],
#         pin_memory=True
#     )
#
#     print(f"Dataset cargado:")
#     print(f"  - Train samples: {train_size}")
#     print(f"  - Val samples: {val_size}")
#     print(f"  - Batch size: {data_config['batch_size']}")
#
#     return train_loader, val_loader

print("Módulo de datos definido")
print()


# ============================================
# PASO 3: Creación del Modelo
# ============================================
print("=" * 60)
print("PASO 3: Creación del Modelo")
print("=" * 60)

# Descomenta las siguientes líneas:

# def create_model(model_config: dict) -> nn.Module:
#     """
#     Crea y configura el modelo según la configuración.
#
#     Args:
#         model_config: Configuración del modelo
#
#     Returns:
#         Modelo configurado
#     """
#     # Cargar modelo preentrenado
#     if model_config['architecture'] == 'resnet18':
#         weights = 'IMAGENET1K_V1' if model_config['pretrained'] else None
#         model = models.resnet18(weights=weights)
#
#         # Modificar primera capa para CIFAR (32x32 en lugar de 224x224)
#         model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         model.maxpool = nn.Identity()  # Remover maxpool para imágenes pequeñas
#
#         # Modificar capa final
#         num_features = model.fc.in_features
#         model.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(num_features, model_config['num_classes'])
#         )
#     else:
#         raise ValueError(f"Arquitectura no soportada: {model_config['architecture']}")
#
#     # Congelar backbone si se especifica
#     if model_config['freeze_backbone']:
#         for name, param in model.named_parameters():
#             if 'fc' not in name:
#                 param.requires_grad = False
#
#     # Contar parámetros
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#     print(f"Modelo creado: {model_config['architecture']}")
#     print(f"  - Total params: {total_params:,}")
#     print(f"  - Trainable params: {trainable_params:,}")
#     print(f"  - Pretrained: {model_config['pretrained']}")
#
#     return model

print("Función create_model definida")
print()


# ============================================
# PASO 4: Loop de Entrenamiento
# ============================================
print("=" * 60)
print("PASO 4: Loop de Entrenamiento")
print("=" * 60)

# Descomenta las siguientes líneas:

# def train_epoch(
#     model: nn.Module,
#     loader: DataLoader,
#     criterion: nn.Module,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
#     log_interval: int = 50
# ) -> tuple[float, float]:
#     """
#     Entrena el modelo por una época.
#
#     Args:
#         model: Modelo a entrenar
#         loader: DataLoader de entrenamiento
#         criterion: Función de pérdida
#         optimizer: Optimizador
#         device: Dispositivo (cuda/cpu)
#         log_interval: Cada cuántos batches mostrar progreso
#
#     Returns:
#         Tuple de (loss promedio, accuracy)
#     """
#     model.train()
#     total_loss = 0.0
#     correct = 0
#     total = 0
#
#     for batch_idx, (data, target) in enumerate(loader):
#         # Mover datos al dispositivo
#         data, target = data.to(device), target.to(device)
#
#         # Forward pass
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#
#         # Backward pass
#         loss.backward()
#         optimizer.step()
#
#         # Métricas
#         total_loss += loss.item()
#         pred = output.argmax(dim=1)
#         correct += pred.eq(target).sum().item()
#         total += target.size(0)
#
#         # Log de progreso
#         if (batch_idx + 1) % log_interval == 0:
#             print(f"    Batch {batch_idx+1}/{len(loader)}: "
#                   f"Loss={loss.item():.4f}")
#
#     avg_loss = total_loss / len(loader)
#     accuracy = 100.0 * correct / total
#
#     return avg_loss, accuracy

print("Función train_epoch definida")
print()


# ============================================
# PASO 5: Evaluación
# ============================================
print("=" * 60)
print("PASO 5: Evaluación")
print("=" * 60)

# Descomenta las siguientes líneas:

# @torch.no_grad()
# def evaluate(
#     model: nn.Module,
#     loader: DataLoader,
#     criterion: nn.Module,
#     device: torch.device
# ) -> tuple[float, float]:
#     """
#     Evalúa el modelo en un dataset.
#
#     Args:
#         model: Modelo a evaluar
#         loader: DataLoader de evaluación
#         criterion: Función de pérdida
#         device: Dispositivo (cuda/cpu)
#
#     Returns:
#         Tuple de (loss promedio, accuracy)
#     """
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     total = 0
#
#     for data, target in loader:
#         data, target = data.to(device), target.to(device)
#
#         output = model(data)
#         loss = criterion(output, target)
#
#         total_loss += loss.item()
#         pred = output.argmax(dim=1)
#         correct += pred.eq(target).sum().item()
#         total += target.size(0)
#
#     avg_loss = total_loss / len(loader)
#     accuracy = 100.0 * correct / total
#
#     return avg_loss, accuracy

print("Función evaluate definida")
print()


# ============================================
# PASO 6: Pipeline Completo
# ============================================
print("=" * 60)
print("PASO 6: Pipeline Completo")
print("=" * 60)

# Descomenta las siguientes líneas:

# def run_pipeline(config: dict) -> tuple[nn.Module, float]:
#     """
#     Ejecuta el pipeline completo de entrenamiento.
#
#     Args:
#         config: Diccionario de configuración
#
#     Returns:
#         Tuple de (modelo entrenado, mejor accuracy)
#     """
#     print("\n" + "=" * 60)
#     print("INICIANDO PIPELINE DE ENTRENAMIENTO")
#     print("=" * 60 + "\n")
#
#     # 1. Configurar dispositivo
#     if config['training']['device'] == 'auto':
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     else:
#         device = torch.device(config['training']['device'])
#     print(f"Dispositivo: {device}")
#
#     # 2. Cargar datos
#     print("\n--- Cargando datos ---")
#     train_loader, val_loader = get_dataloaders(config['data'])
#
#     # 3. Crear modelo
#     print("\n--- Creando modelo ---")
#     model = create_model(config['model']).to(device)
#
#     # 4. Configurar entrenamiento
#     print("\n--- Configurando entrenamiento ---")
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=config['training']['lr'],
#         weight_decay=config['training']['weight_decay']
#     )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=config['training']['epochs']
#     )
#
#     print(f"  - Optimizer: AdamW")
#     print(f"  - LR: {config['training']['lr']}")
#     print(f"  - Scheduler: CosineAnnealing")
#
#     # 5. Loop de entrenamiento
#     print("\n--- Entrenando ---")
#     best_acc = 0.0
#     history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
#
#     for epoch in range(config['training']['epochs']):
#         print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
#         print("-" * 40)
#
#         # Entrenar
#         train_loss, train_acc = train_epoch(
#             model, train_loader, criterion, optimizer, device,
#             log_interval=config['output']['log_interval']
#         )
#
#         # Evaluar
#         val_loss, val_acc = evaluate(model, val_loader, criterion, device)
#
#         # Actualizar scheduler
#         scheduler.step()
#
#         # Guardar historial
#         history['train_loss'].append(train_loss)
#         history['train_acc'].append(train_acc)
#         history['val_loss'].append(val_loss)
#         history['val_acc'].append(val_acc)
#
#         # Mostrar métricas
#         current_lr = optimizer.param_groups[0]['lr']
#         print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
#         print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
#         print(f"  LR: {current_lr:.6f}")
#
#         # Guardar mejor modelo
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_acc': val_acc,
#                 'config': config
#             }, config['output']['model_path'])
#             print(f"  ✓ Nuevo mejor modelo guardado (Acc: {val_acc:.2f}%)")
#
#     # 6. Resumen final
#     print("\n" + "=" * 60)
#     print("ENTRENAMIENTO COMPLETADO")
#     print("=" * 60)
#     print(f"Mejor Val Accuracy: {best_acc:.2f}%")
#     print(f"Modelo guardado en: {config['output']['model_path']}")
#
#     return model, best_acc

print("Función run_pipeline definida")
print()


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("EJECUCIÓN DEL PIPELINE")
    print("=" * 60)
    print()

    # Descomenta para ejecutar el pipeline completo:
    # model, best_acc = run_pipeline(config)

    # Verificar que funciones están definidas
    print("Para ejecutar el pipeline completo:")
    print("1. Descomenta todas las funciones en los pasos 2-6")
    print("2. Descomenta la línea 'model, best_acc = run_pipeline(config)'")
    print("3. Ejecuta: python main.py")
    print()
    print("El pipeline descargará CIFAR-10 (~170MB) la primera vez.")
