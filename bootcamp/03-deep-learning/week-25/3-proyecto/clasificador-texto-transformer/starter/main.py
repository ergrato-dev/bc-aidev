"""
Proyecto: Clasificador de Texto con Transformer
================================================

Implementa un clasificador de sentimientos usando Transformer Encoder.
Objetivo: Accuracy > 85%

Completa los TODOs para construir el modelo.
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Seed para reproducibilidad
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ============================================
# DATASET SINTÉTICO
# ============================================
class SentimentDataset(Dataset):
    """Dataset sintético de sentimientos."""

    def __init__(
        self, num_samples: int = 1000, seq_len: int = 32, vocab_size: int = 1000
    ):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Generar datos sintéticos
        # Patrón: tokens bajos (1-500) = negativo, tokens altos (500-1000) = positivo
        self.data = []
        self.labels = []

        for _ in range(num_samples):
            label = random.randint(0, 1)
            if label == 0:  # Negativo
                tokens = torch.randint(1, vocab_size // 2, (seq_len,))
            else:  # Positivo
                tokens = torch.randint(vocab_size // 2, vocab_size, (seq_len,))
            # Añadir algo de ruido
            noise_idx = random.sample(range(seq_len), seq_len // 4)
            for idx in noise_idx:
                tokens[idx] = random.randint(1, vocab_size - 1)

            self.data.append(tokens)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ============================================
# POSITIONAL ENCODING
# ============================================
class PositionalEncoding(nn.Module):
    """
    Positional Encoding usando funciones sinusoidales.

    TODO: Implementar el forward pass que añade PE a los embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Crear matriz de positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        # TODO: Implementar
        # 1. Añadir positional encoding a x
        # 2. Aplicar dropout
        pass


# ============================================
# MULTI-HEAD ATTENTION
# ============================================
class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Proyecciones y reshape para multi-head
        Q = (
            self.W_q(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_k(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_v(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output
        attn_output = torch.matmul(attn_weights, V)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        return self.W_o(attn_output)


# ============================================
# FEED-FORWARD NETWORK
# ============================================
class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ============================================
# ENCODER LAYER
# ============================================
class EncoderLayer(nn.Module):
    """Una capa del Transformer Encoder."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention con residual y norm
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward con residual y norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


# ============================================
# TRANSFORMER CLASSIFIER
# ============================================
class TransformerClassifier(nn.Module):
    """
    Clasificador de texto usando Transformer Encoder.

    TODO: Completar el __init__ y forward.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        num_classes: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # TODO: Implementar las capas
        # 1. Embedding layer
        # 2. Positional Encoding
        # 3. Encoder layers (usar nn.ModuleList)
        # 4. Classification head (Linear layer)

        self.embedding = None  # TODO
        self.pos_encoding = None  # TODO
        self.encoder_layers = None  # TODO
        self.classifier = None  # TODO

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) - índices de tokens

        Returns:
            logits: (batch, num_classes)
        """
        # TODO: Implementar forward pass
        # 1. Embedding + scale
        # 2. Positional encoding
        # 3. Pasar por encoder layers
        # 4. Pooling (usar primer token o mean pooling)
        # 5. Classification head
        pass


# ============================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Entrena el modelo por una época.

    TODO: Implementar el loop de entrenamiento.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (tokens, labels) in enumerate(dataloader):
        tokens, labels = tokens.to(device), labels.to(device)

        # TODO: Implementar
        # 1. Zero gradients
        # 2. Forward pass
        # 3. Calcular loss
        # 4. Backward pass
        # 5. Optimizer step
        # 6. Acumular métricas

        pass

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evalúa el modelo.

    TODO: Implementar evaluación.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for tokens, labels in dataloader:
            tokens, labels = tokens.to(device), labels.to(device)

            # TODO: Implementar
            # 1. Forward pass
            # 2. Calcular loss
            # 3. Calcular predictions
            # 4. Acumular métricas

            pass

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


# ============================================
# MAIN
# ============================================
def main():
    # Configuración
    VOCAB_SIZE = 1000
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    D_FF = 512
    NUM_CLASSES = 2
    SEQ_LEN = 32
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Usando dispositivo: {DEVICE}")

    # Crear datasets
    train_dataset = SentimentDataset(
        num_samples=2000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE
    )
    test_dataset = SentimentDataset(
        num_samples=500, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Crear modelo
    model = TransformerClassifier(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        num_classes=NUM_CLASSES,
        max_len=SEQ_LEN,
    ).to(DEVICE)

    print(
        f"\nModelo creado con {sum(p.numel() for p in model.parameters()):,} parámetros"
    )

    # TODO: Crear optimizer y criterion
    # optimizer = ...
    # criterion = ...

    # TODO: Loop de entrenamiento
    # for epoch in range(NUM_EPOCHS):
    #     train_loss, train_acc = train_epoch(...)
    #     test_loss, test_acc = evaluate(...)
    #     print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

    print("\n¡Completa los TODOs para entrenar el modelo!")


if __name__ == "__main__":
    main()
