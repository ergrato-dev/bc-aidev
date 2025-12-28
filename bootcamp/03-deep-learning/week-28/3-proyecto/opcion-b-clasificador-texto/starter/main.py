"""
Clasificador de Texto con Transformers
======================================

Proyecto final de Deep Learning - Opción B: NLP

Objetivo: Clasificar sentimiento de reviews de películas
con accuracy > 85% usando fine-tuning de DistilBERT.

Ejecutar:
    python main.py
"""

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ============================================
# CONFIGURACIÓN
# ============================================

CONFIG = {
    "model": {"name": "distilbert-base-uncased", "num_labels": 2},
    "tokenizer": {"max_length": 256, "padding": "max_length", "truncation": True},
    "training": {
        "output_dir": "./results",
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
    },
}

LABEL_NAMES = ["negative", "positive"]


# ============================================
# TODO 1: CARGA DE DATOS
# ============================================


def load_data(use_subset: bool = False, subset_size: int = 1000):
    """
    Carga el dataset IMDB desde Hugging Face.

    Args:
        use_subset: Si True, usa solo una porción del dataset
        subset_size: Tamaño del subset para desarrollo rápido

    Returns:
        Dataset con splits train y test

    Hints:
        - Usar load_dataset("imdb")
        - Para subset: dataset.select(range(subset_size))
    """
    # TODO: Implementar carga de datos
    # 1. Cargar dataset IMDB
    # 2. Opcionalmente crear subset
    # 3. Mostrar información del dataset
    pass


# ============================================
# TODO 2: TOKENIZACIÓN
# ============================================


def get_tokenizer(model_name: str):
    """
    Carga el tokenizer del modelo.

    Args:
        model_name: Nombre del modelo en Hugging Face

    Returns:
        Tokenizer configurado
    """
    # TODO: Cargar tokenizer
    # Usar AutoTokenizer.from_pretrained()
    pass


def tokenize_function(examples, tokenizer, config: dict):
    """
    Tokeniza los ejemplos del dataset.

    Args:
        examples: Batch de ejemplos con campo 'text'
        tokenizer: Tokenizer a usar
        config: Configuración de tokenización

    Returns:
        Ejemplos tokenizados

    Hints:
        - Tokenizar examples["text"]
        - Usar padding, truncation, max_length del config
    """
    # TODO: Implementar tokenización
    pass


def prepare_datasets(dataset, tokenizer, config: dict):
    """
    Prepara los datasets tokenizados.

    Args:
        dataset: Dataset original
        tokenizer: Tokenizer
        config: Configuración

    Returns:
        Datasets tokenizados listos para entrenar
    """
    # TODO: Aplicar tokenización a todo el dataset
    # 1. Usar dataset.map() con tokenize_function
    # 2. Remover columna 'text' original
    # 3. Configurar formato para PyTorch
    pass


# ============================================
# TODO 3: MODELO
# ============================================


def load_model(config: dict):
    """
    Carga el modelo preentrenado para clasificación.

    Args:
        config: Configuración del modelo

    Returns:
        Modelo configurado

    Hints:
        - Usar AutoModelForSequenceClassification.from_pretrained()
        - Especificar num_labels
    """
    # TODO: Cargar modelo
    pass


# ============================================
# TODO 4: MÉTRICAS
# ============================================


def compute_metrics(eval_pred):
    """
    Calcula métricas de evaluación.

    Args:
        eval_pred: Tuple de (logits, labels)

    Returns:
        Diccionario con métricas

    Hints:
        - Usar evaluate.load("accuracy") y evaluate.load("f1")
        - Convertir logits a predicciones con argmax
    """
    # TODO: Implementar cálculo de métricas
    # 1. Extraer logits y labels
    # 2. Convertir logits a predicciones
    # 3. Calcular accuracy y f1
    pass


# ============================================
# TODO 5: ENTRENAMIENTO
# ============================================


def create_trainer(model, train_dataset, eval_dataset, tokenizer, config: dict):
    """
    Crea el Trainer de Hugging Face.

    Args:
        model: Modelo a entrenar
        train_dataset: Dataset de entrenamiento
        eval_dataset: Dataset de evaluación
        tokenizer: Tokenizer
        config: Configuración de entrenamiento

    Returns:
        Trainer configurado

    Hints:
        - Crear TrainingArguments con config
        - Añadir EarlyStoppingCallback
    """
    # TODO: Crear y configurar Trainer
    # 1. Crear TrainingArguments
    # 2. Crear Trainer con callbacks
    pass


def train(trainer):
    """
    Ejecuta el entrenamiento.

    Args:
        trainer: Trainer configurado

    Returns:
        Resultados del entrenamiento
    """
    # TODO: Ejecutar entrenamiento
    # trainer.train()
    pass


# ============================================
# TODO 6: EVALUACIÓN E INFERENCIA
# ============================================


def evaluate_model(trainer, test_dataset):
    """
    Evalúa el modelo en el test set.

    Args:
        trainer: Trainer con modelo entrenado
        test_dataset: Dataset de test

    Returns:
        Métricas de evaluación
    """
    # TODO: Evaluar modelo
    # trainer.evaluate(test_dataset)
    pass


def predict_sentiment(text: str, model, tokenizer, config: dict) -> dict:
    """
    Predice el sentimiento de un texto.

    Args:
        text: Texto a clasificar
        model: Modelo entrenado
        tokenizer: Tokenizer
        config: Configuración

    Returns:
        Diccionario con label y score

    Hints:
        - Tokenizar el texto
        - Hacer forward pass
        - Aplicar softmax para probabilidades
    """
    # TODO: Implementar predicción
    # 1. Tokenizar texto
    # 2. Mover a device del modelo
    # 3. Forward pass
    # 4. Convertir logits a probabilidades
    # 5. Retornar label y score
    pass


def run_inference_examples(model, tokenizer, config: dict):
    """
    Ejecuta inferencia en ejemplos de prueba.
    """
    examples = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money. Don't watch it.",
        "It was okay, nothing special but not bad either.",
        "One of the best movies I've ever seen. Highly recommend!",
        "Boring and predictable. The acting was awful.",
    ]

    print("\n--- Ejemplos de Inferencia ---")
    for text in examples:
        result = predict_sentiment(text, model, tokenizer, config)
        if result:
            print(f"\nTexto: {text[:50]}...")
            print(f"Predicción: {result['label']} (score: {result['score']:.4f})")


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================


def main():
    """Función principal del proyecto."""
    print("=" * 60)
    print("CLASIFICADOR DE SENTIMIENTO - IMDB")
    print("Fine-tuning con DistilBERT")
    print("=" * 60)

    # TODO: Descomentar cuando las funciones estén implementadas

    # # 1. Cargar datos
    # print("\n--- Cargando datos ---")
    # dataset = load_data(use_subset=False)
    #
    # # 2. Preparar tokenizer y datos
    # print("\n--- Tokenizando ---")
    # tokenizer = get_tokenizer(CONFIG['model']['name'])
    # tokenized_datasets = prepare_datasets(
    #     dataset, tokenizer, CONFIG['tokenizer']
    # )
    #
    # # 3. Cargar modelo
    # print("\n--- Cargando modelo ---")
    # model = load_model(CONFIG['model'])
    #
    # # 4. Crear trainer
    # print("\n--- Configurando entrenamiento ---")
    # trainer = create_trainer(
    #     model,
    #     tokenized_datasets['train'],
    #     tokenized_datasets['test'],
    #     tokenizer,
    #     CONFIG['training']
    # )
    #
    # # 5. Entrenar
    # print("\n--- Entrenando ---")
    # train(trainer)
    #
    # # 6. Evaluar
    # print("\n--- Evaluando ---")
    # metrics = evaluate_model(trainer, tokenized_datasets['test'])
    #
    # # 7. Guardar modelo
    # print("\n--- Guardando modelo ---")
    # trainer.save_model('./model')
    # tokenizer.save_pretrained('./model')
    #
    # # 8. Inferencia
    # run_inference_examples(model, tokenizer, CONFIG['tokenizer'])
    #
    # # 9. Resumen
    # print("\n" + "=" * 60)
    # print("RESULTADOS FINALES")
    # print("=" * 60)
    # print(f"Test Accuracy: {metrics['eval_accuracy']*100:.2f}%")
    # print(f"Test F1-Score: {metrics['eval_f1']:.4f}")
    # print("Modelo guardado en: ./model")

    print("\nPara ejecutar el proyecto:")
    print("1. Implementa todos los TODOs")
    print("2. Descomenta el código en main()")
    print("3. Ejecuta: python main.py")


if __name__ == "__main__":
    main()
