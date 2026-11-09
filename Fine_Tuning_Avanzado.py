# Librerias
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Asegurar uso de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# ============================================================
# 1. Cargar y dividir el dataset
# ============================================================

# Cargar ADE corpus v2
raw_dataset = load_dataset("SetFit/ade_corpus_v2_classification")

# Dividir el dataset de entrenamiento en train/val (90/10)
split = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset = {
    "train": split["train"],
    "validation": split["test"],
    "test": raw_dataset["test"]
}

print("Tamaño del conjunto de entrenamiento:", len(dataset["train"]))
print("Tamaño del conjunto de validación:", len(dataset["validation"]))
print("Tamaño del conjunto de prueba:", len(dataset["test"]))

# ============================================================
# 2. Funciones para tokenización y métricas
# ============================================================

# Tokenizador configurable
def tokenize_dataset(model_name, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)
    
    tokenized = {}
    for split_name in dataset:
        tokenized_split = dataset[split_name].map(tokenize_fn, batched=True)
        tokenized_split = tokenized_split.remove_columns(["text"])
        tokenized_split = tokenized_split.rename_column("label", "labels")
        tokenized_split.set_format("torch")
        tokenized[split_name] = tokenized_split
    
    return tokenized, tokenizer

# Métricas de evaluación
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# ======= Utilidades de memoria =======
import os, time
try:
    import psutil  # opcional, para ver RAM del proceso
except Exception:
    psutil = None

def fmt_bytes(n):
    if n is None: return "n/a"
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024

def model_memory_stats(model):
    """Devuelve conteos y tamaño (parámetros + buffers) del modelo."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    # footprint: si existe el método oficial de HF, úsalo; si no, suma params+buffers
    try:
        footprint = int(model.get_memory_footprint())
    except Exception:
        footprint = param_bytes + buffer_bytes
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "footprint_bytes": footprint
    }

def run_with_mem_tracking(fn):
    """
    Ejecuta fn() y mide:
      - Pico de memoria GPU (allocated y reserved)
      - Delta de RAM RSS del proceso
      - Tiempo transcurrido
    """
    rss_before = psutil.Process(os.getpid()).memory_info().rss if psutil else None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        curr_alloc = torch.cuda.memory_allocated()
        curr_reserved = torch.cuda.memory_reserved()
    else:
        peak_alloc = peak_reserved = curr_alloc = curr_reserved = None

    rss_after = psutil.Process(os.getpid()).memory_info().rss if psutil else None

    return out, {
        "elapsed_s": time.time() - t0,
        "gpu_peak_alloc_bytes": peak_alloc,
        "gpu_peak_reserved_bytes": peak_reserved,
        "gpu_curr_alloc_bytes": curr_alloc,
        "gpu_curr_reserved_bytes": curr_reserved,
        "cpu_rss_delta_bytes": (rss_after - rss_before) if (psutil and rss_before is not None) else None
    }


# ============================================================
# 3. Función general de entrenamiento con eval de memoria
# ============================================================
def train_model(model_name, 
                learning_rate=2e-5, 
                batch_size=16, 
                epochs=3, 
                weight_decay=0.01, 
                max_length=128,
                seed=42):
    
    print(f"\n Entrenando modelo: {model_name}")
    print(f"  learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}, weight_decay={weight_decay}")
    
    tokenized_ds, tokenizer = tokenize_dataset(model_name, max_length)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # ---- stats de tamaño del modelo (parámetros/buffers) ----
    size_stats = model_memory_stats(model)
    print(f"  Params totales: {size_stats['total_params']:,} | "
          f"Entrenables: {size_stats['trainable_params']:,} | "
          f"Tamaño modelo: {fmt_bytes(size_stats['footprint_bytes'])}")

    args = TrainingArguments(
        output_dir=f"./results/{model_name.replace('/', '_')}_lr{learning_rate}_bs{batch_size}_ep{epochs}",
        eval_strategy="epoch",   # <--- usa evaluation_strategy
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        compute_metrics=compute_metrics,
    )

    # ---- medir pico de memoria durante TRAIN ----
    train_output, train_mem = run_with_mem_tracking(lambda: trainer.train())
    print(f"  [TRAIN] pico GPU allocated: {fmt_bytes(train_mem['gpu_peak_alloc_bytes'])} | "
          f"reserved: {fmt_bytes(train_mem['gpu_peak_reserved_bytes'])} | "
          f"ΔRAM: {fmt_bytes(train_mem['cpu_rss_delta_bytes'])} | "
          f"tiempo: {train_mem['elapsed_s']:.1f}s")

    # ---- medir pico de memoria durante EVALUATE (en test) ----
    def eval_test():
        return trainer.evaluate(tokenized_ds["test"])
    eval_metrics, eval_mem = run_with_mem_tracking(eval_test)
    print(f"  [EVAL]  pico GPU allocated: {fmt_bytes(eval_mem['gpu_peak_alloc_bytes'])} | "
          f"reserved: {fmt_bytes(eval_mem['gpu_peak_reserved_bytes'])} | "
          f"ΔRAM: {fmt_bytes(eval_mem['cpu_rss_delta_bytes'])} | "
          f"tiempo: {eval_mem['elapsed_s']:.1f}s")

    # Empaquetar estadísticas de memoria
    mem_stats = {
        "model_name": model_name,
        **size_stats,
        "train_gpu_peak_alloc_bytes": train_mem["gpu_peak_alloc_bytes"],
        "train_gpu_peak_reserved_bytes": train_mem["gpu_peak_reserved_bytes"],
        "train_cpu_rss_delta_bytes": train_mem["cpu_rss_delta_bytes"],
        "train_time_s": train_mem["elapsed_s"],
        "eval_gpu_peak_alloc_bytes": eval_mem["gpu_peak_alloc_bytes"],
        "eval_gpu_peak_reserved_bytes": eval_mem["gpu_peak_reserved_bytes"],
        "eval_cpu_rss_delta_bytes": eval_mem["cpu_rss_delta_bytes"],
        "eval_time_s": eval_mem["elapsed_s"],
    }

    return eval_metrics, trainer, tokenized_ds, mem_stats



"""
# ============================================================
# 3. Función general de entrenamiento general
# ============================================================

def train_model(model_name, 
                learning_rate=2e-5, 
                batch_size=16, 
                epochs=3, 
                weight_decay=0.01, 
                max_length=128,
                seed=42):
    
    print(f"\n Entrenando modelo: {model_name}")
    print(f"  learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}, weight_decay={weight_decay}")
    
    tokenized_ds, tokenizer = tokenize_dataset(model_name, max_length)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    args = TrainingArguments(
        output_dir=f"./results/{model_name.replace('/', '_')}_lr{learning_rate}_bs{batch_size}_ep{epochs}",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(tokenized_ds["test"])
    
    return eval_metrics, trainer, tokenized_ds

"""

# ============================================================
# 4. Matriz de Confusión
# ============================================================

def plot_confusion_matrix(trainer, tokenized_ds, model_name):
    preds_output = trainer.predict(tokenized_ds["test"])
    y_true = preds_output.label_ids
    y_pred = np.argmax(preds_output.predictions, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No ADE", "ADE"])
    
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.show()


# ============================================================
# 5. Ejecución de pruebas manuales
# ============================================================

# Ejemplo: entrenamiento con BERT base
eval_bert, trainer_bert, tok_bert = train_model("bert-base-uncased")


# Matriz de confusión modelo BERT base
plot_confusion_matrix(trainer_bert, tok_bert, "BERT base")

# Ejemplo: entrenamiento con BioBERT
eval_biobert, trainer_biobert, tok_biobert = train_model("dmis-lab/biobert-base-cased-v1.1")
plot_confusion_matrix(trainer_biobert, tok_biobert, "BioBERT")

df_results = pd.DataFrame([
    {"Modelo": "BERT base", "Accuracy": eval_bert["eval_accuracy"], "F1": eval_bert["eval_f1"]},
    {"Modelo": "BioBERT", "Accuracy": eval_biobert["eval_accuracy"], "F1": eval_biobert["eval_f1"]}
])

print("\n Comparación de resultados:")
display(df_results)

df_results.set_index("Modelo").plot(kind="bar", figsize=(8, 4), ylim=(0, 1), rot=0)
plt.title("Desempeño en ADE Corpus V2")
plt.ylabel("Puntaje")
plt.grid(axis="y")
plt.show()

preds_output = trainer_biobert.predict(tok_biobert["test"])
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)
texts = dataset["test"]["text"]

wrong_idxs = np.where(y_true != y_pred)[0]
print(f"\n Ejemplos mal clasificados por BioBERT: {len(wrong_idxs)}")

import random
random.seed(42)
sample_wrong = random.sample(list(wrong_idxs), 5)

for idx in sample_wrong:
    print(f"\n Índice: {idx}")
    print("Texto:", texts[idx])
    print(f" Real: {y_true[idx]} |  Predicho: {y_pred[idx]}")

# BERT base
eval_bert, trainer_bert, tok_bert, mem_bert = train_model("bert-base-uncased")

# BioBERT
eval_biobert, trainer_biobert, tok_biobert, mem_biobert = train_model("dmis-lab/biobert-base-cased-v1.1")

