from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("SetFit/ade_corpus_v2_classification")

# Dividir train en train/val
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset = {
    "train": split["train"],
    "validation": split["test"],
    "test": dataset["test"]
}

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized = {}
for split_name in dataset:
    split_ds = dataset[split_name].map(tokenize, batched=True)
    split_ds = split_ds.remove_columns(["text"])
    split_ds = split_ds.rename_column("label", "labels")
    split_ds.set_format("torch")
    tokenized[split_name] = split_ds

from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

# Cargar modelo base
base_model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

# Configurar LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,        # Tipo de tarea: Clasificación de secuencia
    inference_mode=False,              # ¿Se entrena o solo infiere? False: Se va a entrenar el modelo (modo fine-tuning). True: Solo se usará el modelo para inferencia (no se actualizarán los pesos LoRA).
    r=8,                                # rango (menor = más liviano). Representa el rango bajo de la factorización matricial utilizada por LoRA. Valores típicos: 4, 8, 16. Menor r → menos parámetros entrenables, más ligero, pero menos capacidad adaptativa.
    lora_alpha=32,                      # Escala los pesos LoRA con un factor de ganancia. Interpretación: Controla la magnitud de la actualización de LoRA. Una forma de compensar un r pequeño con una escala más alta.
    lora_dropout=0.1,                   # Dropout aplicado solo a las capas LoRA durante el entrenamiento. Ayuda a regularizar y evitar sobreajuste en las adaptaciones de bajo rango.
    bias="none"                         # Controla qué hacer con los términos de sesgo (bias) en las capas del modelo: "none": no aplica ninguna adaptación al bias (recomendado por defecto). "all": adapta todos los términos de bias. "lora_only": adapta bias solo en las capas donde se aplica LoRA.
)

# Aplicar LoRA al modelo base
lora_model = get_peft_model(base_model, peft_config)
lora_model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_model.to(device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

training_args = TrainingArguments(
    output_dir="./lora_bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-4,          # LoRA suele permitir tasas más altas
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics
)

trainer.train()

# Evaluación final
eval_results = trainer.evaluate(tokenized["test"])
print("Resultados en el conjunto de prueba:")
print(eval_results)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Obtener predicciones
pred_output = trainer.predict(tokenized["test"])
y_true = pred_output.label_ids
y_pred = np.argmax(pred_output.predictions, axis=1)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No ADE", "ADE"])

# Visualizar
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", values_format='d')
plt.title("Matriz de Confusión - BERT con LoRA")
plt.show()

# Recuperar textos originales del set de test
text_test = dataset["test"]["text"]

# Índices de errores
wrong_idxs = np.where(y_true != y_pred)[0]
print(f"Total de ejemplos mal clasificados: {len(wrong_idxs)}")

# Muestra aleatoria de 5 errores
import random
random.seed(42)
sample_wrong = random.sample(list(wrong_idxs), 5)

for idx in sample_wrong:
    print(f"\n Índice: {idx}")
    print("Texto:", text_test[idx])
    print(f" Etiqueta real: {y_true[idx]} |  Predicho: {y_pred[idx]}")

# ===============================
# LoRA + BERT con tracking memoria
# ===============================
import os, time
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

# PEFT / LoRA
from peft import LoraConfig, get_peft_model, TaskType

# Opcional para medir RAM del proceso
try:
    import psutil
except Exception:
    psutil = None

# --------------------------------
# Utilidades de medición de memoria
# --------------------------------
def fmt_bytes(n):
    if n is None: return "n/a"
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024

def model_memory_stats(model):
    """Conteo de parámetros y tamaños (params + buffers)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    trainable_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    # footprint aproximado (HF) o fallback a params+buffers
    try:
        footprint = int(model.get_memory_footprint())
    except Exception:
        footprint = param_bytes + buffer_bytes
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "trainable_param_bytes": trainable_param_bytes,
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

# ----------------------
# Dataset y tokenización
# ----------------------
dataset = load_dataset("SetFit/ade_corpus_v2_classification")

# Dividir train en train/val
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset = {
    "train": split["train"],
    "validation": split["test"],
    "test": dataset["test"]
}

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized = {}
for split_name in dataset:
    split_ds = dataset[split_name].map(tokenize, batched=True)
    split_ds = split_ds.remove_columns(["text"])
    split_ds = split_ds.rename_column("label", "labels")
    split_ds.set_format("torch")
    tokenized[split_name] = split_ds

# ----------------------
# Configurar LoRA + modelo
# ----------------------
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)

base_model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)
lora_model = get_peft_model(base_model, peft_config)

# (opcional) imprime resumen de parámetros entrenables
try:
    lora_model.print_trainable_parameters()
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_model.to(device)

# ----------------------
# Métricas
# ----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds)}

# ----------------------
# Entrenamiento con tracking
# ----------------------
training_args = TrainingArguments(
    output_dir="./lora_bert",
    eval_strategy="epoch",  # <-- corregido
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-4,           # LoRA suele tolerar LR más alta
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    seed=42
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics
)

# ---- stats de tamaño del modelo (params/buffers) ----
size_stats = model_memory_stats(lora_model)
print(f"[LoRA/BERT] Params totales: {size_stats['total_params']:,} | "
      f"Entrenables: {size_stats['trainable_params']:,} "
      f"({fmt_bytes(size_stats['trainable_param_bytes'])}) | "
      f"Tamaño modelo cargado: {fmt_bytes(size_stats['footprint_bytes'])}")

# ---- medir pico de memoria durante TRAIN ----
train_out, train_mem = run_with_mem_tracking(lambda: trainer.train())
print(f"[TRAIN] pico GPU allocated: {fmt_bytes(train_mem['gpu_peak_alloc_bytes'])} | "
      f"reserved: {fmt_bytes(train_mem['gpu_peak_reserved_bytes'])} | "
      f"ΔRAM: {fmt_bytes(train_mem['cpu_rss_delta_bytes'])} | "
      f"tiempo: {train_mem['elapsed_s']:.1f}s")

# ---- medir pico de memoria durante EVALUATE (test) ----
def eval_test():
    return trainer.evaluate(tokenized["test"])
eval_metrics, eval_mem = run_with_mem_tracking(eval_test)

print(f"[EVAL ] pico GPU allocated: {fmt_bytes(eval_mem['gpu_peak_alloc_bytes'])} | "
      f"reserved: {fmt_bytes(eval_mem['gpu_peak_reserved_bytes'])} | "
      f"ΔRAM: {fmt_bytes(eval_mem['cpu_rss_delta_bytes'])} | "
      f"tiempo: {eval_mem['elapsed_s']:.1f}s")


# ----------------------
# Paquete de resultados
# ----------------------
mem_stats = {
    "model_name": model_ckpt + " + LoRA",
    "lora_r": peft_config.r,
    "lora_alpha": peft_config.lora_alpha,
    "lora_dropout": peft_config.lora_dropout,
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

print("\nmem_stats =", mem_stats)

