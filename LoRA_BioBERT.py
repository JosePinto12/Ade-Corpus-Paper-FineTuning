# Librerias
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import random

# Confirmar GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo activo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Cargar dataset
dataset_raw = load_dataset("SetFit/ade_corpus_v2_classification")

# Dividir train en 90/10
split = dataset_raw["train"].train_test_split(test_size=0.1, seed=42)
dataset = {
    "train": split["train"],
    "validation": split["test"],
    "test": dataset_raw["test"]
}

# Tokenizador de BioBERT
model_ckpt = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized = {}
for split_name in dataset:
    split_ds = dataset[split_name].map(tokenize, batched=True)
    split_ds = split_ds.remove_columns(["text", "label_text"])
    split_ds.set_format("torch")
    tokenized[split_name] = split_ds

# Modelo base de clasificación binario
base_model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

# Configuración de LoRA con peft
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)

# Aplicar LoRA al modelo base y moverlo a la GPU
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()
model.to(device)  

# Función para calcular metricas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# Inicialización de hiperparámetros para el entrenamiento del modelo BioBERT
training_args = TrainingArguments(
    output_dir="./lora_bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-4,          
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)

# Creamos el Trainer y entrenamos el modelo 
trainer = Trainer(
    model= model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics
)

trainer.train()

pred_output = trainer.predict(tokenized["test"])
y_true = pred_output.label_ids
y_pred = np.argmax(pred_output.predictions, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No ADE", "ADE"])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", values_format='d')
plt.title("Matriz de Confusión - BioBERT LoRA")
plt.show()

text_test = dataset["test"]["text"]
wrong_idxs = np.where(y_true != y_pred)[0]
print(f"\n Total de ejemplos mal clasificados: {len(wrong_idxs)}")

sample_wrong = random.sample(list(wrong_idxs), 5)
for idx in sample_wrong:
    print(f"\n Índice: {idx}")
    print("Texto:", text_test[idx])
    print(f" Real: {y_true[idx]} |  Predicho: {y_pred[idx]}")

# ===============================
# LoRA + BioBERT con tracking memoria
# ===============================
import os, time, random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score

# ---------- Opcional: RAM del proceso ----------
try:
    import psutil
except Exception:
    psutil = None

# ---------- Utilidades de memoria ----------
def fmt_bytes(n):
    if n is None: return "n/a"
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024

def model_memory_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    trainable_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
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

# ---------- Config reproducibilidad ----------
set_seed(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------- Dispositivo ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo activo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ---------- Dataset y tokenización ----------
dataset_raw = load_dataset("SetFit/ade_corpus_v2_classification")
split = dataset_raw["train"].train_test_split(test_size=0.1, seed=42)
dataset = {
    "train": split["train"],
    "validation": split["test"],
    "test": dataset_raw["test"]
}

model_ckpt = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized = {}
for split_name in dataset:
    split_ds = dataset[split_name].map(tokenize, batched=True)
    # Quitar columnas de texto y asegurar clave 'labels'
    to_remove = [c for c in ["text", "label_text"] if c in split_ds.column_names]
    split_ds = split_ds.remove_columns(to_remove).rename_column("label", "labels")
    split_ds.set_format("torch")
    tokenized[split_name] = split_ds

# ---------- Modelo + LoRA ----------
base_model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(base_model, peft_config)
try:
    model.print_trainable_parameters()
except Exception:
    pass
model.to(device)

# ---------- Métricas ----------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds)}

# ---------- Entrenamiento (con tracking) ----------
training_args = TrainingArguments(
    output_dir="./lora_biobert",
    eval_strategy="epoch",   # <-- corregido
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-4,            # LoRA suele tolerar LR más alta
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
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics
)

# Stats de tamaño del modelo (params/buffers)
size_stats = model_memory_stats(model)
print(f"[LoRA/BioBERT] Params totales: {size_stats['total_params']:,} | "
      f"Entrenables: {size_stats['trainable_params']:,} "
      f"({fmt_bytes(size_stats['trainable_param_bytes'])}) | "
      f"Tamaño modelo cargado: {fmt_bytes(size_stats['footprint_bytes'])}")

# Pico de memoria durante TRAIN
_, train_mem = run_with_mem_tracking(lambda: trainer.train())
print(f"[TRAIN] pico GPU allocated: {fmt_bytes(train_mem['gpu_peak_alloc_bytes'])} | "
      f"reserved: {fmt_bytes(train_mem['gpu_peak_reserved_bytes'])} | "
      f"ΔRAM: {fmt_bytes(train_mem['cpu_rss_delta_bytes'])} | "
      f"tiempo: {train_mem['elapsed_s']:.1f}s")

# Pico de memoria durante EVALUATE (test)
def eval_test():
    return trainer.evaluate(tokenized["test"])
eval_metrics, eval_mem = run_with_mem_tracking(eval_test)
print(f"[EVAL ] pico GPU allocated: {fmt_bytes(eval_mem['gpu_peak_alloc_bytes'])} | "
      f"reserved: {fmt_bytes(eval_mem['gpu_peak_reserved_bytes'])} | "
      f"ΔRAM: {fmt_bytes(eval_mem['cpu_rss_delta_bytes'])} | "
      f"tiempo: {eval_mem['elapsed_s']:.1f}s")

# ---------- Paquete de resultados ----------
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