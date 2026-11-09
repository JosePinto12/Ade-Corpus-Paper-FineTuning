# Librerias
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" Dispositivo activo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

dataset = load_dataset("ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")
data = dataset["train"]

# Filtrar ejemplos con anotaciones vacías
def has_valid_entities(example):
    return (
        "drug" in example["indexes"]
        and "effect" in example["indexes"]
        and example["indexes"]["drug"]["start_char"]
        and example["indexes"]["effect"]["start_char"]
    )

data = data.filter(has_valid_entities)

label_list = ["O", "B-DRUG", "I-DRUG", "B-EFFECT", "I-EFFECT"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

def create_bio_labels(example, tokenizer):
    text = example["text"]
    drug_start = example["indexes"]["drug"]["start_char"][0]
    drug_end = example["indexes"]["drug"]["end_char"][0]
    effect_start = example["indexes"]["effect"]["start_char"][0]
    effect_end = example["indexes"]["effect"]["end_char"][0]

    tokens = tokenizer(text, return_offsets_mapping=True, truncation=True, padding="max_length", max_length=128)
    labels = ["O"] * len(tokens["input_ids"])

    for i, (start, end) in enumerate(tokens["offset_mapping"]):
        if start is None or end is None:
            continue
        if drug_start <= start < drug_end:
            labels[i] = "B-DRUG" if start == drug_start else "I-DRUG"
        elif effect_start <= start < effect_end:
            labels[i] = "B-EFFECT" if start == effect_start else "I-EFFECT"

    tokens["labels"] = [label2id.get(l, 0) for l in labels]
    return tokens

def decode_predictions_clean(example, model, tokenizer):
    model.eval()
    inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    labels = [id2label[p] for p in preds]

    annotated = []
    drug_chunks = []
    effect_chunks = []
    current_entity = ""
    current_label = ""

    def clean_token(t):
        return t.replace("##", "") if t.startswith("##") else t

    for token, label in zip(tokens, labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        if label.startswith("B-"):
            if current_entity:
                if current_label == "DRUG":
                    drug_chunks.append(current_entity)
                elif current_label == "EFFECT":
                    effect_chunks.append(current_entity)
            current_entity = clean_token(token)
            current_label = label[2:]
            annotated.append(f"<{current_label}>{clean_token(token)}</{current_label}>")
        elif label.startswith("I-") and current_label == label[2:]:
            current_entity += clean_token(token)
            annotated.append(f"<{current_label}>{clean_token(token)}</{current_label}>")
        else:
            if current_entity:
                if current_label == "DRUG":
                    drug_chunks.append(current_entity)
                elif current_label == "EFFECT":
                    effect_chunks.append(current_entity)
            current_entity = ""
            current_label = ""
            annotated.append(token)

    if current_entity:
        if current_label == "DRUG":
            drug_chunks.append(current_entity)
        elif current_label == "EFFECT":
            effect_chunks.append(current_entity)

    print("\n Texto original:")
    print(example["text"])
    print("\n Texto con etiquetas NER:")
    print(" ".join(annotated))
    print("\n Medicamento extraído:", ", ".join(drug_chunks) if drug_chunks else "[ninguno]")
    print(" Efecto adverso extraído:", ", ".join(effect_chunks) if effect_chunks else "[ninguno]")

model_ckpt_bert = "bert-base-cased"
tokenizer_bert = AutoTokenizer.from_pretrained(model_ckpt_bert)
tokenized_bert = data.map(lambda x: create_bio_labels(x, tokenizer_bert), batched=False)
tokenized_bert.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model_bert = AutoModelForTokenClassification.from_pretrained(
    model_ckpt_bert,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
).to(device)

data_collator = DataCollatorForTokenClassification(tokenizer_bert)

training_args = TrainingArguments(
    output_dir="./results/bert_base",
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none"
)

trainer_bert = Trainer(
    model=model_bert,
    args=training_args,
    train_dataset=tokenized_bert.select(range(5000)),
    eval_dataset=tokenized_bert.select(range(1000)),
    tokenizer=tokenizer_bert,
    data_collator=data_collator
)

trainer_bert.train()

# Evaluación BERT
from torch.utils.data import DataLoader
model_bert.eval()
loader = DataLoader(tokenized_bert.select(range(1000)), batch_size=16)
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()
        outputs = model_bert(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.flatten())

print("\nResultados para BERT Base:")
print(classification_report(all_labels, all_preds, target_names=label_list))

# Matriz de confusión BERT
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list).plot(xticks_rotation=45, cmap="Blues")
plt.title(" Matriz de Confusión - BERT Base")
plt.show()

# Mostrar ejemplos anotados BERT
print("\n Ejemplos predichos por BERT:")
for i in range(10):
    decode_predictions_clean(data[i], model_bert, tokenizer_bert)


model_ckpt_biobert = "dmis-lab/biobert-base-cased-v1.1"
tokenizer_biobert = AutoTokenizer.from_pretrained(model_ckpt_biobert)
tokenized_biobert = data.map(lambda x: create_bio_labels(x, tokenizer_biobert), batched=False)
tokenized_biobert.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model_biobert = AutoModelForTokenClassification.from_pretrained(
    model_ckpt_biobert,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
).to(device)

data_collator = DataCollatorForTokenClassification(tokenizer_biobert)

training_args_biobert = TrainingArguments(
    output_dir="./results/biobert",
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none"
)

trainer_biobert = Trainer(
    model=model_biobert,
    args=training_args_biobert,
    train_dataset=tokenized_biobert.select(range(5000)),
    eval_dataset=tokenized_biobert.select(range(1000)),
    tokenizer=tokenizer_biobert,
    data_collator=data_collator
)

trainer_biobert.train()

# Evaluación BioBERT
model_biobert.eval()
loader = DataLoader(tokenized_biobert.select(range(1000)), batch_size=16)
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()
        outputs = model_biobert(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.flatten())

print("\n Resultados para BioBERT:")
print(classification_report(all_labels, all_preds, target_names=label_list))

# Matriz de confusión BioBERT
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list).plot(xticks_rotation=45, cmap="Greens")
plt.title(" Matriz de Confusión - BioBERT")
plt.show()

# Mostrar ejemplos anotados BioBERT
print("\n Ejemplos predichos por BioBERT:")
for i in range(10):
    decode_predictions_clean(data[i], model_biobert, tokenizer_biobert)

# ===============================
# NER (Token Classification) con tracking de memoria
# ===============================
import os, time, random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments, Trainer, set_seed
)

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

# ---------- Reproducibilidad y dispositivo ----------
set_seed(42)
random.seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" Dispositivo activo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ---------- Carga de dataset ----------
dataset = load_dataset("ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")
data = dataset["train"]

# Filtrar ejemplos con anotaciones válidas
def has_valid_entities(example):
    return (
        "drug" in example["indexes"]
        and "effect" in example["indexes"]
        and example["indexes"]["drug"]["start_char"]
        and example["indexes"]["effect"]["start_char"]
    )
data = data.filter(has_valid_entities)

# ---------- Etiquetas BIO ----------
label_list = ["O", "B-DRUG", "I-DRUG", "B-EFFECT", "I-EFFECT"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

def create_bio_labels(example, tokenizer):
    text = example["text"]
    drug_start = example["indexes"]["drug"]["start_char"][0]
    drug_end   = example["indexes"]["drug"]["end_char"][0]
    effect_start = example["indexes"]["effect"]["start_char"][0]
    effect_end   = example["indexes"]["effect"]["end_char"][0]

    tokens = tokenizer(text, return_offsets_mapping=True, truncation=True, padding="max_length", max_length=128)
    labels = ["O"] * len(tokens["input_ids"])

    for i, (start, end) in enumerate(tokens["offset_mapping"]):
        if start is None or end is None:
            continue
        if drug_start <= start < drug_end:
            labels[i] = "B-DRUG" if start == drug_start else "I-DRUG"
        elif effect_start <= start < effect_end:
            labels[i] = "B-EFFECT" if start == effect_start else "I-EFFECT"

    tokens["labels"] = [label2id.get(l, 0) for l in labels]
    return tokens

# ===============================
# BERT base (Token Classification)
# ===============================
model_ckpt_bert = "bert-base-cased"
tokenizer_bert = AutoTokenizer.from_pretrained(model_ckpt_bert)

tokenized_bert = data.map(lambda x: create_bio_labels(x, tokenizer_bert), batched=False)
tokenized_bert.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model_bert = AutoModelForTokenClassification.from_pretrained(
    model_ckpt_bert,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
).to(device)

data_collator_bert = DataCollatorForTokenClassification(tokenizer_bert)

training_args_bert = TrainingArguments(
    output_dir="./results/ner_bert_base",
    eval_strategy="epoch",   
    logging_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none",
    seed=42
)

trainer_bert = Trainer(
    model=model_bert,
    args=training_args_bert,
    train_dataset=tokenized_bert.select(range(5000)),
    eval_dataset=tokenized_bert.select(range(1000)),
    tokenizer=tokenizer_bert,
    data_collator=data_collator_bert
)

# ---- Stats de tamaño del modelo (params/buffers)
size_stats_bert = model_memory_stats(model_bert)
print(f"[BERT NER] Params totales: {size_stats_bert['total_params']:,} | "
      f"Entrenables: {size_stats_bert['trainable_params']:,} | "
      f"Tamaño modelo: {fmt_bytes(size_stats_bert['footprint_bytes'])}")

# ---- Pico de memoria durante TRAIN
_, train_mem_bert = run_with_mem_tracking(lambda: trainer_bert.train())
print(f"[BERT NER][TRAIN] pico GPU allocated: {fmt_bytes(train_mem_bert['gpu_peak_alloc_bytes'])} | "
      f"reserved: {fmt_bytes(train_mem_bert['gpu_peak_reserved_bytes'])} | "
      f"ΔRAM: {fmt_bytes(train_mem_bert['cpu_rss_delta_bytes'])} | "
      f"tiempo: {train_mem_bert['elapsed_s']:.1f}s")

# ---- Pico de memoria durante EVAL (sobre eval set)
eval_metrics_bert, eval_mem_bert = run_with_mem_tracking(lambda: trainer_bert.evaluate())
print(f"[BERT NER][EVAL ] pico GPU allocated: {fmt_bytes(eval_mem_bert['gpu_peak_alloc_bytes'])} | "
      f"reserved: {fmt_bytes(eval_mem_bert['gpu_peak_reserved_bytes'])} | "
      f"ΔRAM: {fmt_bytes(eval_mem_bert['cpu_rss_delta_bytes'])} | "
      f"tiempo: {eval_mem_bert['elapsed_s']:.1f}s")

mem_stats_bert = {
    "model_name": model_ckpt_bert,
    **size_stats_bert,
    "train_gpu_peak_alloc_bytes": train_mem_bert["gpu_peak_alloc_bytes"],
    "train_gpu_peak_reserved_bytes": train_mem_bert["gpu_peak_reserved_bytes"],
    "train_cpu_rss_delta_bytes": train_mem_bert["cpu_rss_delta_bytes"],
    "train_time_s": train_mem_bert["elapsed_s"],
    "eval_gpu_peak_alloc_bytes": eval_mem_bert["gpu_peak_alloc_bytes"],
    "eval_gpu_peak_reserved_bytes": eval_mem_bert["gpu_peak_reserved_bytes"],
    "eval_cpu_rss_delta_bytes": eval_mem_bert["cpu_rss_delta_bytes"],
    "eval_time_s": eval_mem_bert["elapsed_s"],
}
print("\nmem_stats_bert =", mem_stats_bert)
print("Fila para tabla (BERT):", {
    "modelo": "BERT-base (NER)",
    "Peak VRAM (train, allocated) [GiB]": None if mem_stats_bert["train_gpu_peak_alloc_bytes"] is None else round(mem_stats_bert["train_gpu_peak_alloc_bytes"]/(2**30), 2)
})

# ===============================
# BioBERT (Token Classification)
# ===============================
model_ckpt_biobert = "dmis-lab/biobert-base-cased-v1.1"
tokenizer_biobert = AutoTokenizer.from_pretrained(model_ckpt_biobert)

tokenized_biobert = data.map(lambda x: create_bio_labels(x, tokenizer_biobert), batched=False)
tokenized_biobert.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model_biobert = AutoModelForTokenClassification.from_pretrained(
    model_ckpt_biobert,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
).to(device)

data_collator_biobert = DataCollatorForTokenClassification(tokenizer_biobert)

training_args_biobert = TrainingArguments(
    output_dir="./results/ner_biobert",
    eval_strategy="epoch",   # <-- corregido
    logging_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none",
    seed=42
)

trainer_biobert = Trainer(
    model=model_biobert,
    args=training_args_biobert,
    train_dataset=tokenized_biobert.select(range(5000)),
    eval_dataset=tokenized_biobert.select(range(1000)),
    tokenizer=tokenizer_biobert,
    data_collator=data_collator_biobert
)

# ---- Stats de tamaño del modelo (params/buffers)
size_stats_biobert = model_memory_stats(model_biobert)
print(f"[BioBERT NER] Params totales: {size_stats_biobert['total_params']:,} | "
      f"Entrenables: {size_stats_biobert['trainable_params']:,} | "
      f"Tamaño modelo: {fmt_bytes(size_stats_biobert['footprint_bytes'])}")

# ---- Pico de memoria durante TRAIN
_, train_mem_biobert = run_with_mem_tracking(lambda: trainer_biobert.train())
print(f"[BioBERT NER][TRAIN] pico GPU allocated: {fmt_bytes(train_mem_biobert['gpu_peak_alloc_bytes'])} | "
      f"reserved: {fmt_bytes(train_mem_biobert['gpu_peak_reserved_bytes'])} | "
      f"ΔRAM: {fmt_bytes(train_mem_biobert['cpu_rss_delta_bytes'])} | "
      f"tiempo: {train_mem_biobert['elapsed_s']:.1f}s")

# ---- Pico de memoria durante EVAL (sobre eval set)
eval_metrics_biobert, eval_mem_biobert = run_with_mem_tracking(lambda: trainer_biobert.evaluate())
print(f"[BioBERT NER][EVAL ] pico GPU allocated: {fmt_bytes(eval_mem_biobert['gpu_peak_alloc_bytes'])} | "
      f"reserved: {fmt_bytes(eval_mem_biobert['gpu_peak_reserved_bytes'])} | "
      f"ΔRAM: {fmt_bytes(eval_mem_biobert['cpu_rss_delta_bytes'])} | "
      f"tiempo: {eval_mem_biobert['elapsed_s']:.1f}s")

mem_stats_biobert = {
    "model_name": model_ckpt_biobert,
    **size_stats_biobert,
    "train_gpu_peak_alloc_bytes": train_mem_biobert["gpu_peak_alloc_bytes"],
    "train_gpu_peak_reserved_bytes": train_mem_biobert["gpu_peak_reserved_bytes"],
    "train_cpu_rss_delta_bytes": train_mem_biobert["cpu_rss_delta_bytes"],
    "train_time_s": train_mem_biobert["elapsed_s"],
    "eval_gpu_peak_alloc_bytes": eval_mem_biobert["gpu_peak_alloc_bytes"],
    "eval_gpu_peak_reserved_bytes": eval_mem_biobert["gpu_peak_reserved_bytes"],
    "eval_cpu_rss_delta_bytes": eval_mem_biobert["cpu_rss_delta_bytes"],
    "eval_time_s": eval_mem_biobert["elapsed_s"],
}
print("\nmem_stats_biobert =", mem_stats_biobert)
print("Fila para tabla (BioBERT):", {
    "modelo": "BioBERT (NER)",
    "Peak VRAM (train, allocated) [GiB]": None if mem_stats_biobert["train_gpu_peak_alloc_bytes"] is None else round(mem_stats_biobert["train_gpu_peak_alloc_bytes"]/(2**30), 2)
})


