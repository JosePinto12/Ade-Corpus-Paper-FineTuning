# Librerías
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch

# Verificar dispositivo 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Usando dispositivo:", device)

# 1. Cargar dataset
dataset = load_dataset("SetFit/ade_corpus_v2_classification")

# 2. Tokenización
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_ds = dataset.map(tokenize, batched=True)
tokenized_ds = tokenized_ds.remove_columns(["text"])
tokenized_ds = tokenized_ds.rename_column("label", "labels")
tokenized_ds.set_format("torch")

# 3. Métricas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


model_no_ft = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)
model_no_ft.to(device)  # mover a GPU

args_eval = TrainingArguments(
    output_dir="./eval_no_ft",
    per_device_eval_batch_size=32,
    do_train=False,
    do_eval=True,
    report_to="none",  # desactiva wandb si no lo usamos
)

trainer_no_ft = Trainer(
    model=model_no_ft,
    args=args_eval,
    eval_dataset=tokenized_ds["test"],
    compute_metrics=compute_metrics,
)

print("Evaluación de BERT sin Fine-Tuning:")
trainer_no_ft.evaluate()

model_ft = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)
model_ft.to(device)  # mover a GPU

args_ft = TrainingArguments(
    output_dir="./bert_finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
)


trainer_ft = Trainer(
    model=model_ft,
    args=args_ft,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    compute_metrics=compute_metrics,
)

trainer_ft.train()


print("Evaluación de BERT con Fine-Tuning:")
trainer_ft.evaluate()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Obtener predicciones en el set de test
preds_output = trainer_ft.predict(tokenized_ds["test"])
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No ADE", "ADE"])

# Visualizar
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", values_format='d')
plt.title("Matriz de Confusión - Modelo Fine-Tuned")
plt.show()


import pandas as pd

# Convertir logs a DataFrame
logs = pd.DataFrame(trainer_ft.state.log_history)

# Filtrar valores con 'eval_loss' y 'eval_f1'
eval_logs = logs[logs["step"].notnull()][["step", "eval_loss", "eval_f1"]].dropna()

# Graficar
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(eval_logs["step"], eval_logs["eval_loss"], marker='o')
plt.title("Evaluación: Pérdida por época")
plt.xlabel("Step")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(eval_logs["step"], eval_logs["eval_f1"], marker='o')
plt.title("Evaluación: F1 por época")
plt.xlabel("Step")
plt.ylabel("F1 Score")

plt.tight_layout()
plt.show()


# Comparar métricas (usando claves correctas)
metrics = ["eval_accuracy", "eval_f1"]
values = [
    [eval_no_ft[m] for m in metrics],
    [eval_ft[m] for m in metrics],
]

# Mostrar con etiquetas limpias
df_comp = pd.DataFrame(values, columns=["Accuracy", "F1"], index=["Sin Fine-Tuning", "Con Fine-Tuning"])

# Graficar
df_comp.plot(kind='bar', figsize=(8, 4), rot=0)
plt.title("Comparación de rendimiento: Base vs Fine-Tuned")
plt.ylabel("Puntaje")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()


# Obtener predicciones nuevamente
preds_output = trainer_ft.predict(tokenized_ds["test"])
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)

# Cargar textos originales
test_texts = dataset["test"]["text"]

# Mostrar ejemplos mal clasificados
wrong_idxs = np.where(y_true != y_pred)[0]

print(f"Total de errores: {len(wrong_idxs)}")

# Mostrar 5 errores aleatorios
import random
random.seed(42)
sample_wrong = random.sample(list(wrong_idxs), 5)

for idx in sample_wrong:
    print(f"\n Índice: {idx}")
    print("Texto:", test_texts[idx])
    print(f" Real: {y_true[idx]} | 🟡 Predicho: {y_pred[idx]}")
