# Librerías
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# Cargar el dataset desde Hugging Face
dataset = load_dataset("SetFit/ade_corpus_v2_classification")

# Convertir a pandas DataFrame
df_train = dataset['train'].to_pandas()
df_test = dataset['test'].to_pandas()

# Mostrar las primeras filas
print("Primeras filas del set de entrenamiento:")
display(df_train.head())

print("Primeras filas del set de prueba:")
display(df_test.head())

# Revisión de clases
print("\nDistribución de clases (0 = no ADE, 1 = ADE):")
print(df_train['label'].value_counts())

# Visualización
sns.countplot(data=df_train, x='label')
plt.title("Distribución de clases en el set de entrenamiento")
plt.xlabel("Label (0 = No ADE, 1 = ADE)")
plt.ylabel("Cantidad")
plt.show()

# Longitud promedio de los textos
df_train['length'] = df_train['text'].apply(len)
print("\nEstadísticas de longitud de los textos:")
print(df_train['length'].describe())

sns.histplot(df_train['length'], bins=50, kde=True)
plt.title("Distribución de longitud de los textos")
plt.xlabel("Cantidad de caracteres")
plt.ylabel("Frecuencia")
plt.show()

# Descargar stopwords en español e inglés (aunque los textos están en inglés)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Función para generar texto unido de todos los registros de una clase
def get_class_text(df, label):
    return " ".join(df[df['label'] == label]['text'].values)

# Obtener textos por clase
text_no_ade = get_class_text(df_train, 0)
text_ade = get_class_text(df_train, 1)

# Generar WordClouds
wordcloud_no_ade = WordCloud(width=800, height=400, background_color='white',
                             stopwords=stop_words, max_words=200).generate(text_no_ade)

wordcloud_ade = WordCloud(width=800, height=400, background_color='white',
                          stopwords=stop_words, max_words=200).generate(text_ade)

# Visualizar WordClouds
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(wordcloud_no_ade, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud - Clase 0 (No ADE)")

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_ade, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud - Clase 1 (ADE)")

plt.tight_layout()
plt.show()