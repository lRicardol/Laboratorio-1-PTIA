# Laboratorio-1-PTIA

# ESCUELA COLOMBIANA DE INGENIERÍA
### PRINCIPIOS Y TECNOLOGÍAS IA 2025-2
### REDES NEURONALES
### LABORATORIO 1/4

#### Ricardo Ayala
#### Allan Contreras

OBJETIVOS

Desarrollar competencias básicas para:

Modelar y resolver problemas usando redes neuronales
Implementar los algoritmos hacia adelante (FEED-FORWARD) y hacia atrás con aprendizaje (BACKPROPAGATION)
Apropiar un framework para redes neuronales (keras)
ENTREGABLE

Reglas para el envío de los entregables:

Forma de envío: Esta tarea se debe enviar únicamente a través de la plataforma Moodle en la actividad definida. Se tendrán dos entregas: inicial y final.

Formato de los archivos: Incluyan en un archivo .zip los archivos correspondientes al laboratorio.

Nomenclatura para nombrar los archivos: El archivo deberá ser renombrado, “RN-lab-” seguido por los usuarios institucionales de los autores ordenados alfabéticamente (por ejemplo, se debe adicionar pedroperez al nombre del archivo, si el correo electrónico de Pedro Pérez es pedro.perez@mail.escuelaing.edu.co)

PARTE I. IMPLEMENTACIÓN DE RED NEURONAL
Para este apartado se va a implementar una red neuronal con algoritmo de aprendizaje, en este caso propagación hacia atras del error.

Introducido en la década de 1960 y popularizado casi 30 años después (1989) por Rumelhart, Hinton y Williams en el artículo titulado «Learning representations by back-propagating errors».

IMPLEMENTACIÓN DE RED NEURONAL CON PROPAGACIÓN HACIA ATRÁS
Implementar una red neuronal totalmente conectada desde su definición simple; calculando una salida Yˇ(Yp) para unas entradas X.

Propiedades y parámetros:

Tarea: Clasificación multiple

Tipo de capas: Densas

Métrica para evaluación : ACCURACY

![Screenshot 2025-09-04 122516.png](Imag%2FScreenshot%202025-09-04%20122516.png)

Funciones de activación

Función de activación en Capas ocultas : ReLU

![Screenshot 2025-09-04 122523.png](Imag%2FScreenshot%202025-09-04%20122523.png)

Función de activación en Capa de salida : Sigmoide

![Screenshot 2025-09-04 122530.png](Imag%2FScreenshot%202025-09-04%20122530.png)

Funcion de costo

Función de costo/perdida «error»: Entropia Cruzada «Cross-Entropy»

![Screenshot 2025-09-04 122538.png](Imag%2FScreenshot%202025-09-04%20122538.png)

Paso 1. Derivadas
Incluya en este apartado el proceso de la derivación de las funciones

Derivada función Sigmoide:

![Screenshot 2025-09-04 123557.png](Imag%2FScreenshot%202025-09-04%20123557.png)

---

Derivada función ReLU

![Screenshot 2025-09-04 123923.png](Imag%2FScreenshot%202025-09-04%20123923.png)

---

Derivada función de costo: Entropia Cruzada

![Screenshot 2025-09-04 131931.png](Imag%2FScreenshot%202025-09-04%20131931.png)

![Screenshot 2025-09-04 131940.png](Imag%2FScreenshot%202025-09-04%20131940.png)

![Screenshot 2025-09-04 131948.png](Imag%2FScreenshot%202025-09-04%20131948.png)

---

## Paso 2. Implementación del código para ANN (Dense)

### LIBRERÍA NECESARIA

    import numpy as np
    from abc import ABC, abstractmethod

### FUNCIONES DE BASE: MÉTRICA, COSTO Y ACTIVACIÓN

![Screenshot 2025-09-04 132320.png](Imag%2FScreenshot%202025-09-04%20132320.png)

### MÉTRICA

```python
class Metric(ABC):
    """Clase abstracta para definir métricas de desempeño en redes neuronales."""

    def use(self, name: str) -> "Metric":
        """Obtiene la métrica (objeto) a partir del nombre"""
        if name.lower() == "accuracy":
            return Accuracy()
        else:
            raise ValueError(f"Métrica {name} no implementada")

    @abstractmethod
    def value(self, Y: np.ndarray, Yp: np.ndarray) -> float:
        """Computa el desempeño de la red neuronal"""
        pass
```
```python
class Accuracy(Metric):
    """Métrica de exactitud (aciertos / total)."""

    def value(self, Y: np.ndarray, Yp: np.ndarray) -> float:
        """
        Calcula la exactitud comparando clases verdaderas vs predichas.
        Args:
            Y (ndarray): etiquetas verdaderas en one-hot encoding
            Yp (ndarray): predicciones en forma de probabilidades
        Returns:
            float: exactitud (0-1)
        """
        preds = np.argmax(Yp, axis=1)
        labels = np.argmax(Y, axis=1)
        return np.mean(preds == labels)

```
```python
if __name__ == "__main__":
    # Datos ficticios
    Y_true = np.array([[1,0,0],[0,1,0],[0,0,1]])
    Y_pred = np.array([[0.9,0.05,0.05],[0.1,0.8,0.1],[0.2,0.2,0.6]])
    
    acc = Accuracy()
    print("Accuracy:", acc.value(Y_true, Y_pred))
```

---

### COSTO

```python
class Cost(ABC):
    """Clase abstracta para definir funciones de costo en redes neuronales."""

    def use(self, name: str) -> "Cost":
        """Obtiene la función de costo a partir del nombre"""
        if name.lower() == "crossentropy":
            return CrossEntropy()
        else:
            raise ValueError(f"Función de costo {name} no implementada")

    @abstractmethod
    def value(self, Y: np.ndarray, Yp: np.ndarray) -> float:
        """Computa la función de costo"""
        pass

    @abstractmethod
    def derivative(self, Y: np.ndarray, Yp: np.ndarray) -> np.ndarray:
        """Computa la derivada de la función de costo (gradiente)"""
        pass

```
```python
class CrossEntropy(Cost):
    """Función de costo: Entropía Cruzada."""

    def value(self, Y: np.ndarray, Yp: np.ndarray) -> float:
        """
        Calcula la entropía cruzada.
        Args:
            Y (ndarray): etiquetas verdaderas one-hot
            Yp (ndarray): predicciones del modelo
        Returns:
            float: valor de la pérdida
        """
        eps = 1e-12  # para evitar log(0)
        Yp = np.clip(Yp, eps, 1 - eps)
        return -np.mean(np.sum(Y * np.log(Yp), axis=1))

    def derivative(self, Y: np.ndarray, Yp: np.ndarray) -> np.ndarray:
        """
        Derivada de la entropía cruzada respecto a las predicciones.
        Args:
            Y (ndarray): etiquetas verdaderas
            Yp (ndarray): predicciones
        Returns:
            ndarray: gradiente del costo
        """
        eps = 1e-12
        Yp = np.clip(Yp, eps, 1 - eps)
        return - (Y / Yp)

```
```python
if __name__ == "__main__":
    Y_true = np.array([[1,0,0],[0,1,0],[0,0,1]])
    Y_pred = np.array([[0.9,0.05,0.05],[0.1,0.8,0.1],[0.2,0.2,0.6]])

    ce = CrossEntropy()
    print("CrossEntropy:", ce.value(Y_true, Y_pred))
    print("CrossEntropy derivada:", ce.derivative(Y_true, Y_pred))
```

---

### ACTIVACION

```python
class Activation(ABC):
    """Clase abstracta para definir funciones de activación."""

    def use(self, name: str) -> "Activation":
        """Obtiene la función de activación a partir del nombre"""
        if name.lower() == "sigmoid":
            return Sigmoid()
        elif name.lower() == "relu":
            return Relu()
        else:
            raise ValueError(f"Función de activación {name} no implementada")

    @abstractmethod
    def value(self, X: np.ndarray) -> np.ndarray:
        """Computa la activación"""
        pass

    @abstractmethod
    def derivative(self, X: np.ndarray) -> np.ndarray:
        """Computa la derivada de la activación"""
        pass
```
```python
class Sigmoid(Activation):
    """Función de activación sigmoide."""

    def value(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def derivative(self, X: np.ndarray) -> np.ndarray:
        s = self.value(X)
        return s * (1 - s)
```
```python
class Relu(Activation):
    """Función de activación ReLU."""

    def value(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    def derivative(self, X: np.ndarray) -> np.ndarray:
        return (X > 0).astype(float)

```
```python
if __name__ == "__main__":
    # Datos ficticios
    Y_true = np.array([[1,0,0],[0,1,0],[0,0,1]])
    Y_pred = np.array([[0.9,0.05,0.05],[0.1,0.8,0.1],[0.2,0.2,0.6]])

    X = np.array([[-1, 0, 1]])
    sigmoid = Sigmoid()
    relu = Relu()
```

---

## RED NEURONAL TOTALMENTE CONECTADA «DENSE»

#### Nomenclatura
* **Datos**
    - *c*: número de características
    - *m*: número de ejemplares
    - **x**, **X** : entradas. Un ejemplo (c) o todos los ejemplos (cxm)
    - **y**, **Y** : salidas reales. Un ejemplo (cx1) o todos los ejemplos(cxm)
    - **yp**, **Yp** : salidas estimadas. Un ejemplo (cx1) o todos los ejemplos(cxm)
* **Arquitectura**
    - *L*: número de capas
    - **layers**: **n**[*0*] = c, **layers**[*i*] número de neuronas de la capa *i*
* **Parámetros**
    - **W**: pesos de una capa (**layers**[*l+1*]x**layers**[*l*])
    - **b**: sesgos de una capa (**n**[*l* ]x1)

* **Gradientes**
    - **dW**: gradiente de **W**
    - **db**: gradiente de **b**

*Incluya en este apartado el proceso de la derivación de los gradientes*

![Screenshot 2025-09-04 141426.png](Imag%2FScreenshot%202025-09-04%20141426.png)

---
**Gradiente dW**

![Screenshot 2025-09-04 141716.png](Imag%2FScreenshot%202025-09-04%20141716.png)

---
**Graciente db**

![Screenshot 2025-09-04 142038.png](Imag%2FScreenshot%202025-09-04%20142038.png)

---
**Resulatdo Final**

![Screenshot 2025-09-04 142214.png](Imag%2FScreenshot%202025-09-04%20142214.png)

---

![Screenshot 2025-09-04 142301.png](Imag%2FScreenshot%202025-09-04%20142301.png)


## PARTE 2. USO DE FRAMEWORK PARA REDES NEURONALES
- Paso 1 – Definir problema
  
``` txt
## Descripción:
Queremos construir un clasificador automático que, a partir de medidas de longitud y ancho de sépalos y pétalos,
prediga la especie de flor iris.

Variable objetivo (target):
species

## Clases:

setosa

versicolor

virginica

Métrica:
Accuracy (precisión)

Umbral mínimo aceptado:
85% 
 ```
- codigo
```python
# -*- coding: utf-8 -*-
"""
Plantilla completa: Clasificación con Keras (API Secuencial)
Autor: (tu nombre)
Uso:
  python keras_clasificacion_template.py --data data/dataset.csv --target nombre_columna_objetivo --sep ,

Qué hace:
- Paso 1: Define el problema (se deja sección para documentarlo)
- Paso 2: Explora y prepara datos (EDA básico + preprocesamiento con ColumnTransformer)
- Paso 3: Construye, compila, entrena y evalúa una red neuronal secuencial en Keras
- Divide 70%/10%/20% (train/val/test) con estratificación por clase
- Guarda artefactos: preprocesador, mejor modelo, y gráficos (si hay entorno gráfico)

Requisitos:
pip install tensorflow scikit-learn pandas matplotlib joblib
"""

import argparse
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------
# Utilidades
# -------------------------
def set_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_fig(path: str):
    try:
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        print(f"[INFO] Gráfico guardado en: {path}")
    except Exception as e:
        print(f"[WARN] No se pudo guardar el gráfico en {path}: {e}")
    finally:
        plt.close()

# -------------------------
# Paso 1: Definir el problema (documental)
# -------------------------
PROBLEM_DEFINITION = {
    "descripcion": "Clasificar instancias en N clases usando el dataset proporcionado por el profesor.",
    "metrica": "accuracy",
    "umbral_exito": 0.85,
    "comentarios": "Ajusta esta sección según tu problema real (clases, contexto, balance, etc.)."
}

# -------------------------
# Paso 2: Explorar y preparar los datos
# -------------------------
def load_dataset(csv_path: str, sep: str = ",") -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        print(f"[ERROR] No existe el archivo: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path, sep=sep)
    return df

def eda_basico(df: pd.DataFrame, target_col: str, outdir: str):
    print("\n===== EDA: Info general =====")
    print(df.info())
    print("\n===== EDA: Descripción numérica =====")
    print(df.describe(include='all'))

    # Valores nulos por columna
    print("\n===== EDA: Nulos por columna =====")
    print(df.isnull().sum())

    # Distribución de clases (si target es categórico/entero)
    if target_col in df.columns:
        plt.figure()
        df[target_col].value_counts().sort_index().plot(kind="bar")
        plt.title("Distribución de la variable objetivo")
        plt.xlabel("Clase")
        plt.ylabel("Frecuencia")
        save_fig(os.path.join(outdir, "01_distribucion_clases.png"))

    # Matriz de correlación (solo numéricas)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True)
        plt.figure()
        plt.imshow(corr, interpolation="nearest")
        plt.title("Matriz de correlación (numéricas)")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
        save_fig(os.path.join(outdir, "02_correlacion.png"))

def split_sets(X, y, seed: int = 42):
    # 70% train, 10% val, 20% test con estratificación
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    # de ese 30% para temp, 2/3 será test (=> 20% total) y 1/3 valid (=> 10% total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(2/3), random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Manejar versiones de sklearn (sparse vs dense)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # compatibilidad con versiones anteriores
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )
    return preprocessor

def preprocess_fit_transform(preprocessor, X_train, X_val, X_test, outdir: str):
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    # Asegurar ndarray denso (por si alguna etapa produce sparse)
    if not isinstance(X_train_p, np.ndarray):
        X_train_p = X_train_p.toarray()
        X_val_p = X_val_p.toarray()
        X_test_p = X_test_p.toarray()

    # Guardar preprocesador
    joblib.dump(preprocessor, os.path.join(outdir, "preprocesador.joblib"))
    print(f"[INFO] Preprocesador guardado en {os.path.join(outdir, 'preprocesador.joblib')}")

    return X_train_p, X_val_p, X_test_p

# -------------------------
# Paso 3: Desarrollar la red
# -------------------------
def build_model(input_dim: int, num_classes: int, hidden_units=(128, 64), dropout=0.2):
    model = Sequential()
    # Capa 1
    model.add(Dense(hidden_units[0], activation="relu", input_dim=input_dim))
    model.add(Dropout(dropout))
    # Capa 2
    model.add(Dense(hidden_units[1], activation="relu"))
    model.add(Dropout(dropout))

    if num_classes == 2:
        # Binaria
        model.add(Dense(1, activation="sigmoid"))
    else:
        # Multiclase
        model.add(Dense(num_classes, activation="softmax"))
    return model

def compile_model(model, num_classes: int):
    if num_classes == 2:
        loss = "binary_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    else:
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    return model

def plot_learning_curves(history, outdir: str):
    hist = history.history
    # Accuracy
    if "accuracy" in hist:
        plt.figure()
        plt.plot(hist["accuracy"], label="train_acc")
        if "val_accuracy" in hist:
            plt.plot(hist["val_accuracy"], label="val_acc")
        plt.xlabel("Épocas")
        plt.ylabel("Accuracy")
        plt.legend()
        save_fig(os.path.join(outdir, "03_learning_curves_accuracy.png"))
    # Loss
    if "loss" in hist:
        plt.figure()
        plt.plot(hist["loss"], label="train_loss")
        if "val_loss" in hist:
            plt.plot(hist["val_loss"], label="val_loss")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.legend()
        save_fig(os.path.join(outdir, "04_learning_curves_loss.png"))

def plot_confusion_matrix(cm, class_names, outpath: str):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    save_fig(outpath)

def train_and_evaluate(df: pd.DataFrame, target_col: str, outdir: str, seed: int = 42,
                       epochs: int = 50, batch_size: int = 32):
    set_seeds(seed)
    ensure_dir(outdir)

    # EDA
    eda_basico(df, target_col, outdir)

    # Separación X/y
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el dataset.")
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    # Si y es categórica/string, codificar a enteros
    label_encoder = None
    if y_raw.dtype == "O" or str(y_raw.dtype).startswith("category"):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.values)
        joblib.dump(label_encoder, os.path.join(outdir, "label_encoder.joblib"))
        print(f"[INFO] LabelEncoder guardado en {os.path.join(outdir, 'label_encoder.joblib')}")
        class_names = list(label_encoder.classes_)
    else:
        y = y_raw.values
        class_names = [str(c) for c in np.unique(y)]

    # Splits
    X_train, X_val, X_test, y_train, y_val, y_test = split_sets(X, y, seed=seed)

    # Preprocesamiento
    preprocessor = build_preprocessor(X_train)
    X_train_p, X_val_p, X_test_p = preprocess_fit_transform(
        preprocessor, X_train, X_val, X_test, outdir
    )

    # Modelo
    num_classes = len(np.unique(y))
    model = build_model(input_dim=X_train_p.shape[1], num_classes=num_classes)
    model = compile_model(model, num_classes=num_classes)

    # Callbacks
    ckpt_path = os.path.join(outdir, "best_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)
    ]

    # Entrenamiento
    history = model.fit(
        X_train_p, y_train,
        validation_data=(X_val_p, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks
    )

    # Curvas de aprendizaje
    plot_learning_curves(history, outdir)

    # Evaluación
    print("\n===== Evaluación en Test =====")
    results = model.evaluate(X_test_p, y_test, verbose=0)
    print(dict(zip(model.metrics_names, results)))

    # Predicciones para métricas detalladas
    if num_classes == 2:
        y_prob = model.predict(X_test_p, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_prob = model.predict(X_test_p, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy (test): {acc:.4f}")

    print("\n===== Classification Report =====")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    plot_confusion_matrix(cm, class_names, os.path.join(outdir, "05_confusion_matrix.png"))

    # Guardar un resumen JSON de resultados
    summary = {
        "problem_definition": {
            "descripcion": "Clasificar instancias en N clases usando el dataset proporcionado por el profesor.",
            "metrica": "accuracy",
            "umbral_exito": 0.85
        },
        "input_features": list(X.columns),
        "num_classes": num_classes,
        "class_names": class_names,
        "metrics_names": model.metrics_names,
        "metrics_test": {name: float(val) for name, val in zip(model.metrics_names, results)},
        "accuracy_test": float(acc),
        "artifacts": {
            "best_model": "best_model.keras",
            "preprocessor": "preprocesador.joblib",
            "label_encoder": "label_encoder.joblib" if os.path.exists(os.path.join(outdir, 'label_encoder.joblib')) else None,
            "plots": [
                "01_distribucion_clases.png",
                "02_correlacion.png",
                "03_learning_curves_accuracy.png",
                "04_learning_curves_loss.png",
                "05_confusion_matrix.png",
            ]
        }
    }
    with open(os.path.join(outdir, "resumen_resultados.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Resumen de resultados guardado en {os.path.join(outdir, 'resumen_resultados.json')}")

def parse_args():
    parser = argparse.ArgumentParser(description="Clasificación con Keras (API Secuencial)")
    parser.add_argument("--data", type=str, required=True, help="Ruta al CSV del dataset")
    parser.add_argument("--target", type=str, required=True, help="Nombre de la columna objetivo")
    parser.add_argument("--sep", type=str, default=",", help="Separador del CSV (por defecto ',')")
    parser.add_argument("--outdir", type=str, default="resultados", help="Carpeta de salida de artefactos")
    parser.add_argument("--epochs", type=int, default=50, help="Épocas de entrenamiento (default 50)")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch (default 32)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    return parser.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.outdir)
    print("=== Parámetros ===")
    print(vars(args))

    df = load_dataset(args.data, sep=args.sep)
    train_and_evaluate(
        df=df,
        target_col=args.target,
        outdir=args.outdir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()


```
- CSV
```python
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
4.6,3.1,1.5,0.2,setosa
5,3.6,1.4,0.2,setosa
5.4,3.9,1.7,0.4,setosa
4.6,3.4,1.4,0.3,setosa
5,3.4,1.5,0.2,setosa
4.4,2.9,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
5.4,3.7,1.5,0.2,setosa
4.8,3.4,1.6,0.2,setosa
4.8,3,1.4,0.1,setosa
4.3,3,1.1,0.1,setosa
5.8,4,1.2,0.2,setosa
5.7,4.4,1.5,0.4,setosa
5.4,3.9,1.3,0.4,setosa
5.1,3.5,1.4,0.3,setosa
5.7,3.8,1.7,0.3,setosa
5.1,3.8,1.5,0.3,setosa
5.4,3.4,1.7,0.2,setosa
5.1,3.7,1.5,0.4,setosa
4.6,3.6,1,0.2,setosa
5.1,3.3,1.7,0.5,setosa
4.8,3.4,1.9,0.2,setosa
5,3,1.6,0.2,setosa
5,3.4,1.6,0.4,setosa
5.2,3.5,1.5,0.2,setosa
5.2,3.4,1.4,0.2,setosa
4.7,3.2,1.6,0.2,setosa
4.8,3.1,1.6,0.2,setosa
5.4,3.4,1.5,0.4,setosa
5.2,4.1,1.5,0.1,setosa
5.5,4.2,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
5,3.2,1.2,0.2,setosa
5.5,3.5,1.3,0.2,setosa
4.9,3.1,1.5,0.1,setosa
4.4,3,1.3,0.2,setosa
5.1,3.4,1.5,0.2,setosa
5,3.5,1.3,0.3,setosa
4.5,2.3,1.3,0.3,setosa
4.4,3.2,1.3,0.2,setosa
5,3.5,1.6,0.6,setosa
5.1,3.8,1.9,0.4,setosa
4.8,3,1.4,0.3,setosa
5.1,3.8,1.6,0.2,setosa
4.6,3.2,1.4,0.2,setosa
5.3,3.7,1.5,0.2,setosa
5,3.3,1.4,0.2,setosa
7,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
5.5,2.3,4,1.3,versicolor
6.5,2.8,4.6,1.5,versicolor
5.7,2.8,4.5,1.3,versicolor
6.3,3.3,4.7,1.6,versicolor
4.9,2.4,3.3,1,versicolor
6.6,2.9,4.6,1.3,versicolor
5.2,2.7,3.9,1.4,versicolor
5,2,3.5,1,versicolor
5.9,3,4.2,1.5,versicolor
6,2.2,4,1,versicolor
6.1,2.9,4.7,1.4,versicolor
5.6,2.9,3.6,1.3,versicolor
6.7,3.1,4.4,1.4,versicolor
5.6,3,4.5,1.5,versicolor
5.8,2.7,4.1,1,versicolor
6.2,2.2,4.5,1.5,versicolor
5.6,2.5,3.9,1.1,versicolor
5.9,3.2,4.8,1.8,versicolor
6.1,2.8,4,1.3,versicolor
6.3,2.5,4.9,1.5,versicolor
6.1,2.8,4.7,1.2,versicolor
6.4,2.9,4.3,1.3,versicolor
6.6,3,4.4,1.4,versicolor
6.8,2.8,4.8,1.4,versicolor
6.7,3,5,1.7,versicolor
6,2.9,4.5,1.5,versicolor
5.7,2.6,3.5,1,versicolor
5.5,2.4,3.8,1.1,versicolor
5.5,2.4,3.7,1,versicolor
5.8,2.7,3.9,1.2,versicolor
6,2.7,5.1,1.6,versicolor
5.4,3,4.5,1.5,versicolor
6,3.4,4.5,1.6,versicolor
6.7,3.1,4.7,1.5,versicolor
6.3,2.3,4.4,1.3,versicolor
5.6,3,4.1,1.3,versicolor
5.5,2.5,4,1.3,versicolor
5.5,2.6,4.4,1.2,versicolor
6.1,3,4.6,1.4,versicolor
5.8,2.6,4,1.2,versicolor
5,2.3,3.3,1,versicolor
5.6,2.7,4.2,1.3,versicolor
5.7,3,4.2,1.2,versicolor
5.7,2.9,4.2,1.3,versicolor
6.2,2.9,4.3,1.3,versicolor
5.1,2.5,3,1.1,versicolor
5.7,2.8,4.1,1.3,versicolor
6.3,3.3,6,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3,5.9,2.1,virginica
6.3,2.9,5.6,1.8,virginica
6.5,3,5.8,2.2,virginica
7.6,3,6.6,2.1,virginica
4.9,2.5,4.5,1.7,virginica
7.3,2.9,6.3,1.8,virginica
6.7,2.5,5.8,1.8,virginica
7.2,3.6,6.1,2.5,virginica
6.5,3.2,5.1,2,virginica
6.4,2.7,5.3,1.9,virginica
6.8,3,5.5,2.1,virginica
5.7,2.5,5,2,virginica
5.8,2.8,5.1,2.4,virginica
6.4,3.2,5.3,2.3,virginica
6.5,3,5.5,1.8,virginica
7.7,3.8,6.7,2.2,virginica
7.7,2.6,6.9,2.3,virginica
6,2.2,5,1.5,virginica
6.9,3.2,5.7,2.3,virginica
5.6,2.8,4.9,2,virginica
7.7,2.8,6.7,2,virginica
6.3,2.7,4.9,1.8,virginica
6.7,3.3,5.7,2.1,virginica
7.2,3.2,6,1.8,virginica
6.2,2.8,4.8,1.8,virginica
6.1,3,4.9,1.8,virginica
6.4,2.8,5.6,2.1,virginica
7.2,3,5.8,1.6,virginica
7.4,2.8,6.1,1.9,virginica
7.9,3.8,6.4,2,virginica
6.4,2.8,5.6,2.2,virginica
6.3,2.8,5.1,1.5,virginica
6.1,2.6,5.6,1.4,virginica
7.7,3,6.1,2.3,virginica
6.3,3.4,5.6,2.4,virginica
6.4,3.1,5.5,1.8,virginica
6,3,4.8,1.8,virginica
6.9,3.1,5.4,2.1,virginica
6.7,3.1,5.6,2.4,virginica
6.9,3.1,5.1,2.3,virginica
5.8,2.7,5.1,1.9,virginica
6.8,3.2,5.9,2.3,virginica
6.7,3.3,5.7,2.5,virginica
6.7,3,5.2,2.3,virginica
6.3,2.5,5,1.9,virginica
6.5,3,5.2,2,virginica
6.2,3.4,5.4,2.3,virginica
5.9,3,5.1,1.8,virginica

```

- Paso 2 – Explorar y preparar
``` txt
EDA realizado:

info(): 150 filas × 5 columnas, sin valores nulos.

describe(): estadísticas numéricas de sépalos y pétalos.

Distribución de clases: balanceada (50 ejemplos por especie).

Correlación: alta correlación entre largo/ancho de pétalos y la especie.

## División de datos:

Train: 70% (105 muestras)

Validación: 10% (15 muestras)

Test: 20% (30 muestras)
con estratificación para mantener balance entre clases.

Preprocesamiento (ColumnTransformer):

Numéricas (sepal_length, sepal_width, petal_length, petal_width):

Imputación (mediana)

Escalado estándar (StandardScaler)

Categóricas (ninguna en este caso, salvo target que va codificada con LabelEncoder):

Imputación (moda)

OneHotEncoder

✅ Se guarda el preprocesador.joblib y el label_encoder.joblib para reproducir predicciones.
```
- Paso 3 – Desarrollar la red
``` txt
  Arquitectura (modelo secuencial):

Capa densa (ReLU)

Dropout

Capa densa (ReLU)

Dropout

Capa de salida con softmax (3 clases, multiclase)

Pérdida: sparse_categorical_crossentropy

## Entrenamiento:

Callbacks: EarlyStopping y ModelCheckpoint

Se guarda best_model.keras

## Métricas y resultados:

Curvas de aprendizaje (03_learning_curves_accuracy.png, 04_learning_curves_loss.png)

Accuracy de validación: > 90%

Evaluación en test:

classification_report con precisión, recall y f1-score

accuracy final > 90%

Matriz de confusión (05_confusion_matrix.png)

Artefactos en resultados/:

best_model.keras

preprocesador.joblib

label_encoder.joblib

01_distribucion_clases.png

02_correlacion.png

03_learning_curves_accuracy.png

04_learning_curves_loss.png

05_confusion_matrix.png

resumen_resultados.json
```
- <img width="1495" height="837" alt="image" src="https://github.com/user-attachments/assets/b2303e77-b737-4134-bad1-cb2601213f82" />
- <img width="1852" height="758" alt="image" src="https://github.com/user-attachments/assets/ca66bbf7-2b52-418b-9cc8-1773858bd00d" />
- <img width="1887" height="887" alt="image" src="https://github.com/user-attachments/assets/e4508deb-ddd4-4b3f-8934-f71d16d9f01e" />
- <img width="1919" height="872" alt="image" src="https://github.com/user-attachments/assets/1804af95-b373-48f7-8a19-00c4608c12cd" />
- <img width="1731" height="857" alt="image" src="https://github.com/user-attachments/assets/0da68b06-bbad-4ca1-95e1-3f870b455012" />
- <img width="1569" height="809" alt="image" src="https://github.com/user-attachments/assets/b8b8a3e2-61e5-4cf6-aa95-8eec62f9aa7f" />
- <img width="1454" height="945" alt="image" src="https://github.com/user-attachments/assets/df6a45b5-ebde-4988-bc30-8671a88925a6" />
- <img width="1500" height="991" alt="image" src="https://github.com/user-attachments/assets/3aaeb618-d6db-4894-ad6f-7a962f449f81" />

- <img width="1522" height="872" alt="image" src="https://github.com/user-attachments/assets/e4ff3f19-87a5-4773-bdb8-7a389c92a93a" />
- <img width="1500" height="987" alt="image" src="https://github.com/user-attachments/assets/e6013b7c-e36a-46de-800d-8bdc71b29cf9" />
- <img width="1337" height="984" alt="image" src="https://github.com/user-attachments/assets/03ff6247-63bf-46b3-af9b-986087e2d44d" />

- Paso 4 – Conclusiones
``` txt
  El modelo logró superar el umbral de desempeño planteado (85% de accuracy),
alcanzando una precisión superior al 90% en el conjunto de prueba. Esto valida que la red neuronal
 propuesta es adecuada para la tarea de clasificación de flores Iris.

La exploración de datos mostró un dataset balanceado, con 50 ejemplos por clase (setosa, versicolor, virginica),
 lo cual facilitó el entrenamiento y evitó problemas de desbalance de clases.

Las variables más discriminativas fueron las medidas de los pétalos, en particular largo y ancho,
que presentaron correlaciones altas con la especie. Esto coincide con la literatura clásica del dataset Iris.

El preprocesamiento fue clave para estandarizar las variables numéricas y garantizar que la red neuronal
trabajara con valores comparables.
 La separación estratificada en train/valid/test permitió evaluar el modelo de manera justa.

El uso de callbacks (EarlyStopping y ModelCheckpoint) permitió evitar sobreajuste, guardando automáticamente el mejor modelo entrenado.
 Esto asegura reproducibilidad y un desempeño óptimo en producción.

El flujo completo generó artefactos reutilizables (best_model.keras, preprocesador.joblib, label_encoder.joblib,
 gráficas y resumen en JSON),
 lo cual facilita tanto la trazabilidad como el despliegue futuro del modelo.

##En conclusión
 el pipeline de clasificación desarrollado es robusto, replicable y extensible a
otros problemas de clasificación multiclase,
 siempre que se disponga de un dataset estructurado con suficientes ejemplos por clase.
```
