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
Descripción:
Queremos construir un clasificador automático que, a partir de medidas de longitud y ancho de sépalos y pétalos, prediga la especie de flor iris.

Variable objetivo (target):
species

Clases:

setosa

versicolor

virginica

Métrica:
Accuracy (precisión)

Umbral mínimo aceptado:
85% 
 ```
- Paso 2 – Explorar y preparar
``` txt
EDA realizado:

info(): 150 filas × 5 columnas, sin valores nulos.

describe(): estadísticas numéricas de sépalos y pétalos.

Distribución de clases: balanceada (50 ejemplos por especie).

Correlación: alta correlación entre largo/ancho de pétalos y la especie.

División de datos:

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

Entrenamiento:

Callbacks: EarlyStopping y ModelCheckpoint

Se guarda best_model.keras

Métricas y resultados:

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
- Paso 4 – Conclusiones
``` txt
  El modelo logró superar el umbral de desempeño planteado (85% de accuracy),
alcanzando una precisión superior al 90% en el conjunto de prueba. Esto valida que la red neuronal propuesta es adecuada para la tarea de clasificación de flores Iris.

La exploración de datos mostró un dataset balanceado, con 50 ejemplos por clase (setosa, versicolor, virginica),
 lo cual facilitó el entrenamiento y evitó problemas de desbalance de clases.

Las variables más discriminativas fueron las medidas de los pétalos, en particular largo y ancho,
que presentaron correlaciones altas con la especie. Esto coincide con la literatura clásica del dataset Iris.

El preprocesamiento fue clave para estandarizar las variables numéricas y garantizar que la red neuronal trabajara con valores comparables.
 La separación estratificada en train/valid/test permitió evaluar el modelo de manera justa.

El uso de callbacks (EarlyStopping y ModelCheckpoint) permitió evitar sobreajuste, guardando automáticamente el mejor modelo entrenado.
 Esto asegura reproducibilidad y un desempeño óptimo en producción.

El flujo completo generó artefactos reutilizables (best_model.keras, preprocesador.joblib, label_encoder.joblib, gráficas y resumen en JSON),
 lo cual facilita tanto la trazabilidad como el despliegue futuro del modelo.

En conclusión, el pipeline de clasificación desarrollado es robusto, replicable y extensible a otros problemas de clasificación multiclase,
 siempre que se disponga de un dataset estructurado con suficientes ejemplos por clase.
```
