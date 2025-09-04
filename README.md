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
  """ Abstracta: define entradas, salidas y el comportamiento inicial de los métodos clave para cualquier metrica
  Representa la metrica de una red neuronal
  """
  def use(self, name: str) -> self:
    """ obtiene metrica (OBJ) a partir del nombre
    Args:
      name (str): nombre esperado de la metrica
    Returns:
      self (Metric): objeto metrica
    """
    pass

  def value(self, Y: np.ndarray, Yp:np.ndarray):
    """ computa el desempeño (accuracy) de la red (> 0.6 es 1)
    Args:
      Y (ndarray): valores de salidas esperadas (etiquetadas)
      Yp (ndarray): valores de salidas obtenidas
    Return:
      A (float): valor del desempeño
    """
    pass
```


