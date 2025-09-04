import numpy as np
from abc import ABC, abstractmethod

# MÉTRICAS:

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


# COSTO:

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


# ACTIVACIONES:

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


class Sigmoid(Activation):
    """Función de activación sigmoide."""

    def value(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def derivative(self, X: np.ndarray) -> np.ndarray:
        s = self.value(X)
        return s * (1 - s)


class Relu(Activation):
    """Función de activación ReLU."""

    def value(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    def derivative(self, X: np.ndarray) -> np.ndarray:
        return (X > 0).astype(float)


# CASOS DE PRUEBA:

if __name__ == "__main__":
    # Datos ficticios
    Y_true = np.array([[1,0,0],[0,1,0],[0,0,1]])
    Y_pred = np.array([[0.9,0.05,0.05],[0.1,0.8,0.1],[0.2,0.2,0.6]])

    # Prueba de métricas
    acc = Accuracy()
    print("Accuracy:", acc.value(Y_true, Y_pred))

    # Prueba de función de costo
    ce = CrossEntropy()
    print("CrossEntropy:", ce.value(Y_true, Y_pred))
    print("CrossEntropy derivada:", ce.derivative(Y_true, Y_pred))

    # Prueba de activaciones
    X = np.array([[-1, 0, 1]])
    sigmoid = Sigmoid()
    relu = Relu()

    print("Sigmoid:", sigmoid.value(X))
    print("Sigmoid derivada:", sigmoid.derivative(X))
    print("ReLU:", relu.value(X))
    print("ReLU derivada:", relu.derivative(X))
