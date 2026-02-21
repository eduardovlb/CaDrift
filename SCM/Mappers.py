import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from scipy.spatial.distance import cityblock, canberra, euclidean

from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR

from SCM.TargetFunctions import *
import copy

import random
from abc import ABC, abstractmethod

class Mapping(ABC):
    """
    Abstract base class for all mappers.
    """
    def __init__(self):
        self.fitted = True
        self.current_mean = 0
        self.rho = 0.5
        self.u = 0.05
        self.std = 0.1
        self.lastNoise = 0.0
        self.drift_direction = 0
        self.drift_rate = 0
        self.functions = [LinearFunction, PolynomialFunction, SineFunction, ThresholdFunction, RadialBasisFunction, CheckerboardFunction]
    
    @abstractmethod
    def map(self, X):
        """Map input X to output value."""
        raise NotImplementedError
    
    def get_current_mean(self) -> float:
        return self.current_mean
    
    @abstractmethod
    def drift(self, X, y, new_label_func=None):
        """Apply concept drift to the mapper."""
        raise NotImplementedError
    
    def is_fitted(self):
        """Check if the mapper is fitted."""
        return self.fitted
    
    @abstractmethod
    def generate_untrained_example(self, X):  
        raise NotImplementedError
    
    # @abstractmethod
    def drift_label_function(self):
        raise NotImplementedError
    
    @abstractmethod
    def fit(self, X, y):
        """Fit the mapper to data."""
        raise NotImplementedError
    
    def __str__(self):
        return self.__class__.__name__
    
class IncrementalMapping(Mapping):
    """
    Base class for mappers supporting incremental drift.
    Inherits from Mapping and adds incremental-specific logic.
    """
    def __init__(self):
        super().__init__()
        
    def reset_mean():
        raise NotImplementedError
    
    def get_current_mean(self):
        return self.current_mean
    
    def is_fitted(self):
        return self.fitted
    
    @abstractmethod
    def partial_fit(self, X, y):
        raise NotImplementedError
    
    def __str__(self):
        return self.__class__.__name__
    
class NormalMapper(Mapping):
    """A normal mapper that generates a random value from a normal distribution."""
    def __init__(self, mean: float = None, sigma: float = None, ewma_alpha: float = 0.05, rho: float = 0.5):
        super().__init__()
        self.rho = rho
        self.fitted = True
        self.a = mean if mean is not None else np.random.randint(-10, 10)
        self.sigma = sigma if sigma is not None else np.random.randint(1, 5)
        self.train_has_started = False
        self.ewma_alpha = ewma_alpha
        self.dynamic_mean = self.a
        self.current_value = 0
        
    def map(self, _) -> float:
        """Maps the input to a value from a normal distribution, rounded to 3 decimal places."""
        self.dynamic_mean = (1 - self.ewma_alpha) * self.dynamic_mean + self.ewma_alpha * np.random.normal(self.a, self.sigma)
        self.lastNoise = self.rho * self.lastNoise + np.random.normal(self.u, self.std)
        value = self.dynamic_mean + self.lastNoise
        return float(np.round(value, 3))
    
    def generate_untrained_example(self, X: np.ndarray) -> float:
        """Generates an untrained example by mapping the input X."""
        value = self.map(X)
        return np.round(value, 3)
    
    def drift(self) -> None:
        """Applies concept drift by changing the mean of the normal distribution."""
        self.a = np.random.uniform(-10, 10)
        if self.ewma_alpha == 0: # If no smoothing, reset dynamic mean immediately
            self.dynamic_mean = self.a
    
    def __str__(self) -> str:
        return "Normal Mapper"
    
    def fit(self, X, y) -> None:
        pass
    
class UniformMapper(Mapping):
    """A uniform mapper that generates values uniformly between two bounds."""
    def __init__(self, low: float = None, high: float = None, ewma_alpha: float = 0.05, rho: float = 0.5):
        super().__init__()
        self.rho = rho
        self.fitted = True
        self.a = low if low is not None else np.random.randint(-10, 0)
        self.b = high if high is not None else np.random.randint(0, 10)
        self.train_has_started = False
        self.ewma_alpha = ewma_alpha
        self.center = (self.a + self.b) / 2
        self.current_value = 0
        
    def map(self, _) -> float:
        """Maps the input to a value from a uniform distribution."""
        noise = np.random.uniform(self.a, self.b)
        self.center = (1 - self.ewma_alpha) * self.center + self.ewma_alpha * noise
        self.lastNoise = self.rho * self.lastNoise + np.random.normal(self.u, self.std)
        value = self.center + self.lastNoise
        return np.round(value, 3)
    
    def generate_untrained_example(self, X: np.ndarray) -> float:      
        """Generates an untrained example by mapping the input X."""     
        value = self.map(X)
        return np.round(value, 3)
    
    def drift(self) -> None:
        """Applies concept drift by changing the bounds of the uniform distribution."""
        self.a = np.random.randint(-10, 0)
        self.b = np.random.randint(0, 10)
        if self.ewma_alpha == 0: # If no smoothing, reset center immediately
            self.center = (self.a + self.b) / 2
    
    def __str__(self) -> str:
        return "Uniform Mapper"
    
    def fit(self, X, y) -> None:
        pass

class FunctionMapper(IncrementalMapping):
    def __init__(self, rho: float = 0.5, target_function: TargetFunction = LinearFunction()):
        super().__init__()
        self.rho = rho
        self.fitted = True
        self.a = None
        self.b = None
        self.train_has_started = False
        if target_function is not None:
            self.label_function = target_function
        else:
            self.label_function : TargetFunction = np.random.choice(self.functions)()
        # self.label_function : TargetFunction = LinearFunction()
        self.old_function = None

    def partial_fit(self, X=None, y=None) -> None:
        self.a += self.drift_direction * self.drift_rate

    def start_incremental_drift(self) -> None:
        self.drift_direction = np.random.choice([-1, 1])
        self.drift_rate = np.random.uniform(0.01, 0.1)

    def is_fitted(self) -> bool:
        return self.fitted
    
    # Function Mapper with Linear or Polynomial function is the only who supports structural drift so far
    def structural_drift(self) -> None:
        if not (isinstance(self.label_function, LinearFunction) or isinstance(self.label_function, PolynomialFunction)):
            return f"Structural drift not supported for this mapper."

        for i in range(self.a):
            if self.a[i] == 0: # Include new node if any weight is zero
                direction = np.random.choice([-1,1])
                self.a[i] += direction*np.random.rand()
                return
            
        # Remove the influence of a random parent.
        idx_to_remove = np.random.choice(list(range(len(self.a))))
        self.a[idx_to_remove] = 0

    # def counfounder_drift(self, confounder_node_idx: int = None):
    #     if confounder_node_idx is None:
    #         return # No node to drift
        
    #     if isinstance(self.label_function, LinearFunction):
    #         confounder_drift_direction = np.random.choice([-1, 1])
    #         self.a[confounder_node_idx] += confounder_drift_direction*np.random.rand()
    #         return
        
    #     # If function is not linear, simply change the label function (a more 'severe' confounder drift, work as an endogenous drift as well.)
    #     self.drift()

    
    def generate_untrained_example(self, X: np.ndarray) -> float:
        """Generates an untrained example by mapping the input X to the target function."""
        if (self.a is None or self.b is None):
            self.a = np.random.uniform(-1, 1, size=X.shape[0])
            # self.b = np.random.uniform(-1, 1, size=X.shape[0])
            self.b = np.random.uniform(-1, 1)
            self.train_has_started = True
          
        y = self.label_function.compute_function(X, self.a, self.b)
        return np.round(y, 3)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_parents = X.shape[1]
        if self.label_function is None:
            self.label_function = np.random.choice(self.functions)()
        self.fitted = True

    def drift(self, X, y, new_label_func=None) -> None:
        """Applies concept drift by changing the label function or weights of the linear or polynomial function."""

        self.old_function = copy.deepcopy(self.label_function)
        if self.label_function is LinearFunction or self.label_function is PolynomialFunction:
            drifted_index = np.random.choice(list(range(len(self.a))))
            drift_direction = np.random.choice([-1, 1])
            self.a[drifted_index] += drift_direction*np.random.rand()
            # self.a = np.random.uniform(-1, 1, size=X.shape[0])
            # self.b = np.random.uniform(-1, 1, size=X.shape[0])
            return

        if new_label_func is None:
            new_label_func : TargetFunction = np.random.choice(self.functions)()
            while new_label_func.__str__() == self.label_function.__str__():
                new_label_func : TargetFunction = np.random.choice(self.functions)()

        self.label_function = new_label_func

    def map(self, X: np.ndarray) -> float:
        """Maps the cause and effect relation from the parents nodes to this vertex."""
        if (not self.fitted):
            raise RuntimeError("Model not fitted.")
        # if X.shape[1] != self.n_parents:
        #     raise ValueError(f"Expected input with {self.n_parents} features, got {X.shape[1]}")
        
        if (self.a is None or self.b is None):
            self.a = np.random.uniform(-1, 1, size=X.shape[0])
            # self.b = np.random.uniform(-1, 1, size=X.shape[0])
            self.b = np.random.uniform(-1, 1)
            self.train_has_started = True
        
        self.lastNoise = self.rho * self.lastNoise +  np.random.normal(self.u, self.std)
        value =  self.label_function.compute_function(X.flatten(), self.a, self.b) + self.lastNoise
        return np.round(value, 3)

class RandomMLPMapper(Mapping):
    """A random MLP mapper that generates a random MLP model."""
    def __init__(self, hidden_dims: tuple =(10, 10), activation: str = 'tanh', rho: float = 0.5):
        super().__init__()
        self.rho = rho
        self.fitted = False
        self.train_has_started = False
        self.label_function: None
        self.old_function = None
        self.a = None
        self.b = None

        # Random MLP
        self.hidden_dims = hidden_dims
        self.activation_fn = np.tanh if activation == 'tanh' else lambda x: np.maximum(0, x)

        self.weights = []
        self.biases = []
        self.input_dim = None # To be set during fitting

    def start_incremental_drift(self) -> None:
        self.drift_direction = np.random.choice([-1, 1])
        self.drift_rate = np.random.uniform(0.01, 0.1)

    def _initialize_random_mlp(self, input_dim: int, hidden_dims: tuple =(10,)) -> None:
        """Initializes a random MLP with Xavier initialization."""
        self.weights = []
        self.biases = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            limit = np.sqrt(6 / (dims[i] + dims[i+1]))
            w = np.random.uniform(-limit, limit, size=(dims[i], dims[i+1]))
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)
        limit = np.sqrt(6 / (dims[-1] + 1))
        w = np.random.uniform(-limit, limit, size=(dims[-1], 1))
        b = np.zeros(1)
        self.weights.append(w)
        self.biases.append(b)

    def partial_fit(self, X=None, y=None) -> None:
        pass

    def _forward(self, X: np.ndarray) -> float:
        """Forward pass through the MLP."""
        out = X
        for i in range(len(self.weights) - 1):
            out = self.activation_fn(out @ self.weights[i] + self.biases[i])
        return (out @ self.weights[-1] + self.biases[-1]).ravel()

    def map(self, X: np.ndarray) -> float:
        """Maps the cause and effect relation from the parents nodes to this vertex."""
        if not self.fitted:
            self._initialize_random_mlp(input_dim=X.shape[1])
            self.fitted = True

        if X.shape[1] != self.n_parents:
            raise ValueError(f"Expected input with {self.n_parents} features, got {X.shape[1]}")
        
        self.lastNoise = self.rho * self.lastNoise + np.random.normal(self.u, self.std)
        value =  self._forward(X) + self.lastNoise
        return np.round(value, 3)

    def is_fitted(self) -> bool:
        return self.fitted

    def fit(self, X: np.ndarray=None, y=None) -> None:
        """Fit the mapper to data."""
        if X is not None:
            self._initialize_random_mlp(input_dim=X.shape[1])
            self.fitted = True
            self.n_parents = X.shape[1]

    def generate_untrained_example(self, X: np.ndarray) -> float:
        """Generates an untrained example by mapping the input X."""
        if self.a is None or self.b is None:
            self.a = np.random.uniform(-1, 1, size=X.shape[0])
            # self.b = np.random.uniform(-1, 1, size=X.shape[0])
            self.b = np.random.uniform(-1, 1)
            self.train_has_started = True
            self._initialize_random_mlp(input_dim=X.shape[0])
            self.n_parents = X.shape[0]
            self.fitted = True
        pred = self.map(X.reshape(1,-1))
        value =  float(pred)
        return np.round(value, 3)

    def drift(self, X: np.ndarray = None, y=None, new_label_func=None) -> None:
        """Applies concept drift by reinitializing the network."""
        if X is not None:
            self._initialize_random_mlp(input_dim=X.shape[1])

    def __str__(self) -> str:
        return "Random MLP Mapper"

class MLPMapping(IncrementalMapping):
    """A mapping that uses a Multi-Layer Perceptron (MLP) for regression."""
    def __init__(self, rho: float = 0.5):
        super().__init__()
        self.rho = rho
        self.fitted = False
        self.train_has_started = False
        self.model = None
        self.a = None
        self.b = None
        self.label_function : TargetFunction = np.random.choice(self.functions)()
        self.old_function = None
        self.model = self._generate_model()
        
    def generate_untrained_example(self, X: np.ndarray) -> float:
        """Generates an untrained example by mapping the input X to the target function."""
        if (self.a is None or self.b is None):
            self.a = np.random.uniform(-1, 1, size=X.shape[0])
            # self.b = np.random.uniform(-1, 1, size=X.shape[0])
            self.b = np.random.uniform(-1, 1)
            self.train_has_started = True
          
        y = self.label_function.compute_function(X, self.a, self.b)
        return np.round(y, 3)
    
        
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the mapper to data."""
        self.n_parents = X.shape[1]
        self.model = self._generate_model()
        self.model.fit(X,y)
        self.fitted = True
        self.current_mean = np.mean(X)
        
    def is_fitted(self) -> bool:
        return self.fitted
    
    def start_incremental_drift(self) -> None:
        self.drift_label_function()
    
    def partial_fit(self, X=None, y=None) -> None:
        """Incrementally fit the model to new data."""
        if X is not None and y is not None:
            X = np.array(X).reshape(1, -1)
            y = np.array([y])
            # print(y)
            if not self.train_has_started:
                self.n_parents = X.shape[1]
                self.train_has_started = True
            self.model.partial_fit(X, y)
            self.fitted = True
        else:
            if not self.train_has_started:
                raise ValueError("Need to train first with data")
            if self.a is None or self.b is None:
                self.a = np.random.uniform(-1, 1, size=self.n_parents)
                # self.b = np.random.uniform(-1, 1, size=self.n_parents)
                self.b = np.random.uniform(-1, 1)
                self.train_has_started = True
            X = np.random.normal(0, 1, size=(1, self.n_parents))
            # y = np.array([self.label_function.compute_function(X[i], self.a, self.b) for i in range(X.shape[0])])
            y = np.array([self.label_function.compute_function(X[0], self.a, self.b)])
            # print(y)
            self.model.partial_fit(X, y)
            self.fitted = True
        
    def map(self, X: np.ndarray) -> float:
        """Maps the cause and effect relation from the parents nodes to this vertex."""
        if (not self.fitted):
            raise RuntimeError("Model not fitted.")
        if X.shape[1] != self.n_parents:
            raise ValueError(f"Expected input with {self.n_parents} features, got {X.shape[1]}")
        
        self.lastNoise = self.rho * self.lastNoise +  np.random.normal(self.u, self.std)
        value =  self.model.predict(X) + self.lastNoise
        return float(np.round(np.asarray(value).ravel()[0], 3))
    
    def _generate_model(self) -> MLPRegressor:
        return MLPRegressor(hidden_layer_sizes=(10,), max_iter=10, solver='adam', learning_rate_init=0.001, warm_start=True)
    
    def drift_label_function(self, new_func: TargetFunction = None) -> None:
        if new_func is None:
            new_func : TargetFunction = np.random.choice(self.functions)()
            while new_func.__str__() == self.label_function.__str__():
                new_func : TargetFunction = np.random.choice(self.functions)()
        self.old_function = copy.deepcopy(self.label_function)
        self.label_function = new_func
    
    def drift(self, X: np.ndarray = None, y: np.ndarray = None, new_label_func: TargetFunction = None) -> None:
        """Apply concept drift to this mapper.

        Behavior:
        - If `new_label_func` is provided, replace the internal target/label
            function used to synthesize labels (simulates a label-function change).
        - The internal sklearn model is reinitialized and, if `(X, y)` are
            provided, retrained immediately on that data. If `(X, y)` are not
            provided, the model is replaced and will learn the new concept when
            `fit`/`partial_fit` is next called.

        This method simulates an abrupt drift in both the label-generating
        function and the fitted model parameters.
        """
        self.drift_label_function(new_label_func)
        self.model = self._generate_model()
        self.fit(X, y)
        
    def __str__(self) -> str:
        return "MLP Mapper"        

class TreeMapper(Mapping):
    """Decision tree based mapper.

    Uses a sklearn DecisionTreeRegressor to model the mapping from parent
    features to the child's value. Supports generating synthetic untrained
    examples via an internal target function and simulating label-function
    drift by replacing the target function or retraining the tree.
    """
    def __init__(self, rho: float = 0.5):
        super().__init__()
        self.rho = rho
        self.fitted = False
        self.a = None
        self.b = None
        self.train_has_started = False
        self.label_function : TargetFunction = np.random.choice(self.functions)()
        self.model = self._generate_model()

    def partial_fit(self, X=None, y=None) -> None:
        pass

    def start_incremental_drift(self) -> None:
        self.drift_direction = np.random.choice([-1, 1])
        self.drift_rate = np.random.uniform(0.01, 0.1)
        
    def is_fitted(self) -> bool:
        return self.fitted
    
    def generate_untrained_example(self, X: np.ndarray) -> float:           
        if (self.a is None or self.b is None):
            self.a = np.random.uniform(-1, 1, size=X.shape[0])
            # self.b = np.random.uniform(-1, 1, size=X.shape[0])
            self.b = np.random.uniform(-1, 1)
            self.train_has_started = True
          
        y = self.label_function.compute_function(X=X, a=self.a, b=self.b)
        return np.round(y, 3)
        
    def fit(self, X, y):
        self.n_parents = X.shape[1]
        if (not self.fitted):
            self.model = self._generate_model()
        
        self.model.fit(X,y)
        self.fitted = True
        self.current_mean = np.mean(X)
    
    def map(self, X: np.ndarray) -> float:
        """Maps the cause and effect relation from the parent(s) vertex(ices) to this vertex."""
        if (not self.fitted):
            raise RuntimeError("Model not fitted.")
        if X.shape[1] != self.n_parents:
            raise ValueError(f"Expected input with {self.n_parents} features, got {X.shape[1]}")
        self.lastNoise = self.rho * self.lastNoise +  np.random.normal(self.u, self.std)
        value =  self.model.predict(X) + self.lastNoise
        return float(np.round(np.asarray(value).ravel()[0], 3))
    
    def _generate_model(self) -> DecisionTreeRegressor:
        max_depth = np.random.randint(5, 25)
        return DecisionTreeRegressor(max_depth=max_depth)
    
    def drift_label_function(self, new_func: TargetFunction = None) -> None:
        if new_func is None:
            new_func : TargetFunction = np.random.choice(self.functions)()
            while new_func.__str__() == self.label_function.__str__():
                new_func : TargetFunction = np.random.choice(self.functions)()
        self.label_function = new_func
    
    def drift(self, X: np.ndarray = None, y: np.ndarray = None, new_label_func: TargetFunction = None) -> None:
        """Simulate concept drift for the tree mapper.

        Modes of operation:
        - If `new_label_func` is provided the target function used to
            synthesize untrained examples is replaced (label-function drift).
        - If `(X, y)` are provided the underlying decision tree is
            reinitialized and retrained on the provided data (abrupt model
            parameter drift).
        - If no `(X, y)` are provided, only the label-function may change and
            the fitted tree remains until retrained later.

        This method is intended to model abrupt or label-function changes in
        downstream nodes.
        """
        self.drift_label_function(new_func=new_label_func)
        if X is not None and y is not None:
                self.model = self._generate_model()
                self.model.fit(X, y)
        
    def __str__(self) -> str:
        return "Decision Tree Mapper"

class AbstractCategoricalMapper(Mapping):
    """Base class for categorical (discrete-label) mappers.

    This class provides utilities for sampling a number of classes (K),
    storing embedding vectors (optional), tracking swaps used for severe
    drift simulations, and sampling a label when no classifier is fitted.
    Concrete categorical mappers should implement `fit`, `map`, and
    `generate_untrained_example`.
    """
    def __init__(self, min_classes: int = 2, max_classes: int = 20, embed: bool = False):
        super().__init__()
        self.fitted = False
        self.n_parents = None
        self.K = None
        self.embed = embed
        self.embeddings = None
        self.min_classes = min_classes
        self.max_classes = max_classes
        self.class_swaps = {}
        self.n_classes = None

    def _sample_K(self) -> int:
        raw_k = int(np.round(np.random.gamma(2.0, 2.0))) + 2
        return np.clip(raw_k, self.min_classes, self.max_classes)

    def severe_drift(self) -> None:
        """Simulates severe drift.

        This operation swaps the mapping of two classes, which effectively
        permutes labels to simulate severe concept drift in the
        label space.
        """
        if self.K < 2:
            return
        class_a, class_b = np.random.choice(range(self.n_classes), size=2, replace=False)
        swap_a = self.class_swaps.get(class_a, class_a)
        swap_b = self.class_swaps.get(class_b, class_b)
        self.class_swaps[class_a] = swap_b
        self.class_swaps[class_b] = swap_a

    def drift_label_function(self) -> None:
        pass

    def sample_label(self) -> int:
        if hasattr(self, "class_weights") and self.class_weights is not None:
            probs = self.class_weights / self.class_weights.sum()
            return np.random.choice(self.K, p=probs)
        else:
            return np.random.randint(self.K)

class PrototypeCategoricalMapper(AbstractCategoricalMapper):
    """Categorical mapper using prototype vectors.

    This mapper maintains a set of prototype vectors in the parent feature
    space and assigns incoming examples to the nearest prototype. Each
    prototype maps to a class label. Supports incremental prototype
    updates and several types of simulated drift (prototype shifts,
    distance-function change, reinitialization from data).
    """
    def __init__(self, embed: bool = False, min_classes: int = 2, max_classes: int = 20, distance: str = "euclidean"):
        super().__init__(min_classes, max_classes, embed)
        self.prototypes = None
        self.parents_mean = None
        self.parents_count = 0
        self.distance = distance
        self.prototype_to_class = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> None:
        if X is None:
            raise ValueError("X must not be None")

        self.n_parents = X.shape[1]
        
        self.n_classes = self._sample_K()
        self.K = np.random.randint(self.n_classes, self.max_classes+1)

        self.prototype_to_class = np.random.choice(self.n_classes, size = self.K)

        # parent_mean = np.mean(X, axis=0)
        # parent_std = np.std(X, axis=0)

        self._initialize_centers(X)

        # scaling_factor = 0.5

        # self.prototypes = np.random.normal(
        #     loc=parent_mean,
        #     scale=scaling_factor * parent_std,
        #     size=(self.K, self.n_parents)
        # )

        if self.embed:
            self.embeddings = np.random.normal(0, 1, size=(self.K, 4))

        self.fitted = True
        self.class_swaps = {}
        self.current_mean = np.mean(X)

    def is_fitted(self) -> bool:
        return self.fitted
    
    def _initialize_centers(self, X: np.ndarray) -> None:
        parent_mean = np.mean(X, axis=0)
        parent_std = np.std(X, axis=0) * 1.0 # Multiply by std to allow for farther prototypes for evolving after drift events
        parent_std = np.where(parent_std < 1e-6, 1.0, parent_std)

        scaling_factor = 0.5

        self.prototypes = np.random.normal(
            loc=parent_mean,
            scale=scaling_factor * parent_std,
            size=(self.K, self.n_parents)
        )  

    # def _initialize_centers(self, X: np.ndarray) -> None:
    #     parent_mean = np.mean(X, axis=0)
    #     self.n_parents = X.shape[1]

    #     # --- Data geometry ---
    #     data_radii = np.linalg.norm(X - parent_mean, axis=1)
    #     R_data = np.percentile(data_radii, 90)

    #     # Typical spacing
    #     base_scale = np.percentile(data_radii, 50)

    #     # --- Adaptive parameters ---
    #     shell_width = 0.5 * R_data                    # not huge
    #     inner_radius = 1.05 * R_data                  # just outside support
    #     outer_radius = inner_radius + shell_width

    #     # Repulsion shrinks as K grows
    #     min_proto_dist = base_scale * (0.6 / np.sqrt(self.K))
    #     print(f"min_proto_dist: {min_proto_dist}")

    #     max_trials = 100
    #     prototypes = []

    #     for _ in range(self.K):
    #         for _ in range(max_trials):
    #             direction = np.random.normal(0, 1, self.n_parents)
    #             direction /= np.linalg.norm(direction) + 1e-8

    #             radius = np.random.uniform(inner_radius, outer_radius)
    #             candidate = parent_mean + radius * direction

    #             if prototypes:
    #                 dists = np.linalg.norm(np.asarray(prototypes) - candidate, axis=1)
    #                 if np.min(dists) < min_proto_dist:
    #                     continue

    #             prototypes.append(candidate)
    #             break
    #         else:
    #             # fallback: slight radial jitter near boundary
    #             prototypes.append(
    #                 parent_mean + inner_radius * direction
    #             )

    #     self.prototypes = np.asarray(prototypes)

    
    def drift_label_function(self) -> None:
        pass

    def drift(self, X: np.ndarray = None, y: np.ndarray = None, new_label_func: TargetFunction = None) -> None:
        """Simulate concept drift for prototype-based categorical mapping.

        This mapper supports several drift behaviours chosen randomly
        (unless `X` is provided which forces reinitialization from data):

        - Prototype perturbation (shift a prototype randomly): models a
          change in the class distribution in feature space (abrupt/gradual).
        - Distance function change (e.g., switch from euclidean to
          manhattan): models a change in the decision rule.
        - Reinitialization from `X`: re-compute prototype centers from new
          data, useful to simulate abrupt reconfiguration.

        The `partial_fit` method implements incremental prototype shifts to
        simulate incremental drift over many small updates.
        """
        num = np.random.rand()
        if num < 0.33:
            self._change_prototypes()
            return
        elif num < 0.66:
            self._change_distance()
            return
        elif X is not None:
            self._initialize_centers(X)
            return

        self._change_prototypes()

    def start_incremental_drift(self) -> None:
        pass

    def partial_fit(self, X= None, y=None, step_size: float = 0.1) -> None:
        """Incrementally update prototypes to simulate incremental drift."""
        shifting_class = np.random.choice(range(self.K))
        shift_vector = np.random.normal(0, step_size, size=self.prototypes[shifting_class].shape)
        
        self.prototypes[shifting_class] += shift_vector#*step_size
            
    def _change_prototypes(self) -> None:
        """Simulate concept drift by shifting a prototype randomly."""
        max_shift = 1.0
        drifting_class = np.random.choice(range(self.K))
        shift_vector = np.random.normal(0, max_shift, size=self.prototypes[drifting_class].shape)
        
        new_position = self.prototypes[drifting_class] + shift_vector
                
        self.prototypes[drifting_class] = new_position
        
        
    def _change_distance(self, new_distance: str = None) -> None:
        """Simulate concept drift by changing the distance function"""
        all_distances = ["euclidean", "manhattan", "canberra"]
        
        if new_distance is None:
            options = [d for d in all_distances if d != self.distance]
            self.distance = np.random.choice(options)
        else:
            if new_distance not in all_distances:
                raise ValueError("Unsupported distance type.")
            self.distance = new_distance

    def generate_untrained_example(self, X: np.ndarray) -> int:
        """Generates an untrained example by mapping the input X to the nearest prototype."""
        X = X.flatten()

        if self.prototypes is None:
            self.n_parents = X.shape[0]
            self.n_classes = self._sample_K()
            self.K = np.random.randint(self.n_classes, self.max_classes+1)
            self.prototype_to_class = np.random.choice(self.n_classes, size=self.K)
            parent_mean = X
            parent_std = np.ones_like(parent_mean)

            scaling_factor = 0.5

            self.prototypes = np.random.normal(
                loc=parent_mean,
                scale=scaling_factor * parent_std,
                size=(self.K, self.n_parents)
            )

            if self.embed:
                self.embeddings = np.random.normal(0, 1, size=(self.K, 4))        

        dists = np.linalg.norm(self.prototypes - X, axis=1)
        idx = np.argmin(dists)
        
        self.prototypes[idx] = (self.prototypes[idx] + X) / 2

        # return self.embeddings[idx] if self.embed else idx
        return idx

    def map(self, X: np.ndarray) -> int:
        """Maps the cause and effect relation from the parent(s) vertex(ices) to this vertex. Classes are determined by nearest prototype."""
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        if X is None:
            raise ValueError("Cannot map from None input.")
        if X.shape[1] != self.n_parents:
            raise ValueError(f"Expected input with {self.n_parents} features, got {X.shape[1]}")

        dists = self._compute_distance(X, self.prototypes)
        idx = np.argmin(dists)
        class_idx = self.prototype_to_class[idx]

        # Apply severe drift swaps if any
        final_idx = self.class_swaps.get(class_idx, class_idx)

        return self.embeddings[final_idx] if self.embed else final_idx
        
    def _compute_distance(self, X: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        """Compute distances between input X and prototypes based on the selected distance metric."""
        if self.distance == "euclidean":
            return np.linalg.norm(prototypes - X, axis=1)
        elif self.distance == "manhattan":
            return np.sum(np.abs(prototypes - X), axis=1)
        elif self.distance == "canberra":
            return np.sum(np.abs(prototypes - X) / (np.abs(prototypes) + np.abs(X) + 1e-8), axis=1)
        else:
            raise ValueError(f"Unsupported distance: {self.distance}")
    
    def __str__(self) -> str:
        return "Categorical Mapper"
    
    def sample_label(self) -> int:
        return np.random.randint(self.K)

class SGDMapper(IncrementalMapping):
    """Incremental linear mapper using SGDRegressor.

    Implements a simple linear regression using sklearn's SGDRegressor that
    supports `partial_fit` style updates and simulated drifts by replacing
    the underlying label function or retraining the estimator.
    """
    def __init__(self, rho = 0.5):
        super().__init__()
        self.rho = rho
        self.fitted = False
        self.train_has_started = False
        self.model = None
        self.a = None
        self.b = None
        self.label_function : TargetFunction = np.random.choice(self.functions)()
        self.old_function = None
        self.output_mean = 0
        self.num_samples_seen = 0
        
    def generate_untrained_example(self, X: np.ndarray) -> float:           
        """Generates an untrained example by mapping the input X to the target function."""
        if (self.a is None or self.b is None):
            self.a = np.random.uniform(-1, 1, size=X.shape[0])
            # self.b = np.random.uniform(-1, 1, size=X.shape[0])
            self.b = np.random.uniform(-1, 1)
            self.train_has_started = True
          
        y = self.label_function.compute_function(X, self.a, self.b)
        return y
    
    def start_incremental_drift(self) -> None:
        """Initiates incremental drift by changing the label function."""
        self.drift_label_function()

    def partial_fit(self, X=None, y=None) -> None:
        """Incrementally fit the model to new data, simulating incremental drift."""
        if X is not None and y is not None:
            X = np.array(X).reshape(1, -1)
            y = np.array([y])
            # print(y)
            if not self.train_has_started:
                self.model = self._generate_model()
                self.n_parents = X.shape[1]
                self.train_has_started = True
            self.model.partial_fit(X, y)
        else:
            if not self.train_has_started:
                raise ValueError("Need to train first with data")
            if self.a is None or self.b is None:
                self.a = np.random.uniform(-1, 1, size=self.n_parents)
                self.b = np.random.uniform(-1, 1)
                self.train_has_started = True
            X = np.random.normal(0, 1, size=(1, self.n_parents))
            y = np.array([self.label_function.compute_function(X[0], self.a, self.b)])
            # print(y)
            self.model.partial_fit(X, y)
        
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the mapper to data."""
        self.n_parents = X.shape[1]
        if (not self.fitted):
            self.model = self._generate_model()
        
        self.model.fit(X,y)
        self.fitted = True
        
    def is_fitted(self) -> bool:
        return self.fitted
    
    def reset_mean(self) -> None:
        self.output_mean = 0
        self.num_samples_seen = 0
        
    def map(self, X: np.ndarray) -> float:
        """Maps the cause and effect relation from the parent(s) vertex(ices) to this vertex."""
        if (not self.fitted):
            raise RuntimeError("Model not fitted.")
        if X.shape[1] != self.n_parents:
            raise ValueError(f"Expected input with {self.n_parents} features, got {X.shape[1]}")
        
        self.lastNoise = self.rho * self.lastNoise +  np.random.normal(self.u, self.std)

        prediction = self.model.predict(X) + self.lastNoise

        new_value = prediction.mean() if isinstance(prediction, np.ndarray) else prediction
        self.num_samples_seen += 1
        self.output_mean += (new_value - self.output_mean) / self.num_samples_seen

        return float(np.round(np.asarray(new_value).ravel()[0], 3))
    
    def _generate_model(self) -> SGDRegressor:
        return SGDRegressor(max_iter=10)
    
    def drift_label_function(self, new_func: TargetFunction = None) -> None:
        """Drifts the mapping function of the current vertex."""
        if new_func is None:
            new_func : TargetFunction = np.random.choice(self.functions)()
            while new_func.__str__() == self.label_function.__str__():
                new_func : TargetFunction = np.random.choice(self.functions)()
        self.old_function = copy.deepcopy(self.label_function)
        self.label_function = new_func
    
    def drift(self, X: np.ndarray = None, y: np.ndarray = None, new_label_func: TargetFunction = None) -> None:
        """Drifts the mapping function of the current vertex."""
        self.drift_label_function(new_label_func)
        if X is None or y is None:
            X = np.random.normal(0, 1, size=(1000,self.n_parents))
            y = np.random.normal(0, 1, 1000)        
        self.model = self._generate_model()
        self.model.fit(X, y)
        
    def __str__(self) -> str:
        return "SGD Regressor Mapper"
    

class OnlineGaussianCategoricalMapper(AbstractCategoricalMapper):
    """Online Gaussian-based categorical mapper.

    Each class/component is represented by an online estimate of a Gaussian
    mean and variance in the parent feature space. Incoming examples update
    the component statistics incrementally, making this mapper suitable for
    streaming scenarios and incremental drift simulations.
    """
    def __init__(self, min_classes=2, max_classes=20, embed=False):
        super().__init__(min_classes, max_classes, embed)
        self.class_means = None
        self.class_vars = None
        self.class_counts = None
        self.component_to_class = None

    def fit(self, X: np.ndarray, y=None) -> None:
        """Fit the online Gaussian categorical mapper to data."""
        if X is None:
            raise ValueError("X must not be None")
        self.n_parents = X.shape[1]
        self.n_classes = self._sample_K()
        self.K = np.random.randint(self.n_classes, self.max_classes+1)
        
        self.component_to_class = np.random.choice(self.n_classes, size=self.K)

        self.class_means = np.random.normal(loc=np.mean(X, axis=0), scale=0.5, size=(self.K, self.n_parents))
        self.class_vars = np.ones((self.K, self.n_parents))
        self.class_counts = np.ones(self.K) 

        if self.embed:
            self.embeddings = np.random.normal(0, 1, size=(self.K, 4))
            
        self.class_swaps = {}

        self.fitted = True
        
    def _initialize_centers(self, X: np.ndarray) -> None:
        """Initialize class centers from data X."""
        self.class_means = np.random.normal(loc=np.mean(X, axis=0), scale=0.5, size=(self.K, self.n_parents))
        self.class_vars = np.ones((self.K, self.n_parents))
        self.class_counts = np.ones(self.K) 

    def generate_untrained_example(self, X: np.ndarray) -> int:
        """Generates an untrained example by mapping the input X to the nearest class based on Gaussian likelihoods."""
        X = X.flatten()

        if self.class_means is None:
            self.n_parents = X.shape[0]
            self.n_classes = self._sample_K()
            self.K = np.random.randint(self.n_classes, self.max_classes+1)
            
            self.component_to_class = np.random.choice(self.n_classes, size=self.K)

            self.class_means = np.random.normal(loc=X, scale=0.5, size=(self.K, self.n_parents))
            self.class_vars = np.ones((self.K, self.n_parents))
            self.class_counts = np.ones(self.K)

            if self.embed:
                self.embeddings = np.random.normal(0, 1, size=(self.K, 4))

        class_likelihoods = np.zeros(self.n_classes)
        for k in range(self.K):
            var = np.maximum(self.class_vars[k], 1e-6)
            X_reshaped = X.reshape(-1, self.n_parents)
            exponent = -0.5 * np.sum((X_reshaped - self.class_means[k]) ** 2 / var, axis=1)
            coeff = -0.5 * np.sum(np.log(2 * np.pi * var))
            likelihood = np.exp(coeff + exponent)
            class_likelihoods[self.component_to_class[k]] += likelihood

        class_idx = np.argmax(class_likelihoods, axis=0)

        self.class_counts[class_idx] += 1
        alpha = 1.0 / self.class_counts[class_idx]

        old_mean = self.class_means[class_idx].copy()
        self.class_means[class_idx] = (1 - alpha) * self.class_means[class_idx] + alpha * X
        self.class_vars[class_idx] = (1 - alpha) * self.class_vars[class_idx] + alpha * (X - old_mean) ** 2

        return self.embeddings[class_idx] if self.embed else class_idx

    def map(self, X: np.ndarray) -> int:
        """Maps the cause and effect relation from the parent(s) vertex(ices) to this vertex. Classes are determined by Gaussian likelihoods."""
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        if X is None:
            raise ValueError("Cannot map from None input.")
        if X.shape[1] != self.n_parents:
            raise ValueError(f"Expected input with {self.n_parents} features, got {X.shape[1]}")

        class_likelihoods = np.zeros(self.n_classes)
        for k in range(self.K):
            var = np.maximum(self.class_vars[k], 1e-6)
            X_reshaped = X.reshape(-1, self.n_parents)
            exponent = -0.5 * np.sum((X_reshaped - self.class_means[k]) ** 2 / var, axis=1)
            coeff = -0.5 * np.sum(np.log(2 * np.pi * var))
            likelihood = np.exp(coeff + exponent)
            class_likelihoods[self.component_to_class[k]] += likelihood

        class_idx = np.argmax(class_likelihoods, axis=0)

        if np.isscalar(class_idx):
            final_idx = self.class_swaps.get(class_idx, class_idx)
        else:
            final_idx = np.array([self.class_swaps.get(i, i) for i in class_idx])

        return self.embeddings[final_idx] if self.embed else final_idx
    
    def drift_label_function(self) -> None:
        pass

    def drift(self, X: np.ndarray = None, y=None, new_label_func=None) -> None:
        """Simulate concept drift for the online Gaussian categorical mapper.

        Behavior:
        - With probability 0.5 perform an abrupt shift of a randomly chosen
            class mean: add a random vector to the class centroid (simulates
            class appearance change).
        - Otherwise reinitialize component centers from `X` (if `X` is
            provided), simulating a larger reconfiguration of the components.

        Incremental drift can be simulated via `start_incremental_drift()`
        followed by repeated `partial_fit()` calls which slightly shift class
        means over time.
        """
        num = np.random.rand()
        if (num < 0.5):        
            max_shift = 1.0
            drifting_class = np.random.choice(range(self.K))

            shift_vector = np.random.normal(0, max_shift, size=self.class_means[drifting_class].shape)

            self.class_means[drifting_class] += shift_vector
        else:
            self._initialize_centers(X)        

    def start_incremental_drift(self) -> None:
        pass

    def partial_fit(self, X=None, y=None, step_size=0.01) -> None:
        """Incrementally update class means to simulate incremental drift."""
        shifting_class = np.random.choice(range(self.K))
        shift_vector = np.random.normal(0, step_size, size=self.class_means[shifting_class].shape)
        
        self.class_means[shifting_class] += shift_vector*step_size

    def __str__(self):
        return "Online Gaussian Categorical Mapper"

class RandomRBFCategoricalMapper(AbstractCategoricalMapper):
    """RBF-based categorical mapper.

    Uses radial-basis responses (Gaussian kernels) around class centers to
    compute soft-responses; the highest-response component determines the
    predicted label. Supports incremental updates and concept-drift
    operations (center shifts or reinitialization).
    """
    def __init__(self, min_classes: int = 2, max_classes: int = 20, embed: bool = False):
        super().__init__(min_classes, max_classes, embed)
        self.class_means = None
        self.radii = None
        self.class_counts = None
        self.component_to_class = None
        self.n_partial_fit_calls = 0

    def _sample_K(self) -> int:
        """Sample the number of classes K from a gamma distribution."""
        raw_k = int(np.round(np.random.gamma(2.0, 2.0))) + 2
        return np.clip(raw_k, self.min_classes, self.max_classes)

    def fit(self, X: np.ndarray, y=None) -> None:
        """Fit the RBF categorical mapper to data."""
        if X is None:
            raise ValueError("X must not be None")
        self.n_parents = X.shape[1]
        self.n_classes = self._sample_K()
        self.K = np.random.randint(self.n_classes, self.max_classes+1)

        self.component_to_class = np.random.choice(self.n_classes, size=self.K)

        self._initialize_centers(X)

        if self.embed:
            self.embeddings = np.random.normal(0, 1, size=(self.K, 4))

        self.class_swaps = {}
        self.fitted = True
        
    def _initialize_centers(self, X: np.ndarray) -> None:
        """Initialize class centers from data X."""
        self.class_means = np.random.normal(loc=np.mean(X, axis=0), scale=0.5, size=(self.K, self.n_parents))
        self.radii = np.full(self.K, np.std(X))
        self.class_counts = np.ones(self.K)

    def generate_untrained_example(self, X: np.ndarray) -> int:
        """Generates an untrained example by mapping the input X to the nearest class based on RBF responses."""
        X = X.flatten()

        if self.class_means is None:
            self.n_parents = X.shape[0]
            self.K = self._sample_K()
            self.class_means = np.random.normal(loc=X, scale=0.5, size=(self.K, self.n_parents))
            self.radii = np.ones(self.K)
            self.class_counts = np.ones(self.K)

            if self.embed:
                self.embeddings = np.random.normal(0, 1, size=(self.K, 4))

        dists = np.linalg.norm(self.class_means - X, axis=1)
        responses = np.exp(-dists**2 / (2 * self.radii**2))
        idx = np.argmax(responses)

        self.class_counts[idx] += 1
        alpha = 1.0 / self.class_counts[idx]

        old_mean = self.class_means[idx].copy()
        self.class_means[idx] = (1 - alpha) * self.class_means[idx] + alpha * X

        dist_to_mean = np.linalg.norm(X - old_mean)
        self.radii[idx] = (1 - alpha) * self.radii[idx] + alpha * dist_to_mean

        return self.embeddings[idx] if self.embed else idx
    
    def drift_label_function(self) -> None:
        pass

    def map(self, X: np.ndarray) -> int:
        """Maps the cause and effect relation from the parent(s) vertex(ices) to this vertex. Classes are determined by RBF responses."""
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        if X is None:
            raise ValueError("Cannot map from None input.")
        if X.shape[1] != self.n_parents:
            raise ValueError(f"Expected input with {self.n_parents} features, got {X.shape[1]}")

        dists = []
        for k in range(self.K):
            X_reshaped = X.reshape(-1, self.n_parents)
            dist = np.linalg.norm(X_reshaped - self.class_means[k], axis=1)
            response = np.exp(-dist**2 / (2 * (self.radii[k]**2 + 1e-6)))
            dists.append(response)

        dists = np.array(dists)
        idx = np.argmax(dists, axis=0)
        class_idx = self.component_to_class[idx]
        final_idx = np.array([self.class_swaps.get(i, i) for i in class_idx])

        return self.embeddings[final_idx] if self.embed else final_idx

    def drift(self, X: np.ndarray = None, y=None, new_label_func: TargetFunction = None) -> None:
        """Abrupt drift: randomly shift a class center or reinitialize.

        This method either applies a relatively large random displacement to
        a single class centroid (simulating an abrupt class appearance
        change), or reinitializes the component centers from `X` which models
        a larger reconfiguration. Use `partial_fit` for smooth incremental
        drift instead.
        """
        num = np.random.rand()
        if (num < 0.5):
            max_shift = 1.0
            drifting_class = np.random.choice(range(self.K))
            shift_vector = np.random.normal(0, max_shift, size=self.class_means[drifting_class].shape)
            self.class_means[drifting_class] += shift_vector
        else:
            self._initialize_centers(X)

    def start_incremental_drift(self) -> None:
        """Prepare for incremental drift by resetting shift tracking."""
        self.n_partial_fit_calls = 0
        if hasattr(self, "incremental_shift_vectors"):
            del self.incremental_shift_vectors

    def partial_fit(self, X: np.ndarray = None, y=None, step_size: float = 0.001) -> None:
        """Apply a small, incremental shift to a selected class."""
        if not hasattr(self, "incremental_shift_vectors"):
            self.incremental_shift_vectors = np.zeros_like(self.class_means)
            self.incremental_classes = range(self.K) 

        for c in self.incremental_classes:
            shift_vector = np.random.normal(0, 1, size=self.class_means[c].shape)
            shift_vector /= np.linalg.norm(shift_vector) 
            self.incremental_shift_vectors[c] = shift_vector * step_size

        for c in self.incremental_classes:
            self.class_means[c] += self.incremental_shift_vectors[c]
        self.n_partial_fit_calls += 1

    def save_concept(self) -> None:
        """Save the current concept (class means and radii)."""
        self._saved_class_means = self.class_means.copy()
        self._saved_radii = self.radii.copy()

    def restore_concept(self) -> None:
        """Restore the saved concept (class means and radii)."""
        if hasattr(self, "_saved_class_means"):
            self.class_means = self._saved_class_means.copy()
        if hasattr(self, "_saved_radii"):
            self.radii = self._saved_radii.copy()

    def __str__(self) -> str:
        return "RandomRBF Categorical Mapper"

class RotatingHyperplaneMapper(AbstractCategoricalMapper):
    """Binary rotating-hyperplane mapper.

    Implements a simple binary classifier defined by a hyperplane that can
    be rotated over time to simulate gradual/incremental concept drift.
    Optional label embeddings are supported via `embed=True`.
    """
    def __init__(self, noise_rate: float = 0.0, margin: float = 0.0, rotation_speed: float = 0.05, embed: bool = False):
        super().__init__(min_classes=2, max_classes=2, embed=embed) 
        self.noise_rate = noise_rate
        self.margin = margin
        self.rotation_speed = rotation_speed
        self.fitted = False
        self.K = 2

    def _sample_K(self) -> int:
        """Sample the number of classes K. Always returns 2 for binary classification."""
        return 2  # current version of rotating hyperplane only supports binary classification

    def fit(self, X: np.ndarray, y=None) -> None:
        """Fit the rotating hyperplane mapper to data."""
        if X is None:
            raise ValueError("X must not be None")

        self.n_parents = X.shape[1]
        self.n_classes = 2

        w = np.random.normal(0, 1, size=self.n_parents)
        self.w = w / np.linalg.norm(w)
        self.bias = 0.0

        if self.embed:
            self.embeddings = np.random.normal(0, 1, size=(self.n_classes, 4))

        self.fitted = True

    def is_fitted(self) -> bool:
        return self.fitted

    def generate_untrained_example(self, X) -> int:
        return 0

    def map(self, X) -> int:
        """Maps the cause and effect relation from the parent(s) vertex(ices) to this vertex using the rotating hyperplane."""
        if not self.fitted:
            raise RuntimeError("Mapper not fitted.")
        if X is None:
            raise ValueError("Cannot map from None input.")
        if X.shape[1] != self.n_parents:
            raise ValueError(f"Expected input with {self.n_parents} features, got {X.shape[1]}")

        scores = X @ self.w - self.bias
        labels = (scores >= 0).astype(int)

        if self.margin > 0:
            mask = np.abs(scores) < self.margin
            labels[mask] = np.random.randint(0, 2, size=mask.sum())

        if self.noise_rate > 0:
            flip_mask = np.random.rand(len(labels)) < self.noise_rate
            labels[flip_mask] = 1 - labels[flip_mask]

        return self.embeddings[labels] if self.embed else labels

    def drift(self, X=None, y=None, new_label_func=None) -> None:
        """Abrupt drift: reinitialize the hyperplane randomly."""
        w = np.random.normal(0, 1, size=self.n_parents)
        self.w = w / np.linalg.norm(w)

    def start_incremental_drift(self) -> None:
        if hasattr(self, "rotation_axes"):
            del self.rotation_axes

    def partial_fit(self, X=None, y=None, step_size: float = 0.05, _=None) -> None:
        """
        Incremental drift: rotate the hyperplane a bit.

        If there is only one parent feature, skip rotation (no axes to rotate).
        This avoids ValueError when n_parents < 2.
        """
        if self.n_parents < 2:
            # Not enough dimensions to rotate
            return
        if not hasattr(self, "rotation_axes"):
            self.rotation_axes = np.random.choice(self.n_parents, 2, replace=False)

        i, j = self.rotation_axes
        angle = self.rotation_speed * step_size
        R = np.eye(self.n_parents)
        R[i, i] = np.cos(angle)
        R[j, j] = np.cos(angle)
        R[i, j] = -np.sin(angle)
        R[j, i] = np.sin(angle)

        self.w = R @ self.w
        self.w /= np.linalg.norm(self.w)

    def save_concept(self) -> None:
        """Save the current concept (hyperplane parameters)."""
        self._saved_w = self.w.copy()
        self._saved_bias = self.bias

    def restore_concept(self) -> None:
        """Restore the saved concept (hyperplane parameters)."""
        if hasattr(self, "_saved_w"):
            self.w = self._saved_w.copy()
        if hasattr(self, "_saved_bias"):
            self.bias = self._saved_bias

    def __str__(self) -> str:
        return "Rotating Hyperplane Mapper"

    def sample_label(self) -> int:
        return np.random.randint(0, 2)