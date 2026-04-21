from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np
from enum import Enum
import networkx as nx


class ConsumerType(Enum):
    HUMAN = "human"
    AI = "ai"


@dataclass
class Gaussian:
    """
    Represents a Gaussian distribution with mean and standard deviation.
    """
    mean: float
    std: float
    
    def sample(self) -> float:
        """Sample a value from the Gaussian distribution."""
        return np.random.normal(self.mean, self.std)


@dataclass
class Producer:
    """
    Represents a producer in the Mixed Customer Response simulation.
    """
    id: int


@dataclass
class Product:
    """
    Represents a product in the Mixed Customer Response simulation.
    """
    id: int
    price: float
    quality_prior: Gaussian
    producer: Producer

    def quality_at(self, quantity: int) -> Gaussian:
        """Calculate the quality of the product based on the quantity requested."""
        return Gaussian(
            mean=self.quality_prior.mean * (1 - np.exp(-quantity / 10)),
            std=self.quality_prior.std,
        )

@dataclass
class RankedMenu:
    """
    Represents a ranked menu of products for a consumer.
    """
    products: list['Product']


@dataclass
class UtilityModel:
    pass


@dataclass
class MNL(UtilityModel):
    """
    Represents the utility function for a consumer in the Mixed Customer Response simulation.
    """
    price_sensitivity: float
    quality_sensitivity: float
    ranking_sensitivity: float

    def utility(self, product: 'Product', ranking: RankedMenu) -> float:
        """Calculate the utility of a product for a consumer."""
        return (
            -self.price_sensitivity * product.price
            + self.quality_sensitivity * product.quality_prior.mean
            - self.ranking_sensitivity * ranking.products.index(product)
        )

    def choice_probabilities(self, products: list['Product'], ranking: RankedMenu) -> np.ndarray:
        """Calculate the choice probabilities for a list of products."""
        utilities = np.array([self.utility(product, ranking) for product in products])
        exp_utilities = np.exp(utilities - np.max(utilities))
        return exp_utilities / np.sum(exp_utilities)


@dataclass
class Demand:
    product: Product
    quantity: int


@dataclass
class Consumer:
    """
    Represents a consumer in the Mixed Customer Response simulation.
    """
    id: int
    demand: Demand
    type: ConsumerType
    prior: Gaussian
    utility_model: MNL


@dataclass
class Recommender:
    """
    Represents a recommender system in the Mixed Customer Response simulation.
    """
    signaling_strategy: Any
    prior: Gaussian


@dataclass
class Environment:
    """
    Environment configuration and state for the Mixed Customer Response simulation.
    """
    seed: int
    
    recommender: Optional[Recommender]
    consumers: list[Consumer]
    products: list[Product]
    
    network: nx.MultiDiGraph
    
    def __post_init__(self):
        np.random.seed(self.seed)

    def reset(self):
        """Reset the environment state."""
        pass

    def step(self, action):
        """Perform a step in the environment."""
        pass
