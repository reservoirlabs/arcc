import random
from abc import ABC, abstractmethod
from typing import List, Any


class DataRange(ABC):
    """
    Abstract base class of some range of data.
    """

    @abstractmethod
    def get_arbitrary(self) -> Any:
        """
        Get an arbitrary value.
        """
        pass

    @abstractmethod
    def mutate_value(self, value: Any, granularity: int) -> Any:
        """
        Perform a small mutation on a value and return the new value.
        """
        pass

    @abstractmethod
    def contains(self, val: Any) -> bool:
        """
        Check if value is contained in this range.
        """
        pass


class ContinuousRange(DataRange):
    """
    Continuous range of data. Floats between some lower and upper bound (inc).
    """
    def __init__(self, lb: float, ub: float):
        assert lb <= ub
        self.lb = lb
        self.ub = ub

    def __str__(self) -> str:
        return f"continuous [{self.lb}, {self.ub}]"

    def get_arbitrary(self) -> Any:
        return random.uniform(self.lb, self.ub)

    def mutate_value(self, value: Any, granularity: int) -> Any:
        """
        To mutate, get a new limited range centered at the current location,
        and get a uniform value up or down from that.
        """
        mutation_range = (self.ub - self.lb) / granularity
        new_lb = value - mutation_range
        new_ub = value + mutation_range
        val = ContinuousRange(new_lb, new_ub).get_arbitrary()
        # take into account our own bounds
        val = max(self.lb, val)
        val = min(self.ub, val)
        return val

    def contains(self, val: Any) -> bool:
        assert isinstance(val, float)
        return self.lb <= val <= self.ub


class IntegralRange(DataRange):
    """
    Integral range of data. Integers between some lower and upper bound (inc).
    """
    def __init__(self, lb: int, ub: int):
        assert lb <= ub
        self.lb = lb
        self.ub = ub

    def __str__(self) -> str:
        return f"integral [{self.lb}, {self.ub}]"

    def get_arbitrary(self) -> Any:
        return random.randint(self.lb, self.ub)

    def mutate_value(self, value: Any, granularity: int) -> Any:
        """
        To mutate, get a new limited range centered at the current location,
        and get a uniform value up or down from that.
        """
        # +7 to round up on mutation range
        mutation_range = (self.ub - self.lb + granularity - 1) // granularity
        new_lb = value - mutation_range
        new_ub = value + mutation_range
        val = IntegralRange(new_lb, new_ub).get_arbitrary()
        # take into account our own bounds
        val = max(self.lb, val)
        val = min(self.ub, val)
        return val

    def contains(self, val: Any) -> bool:
        assert isinstance(val, int)
        return self.lb <= val <= self.ub


class DiscreteRange(DataRange):
    """
    Discrete range of data. Some list of possibilities
    """

    def __init__(self, options: List[str]):
        self.options = options

    def __str__(self) -> str:
        return f"discrete {self.options}"

    def get_arbitrary(self) -> Any:
        return random.choice(self.options)

    def mutate_value(self, value: Any, granularity: int) -> Any:
        """
        To mutate, get our current index and mutate it.
        """
        index = self.options.index(value)
        new_index = IntegralRange(0, len(self.options) - 1)\
            .mutate_value(index, granularity)
        return self.options[new_index]

    def contains(self, val: Any) -> bool:
        assert isinstance(val, str)
        return val in self.options
