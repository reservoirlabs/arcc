import random
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, TYPE_CHECKING

from src.arcc_main import get_logger
from src.assignment import Assignment

# only import TunableArg when type checking, to avoid circular imports
if TYPE_CHECKING:
    from src.production import TunableArg


class SearchStrategy(ABC):
    """
    Abstract base class for a strategy that finds an optimal assignment.
    """

    @abstractmethod
    def __init__(self, _root: "TunableArg"):
        """
        Must be able to initialize with just the tunable arguments.
        """
        pass

    @abstractmethod
    def get_assignment(self) -> Optional[Assignment]:
        """
        Return an assignment to be tested, or None if finished. Once None is
        returned, None should always be returned for subsequent calls (important
        for multi-threading).
        """
        pass

    @abstractmethod
    def notify_results(self, assignment: Assignment, runtime: float):
        """
        Notify the strategy with new results.
        """
        pass

    @abstractmethod
    def get_optimal(self) -> Optional[Tuple[Assignment, float]]:
        """
        Return the best known assignment, or None if it doesn't exist.
        """
        pass


class DummySearch(SearchStrategy):
    """
    Simple strategy that only runs one runs and doesn't store results. Useful
    for testing/debug.
    """

    def __init__(self, root: "TunableArg"):
        super().__init__(root)
        self.root = root
        self.has_returned = False

    def get_assignment(self) -> Optional[Assignment]:
        """
        Return an arbitrary assignment for each input
        """
        if self.has_returned:
            return None
        else:
            self.has_returned = True
            return self.root.get_arbitrary_assignment()

    def notify_results(self, assignment: Assignment, runtime: float):
        # don't do anything, since we're dummy
        pass

    def get_optimal(self) -> Optional[Tuple[Assignment, float]]:
        # we don't store the optimal assignment
        return None


class RandomSearch(SearchStrategy):
    """
    Simple strategy based on randomly searching the search space and returning
    the best result.
    """
    def __init__(self, root: "TunableArg"):
        super().__init__(root)
        self.root = root
        self.best_observed = None
        self.best_time = None

    def get_assignment(self) -> Optional[Assignment]:
        """
        Return an arbitrary assignment for each input
        """
        return self.root.get_arbitrary_assignment()

    def notify_results(self, assignment: Assignment, runtime: float):
        """
        Update the best known assignment if it has a better time
        """
        if self.best_time is None or self.best_time > runtime:
            self.best_observed = assignment
            self.best_time = runtime

    def get_optimal(self) -> Optional[Tuple[Assignment, float]]:
        """
        Return the best observed
        """
        if self.best_observed is None:
            return None
        else:
            return self.best_observed, self.best_time


class MutationSearch(SearchStrategy):
    """
    Mutation search. Start from an initial assignment (random), and randomly
    mutate to better times until multiple failed mutations in a row. When this
    occurs, restart the process.
    """
    def __init__(self, root: "TunableArg"):
        super().__init__(root)
        # mutates by 1/this for one of args to mutate
        self.MUTATION_GRANULARITY = 8
        # abandon after this many restarts
        self.MAX_RESTARTS = 5
        # restart after this many failed mutations
        self.MAX_FAILED_MUTATIONS = 3
        # current number of failed mutations/restarts
        self.failed_mutations = 0
        self.restarts = 0
        # time and assignment of the current "head"
        self.curr_time = None
        self.curr_assignment: Optional[Assignment] = None
        self.root = root
        # all tried assignments - this will have the global bests
        self.tried_assignments: Dict[Assignment, float] = {}

    def get_assignment(self) -> Optional[Assignment]:
        """
        get the next assignment to try. This is going to be a mutated version
        of the current head (if there is one), otherwise a new random assignment
        """
        if self.curr_assignment is None:
            get_logger().debug("restarting")
            self.restarts += 1
            self.failed_mutations = 0
            # once we've restarted enough times, give up
            if self.restarts >= self.MAX_RESTARTS:
                return None
            return self.root.get_arbitrary_assignment()
        else:
            mutated = self.curr_assignment.path_assignments()
            # only if there's anything to mutate
            assert len(mutated) > 0
            # choose a random argument to mutate, and mutate it
            to_mutate = random.randint(0, len(mutated) - 1)
            key, old_val = mutated[to_mutate]
            new_val = self.root.lookup(key).data_range\
                .mutate_value(old_val, self.MUTATION_GRANULARITY)
            mutated[to_mutate] = key, new_val
            # build and return the mutated assignment
            mutated = Assignment.from_path_assignments(mutated)
            return mutated

    def notify_results(self, assignment: Assignment, runtime: float):
        """
        Update the head with the new assignment if it's better. If it's worse,
        and we've had too many failed mutations, delete the current head. In any
        case, save the new runtime.
        """
        if self.curr_time is None or runtime < self.curr_time:
            # successful mutation found! reset the number of failed mutations
            # and update our head
            get_logger().debug("successful mutation found")
            self.failed_mutations = 0
            self.curr_assignment = assignment
            self.curr_time = runtime
        else:
            self.failed_mutations += 1
            if self.failed_mutations == self.MAX_FAILED_MUTATIONS:
                get_logger().debug("too many failed mutations, restarting")
                self.curr_assignment = None
                self.curr_time = None
        self.tried_assignments[assignment] = \
            min(self.tried_assignments.get(assignment, runtime), runtime)

    def get_optimal(self) -> Optional[Tuple[Assignment, float]]:
        """
        Get the global best assignment we've found so far. Because we save the
        best time for every assignment, just walk through and pick the best one.
        """
        best_assignment = None
        best_time = None
        for assignment, runtime in self.tried_assignments.items():
            if best_time is None or runtime < best_time:
                best_assignment = assignment
                best_time = runtime
        if best_assignment is None:
            return None
        else:
            return best_assignment, best_time
