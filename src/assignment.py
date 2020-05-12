from collections import defaultdict
from typing import List, Any, Dict, Tuple, Union


class Assignment:
    """
    Represents an assignment of values to tunable arguments. Helpful to avoid
    hashing oddities and easy pretty printing. Maps the child arg to the
    assignment for that child, or the actual value if that child is a leaf.
    """
    def __init__(self, items: Dict[str, Union["Assignment", Any]]):
        self.items = items

    @classmethod
    def from_path_assignments(cls, paths: List[Tuple[List[str], Any]]) \
            -> "Assignment":
        """
        Build an assignment from a raw path assignment.
        """
        # this is the raw map that will be used at the end
        items = {}
        # because we need all paths with the same prefix to go together,
        # store them here
        raw = defaultdict(lambda: list())
        for path, val in paths:
            assert len(path) > 0
            # if the path has length one, there must not be any others with the
            # same path
            if len(path) == 1:
                assert path[0] not in items
                assert path[0] not in raw
                items[path[0]] = val
            else:
                raw[path[0]].append((path[1:], val))
        # for each path with the same prefix, build recursively the remainder
        for path0, remaining in raw.items():
            items[path0] = Assignment.from_path_assignments(remaining)
        return Assignment(items)

    def lookup(self, path: List[str]) -> Union["Assignment", Any]:
        """
        Lookup the assignment or value for a path.
        """
        assert len(path) > 0
        if len(path) == 1:
            return self.items[path[0]]
        else:
            return self.items[path[0]].lookup(path[1:])

    def dict(self) -> Dict[str, Union["Assignment", Any]]:
        """
        Build and return a dict the user can use.
        """
        return {key: val for key, val in self.items}

    def immutable(self):
        """
        Python doesn't have a `frozendict`, so model with a frozenset instead.
        """
        return frozenset(self.items.items())

    def as_json(self) -> Any:
        return {key: value.as_json() if isinstance(value, Assignment) else value
                for key, value in self.items.items()}

    def path_assignments(self) -> List[Tuple[List[str], Any]]:
        tr = []
        for child, val in self.items.items():
            if isinstance(val, Assignment):
                tr += [([child] + path, val)
                       for path, val in val.path_assignments()]
            else:
                tr += [([child], val)]
        return tr

    def flattened_str(self) -> str:
        return '\n'.join(f"{'.'.join(key)}: {val}"
                         for key, val in self.path_assignments())

    def __str__(self) -> str:
        return self.flattened_str()

    # forward hash/eq to the immutable representation
    def __hash__(self):
        return hash(self.immutable())

    def __eq__(self, other):
        return self.immutable() == other.immutable()
