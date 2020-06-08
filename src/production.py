import itertools
import json
import os
import pathlib
import re
import shutil
import tempfile
from abc import ABC
from pathlib import Path
from typing import List, Optional, Any, Dict, Iterator, Union, Callable

import src.meta_data_info
import src.meta_data_parser
from src.arcc_main import get_logger
from src.assignment import Assignment
from src.data_range import DataRange, ContinuousRange, DiscreteRange, \
    IntegralRange


class ArgFile(ABC):
    pass


class ClassicArgFile(ArgFile):
    def __init__(self, file):
        self.file = file


class NewArgFile(ArgFile):
    def __init__(self, file):
        self.file = file


class RstreamArgFile(ArgFile):
    pass


class StringFormatter:
    """
    Wrapper around a string that's meant to be formatted in the context of an
    argument. For example, "{VAL} < {CHILDA} + {CHILDB}" when expanded with an
    arg would expand to the string where $VAL is substituted for the value of
    the current node and $CHILDA and $CHILDB are substituted for the values of
    the children with the same name, or error if they don't exist.
    """
    def __init__(self, format_str: str):
        self.format_str = format_str

    def expand_assignment(self, assignment: Union[Any, "Assignment"]) -> str:
        def find_replacement(match) -> str:
            """
            regex sub requires a function that takes in a match and returns
            a replacement string for it. Get the match inside the {} and
            look up inside the assignment the actual value
            """
            var = match.group(1)
            path = var.split('.')
            # SELF is a special case - we should be a value already
            if var == "SELF":
                assert not isinstance(assignment, Assignment)
                val = assignment
            else:
                # otherwise, lookup the path, and make sure it's a value
                assert isinstance(assignment, Assignment)
                val = assignment.lookup(path)
                assert not isinstance(val, Assignment)
            return str(val)
        return re.sub(r"{(\w+)}", find_replacement, self.format_str)

    def expand_dict(self, with_dict: Dict[str, str]) -> str:
        """
        expand the key, but using a known dictionary to find substitutions. in
        theory, expand_assignment could be rewritten by building the full dict
        and then calling this, but that could be quite slow.
        """
        return re.sub(r"{(\w+)}", lambda match: with_dict[match.group(1)],
                      self.format_str)


class TunableArg:
    """
    Class representing an argument or group of arguments that can be
    automatically tuned.
    """
    def __init__(self, name: str, env: Dict[StringFormatter, StringFormatter],
                 constraint: Optional[StringFormatter],
                 children: List["TunableArg"],
                 data_range: Optional[DataRange]):
        # name given to this arg. used for setting command-line/environment.
        # NOTE: the "full name" will be ...GRANDPARENT_NAME.PARENT_NAME.MY_NAME
        self.name = name
        # how the environment variable keys and values should be formatted
        self.env = env
        # how the constraint should be formatted
        self.constraint = constraint
        # children - empty list for leaf.
        # maps child name to child object
        self.children = {child.name: child for child in children}
        # possible values for this arg. non-None in leaf.
        self.data_range = data_range

    def get_arbitrary_assignment(self) -> Union["Assignment", Any]:
        # recursively build an arbitrary assignment
        # if we don't have a data range, build recursively from children
        if self.data_range is None:
            return Assignment({name: child.get_arbitrary_assignment()
                               for name, child in self.children.items()})
        # otherwise, just return an arbitrary value
        else:
            return self.data_range.get_arbitrary()

    def lookup(self, path: List[str]) -> "TunableArg":
        # lookup a relative path for this arg
        if len(path) == 0:
            return self
        else:
            return self.children[path[0]].lookup(path[1:])

    def build_env(self, path: Optional[List[str]],
                  assignment: Union[Any, "Assignment"]) -> Dict[str, str]:
        """
        Build an environment
        :param path: relative path to get to this arg
        :param assignment: remaining assignment starting at this arg
        """
        tr = {}
        # root case - there's ambiguity because for a child one below the root
        # or at the root would both be the empty list. So, the path to the root
        # is represented by None and handled specially.
        if path is None:
            assert self.name == "GLOBAL"
            new_path = []
            tr.update({key.expand_dict({}): val.expand_dict({})
                      for key, val in self.env.items()})
        else:
            new_path = path + [self.name]
            # NOTE: normally, we'd use periods as a separator. However,
            # environment variables don't support periods in their name, so use
            # underscores instead. This shouldn't cause an issue, because even
            # though we are creating ambiguity, we don't need to go in the
            # reverse direction.
            tr.update({
                # the key allows `name` and `path` specifiers
                key.expand_dict({
                    "name": self.name, "path": '_'.join(new_path)}):
                # the value allows any child value specifier
                val.expand_assignment(assignment)
                for key, val in self.env.items()
            })
        # build the remainder path recursively
        for child_name, child in self.children.items():
            tr.update(child.build_env(
                new_path, assignment.lookup([child_name])))
        return tr

    def is_valid_assignment(self, assignment: Union[Any, "Assignment"]) -> bool:
        """
        Check if an assignment recursively satisfies all constraints.
        """
        # if we have a constraint, expand, evaluate, and interpret as a bool
        if self.constraint is not None:
            expanded = self.constraint.expand_assignment(assignment)
            res = eval(expanded)
            assert isinstance(res, bool)
            if not res:
                return False
        # check all our child constraints recursively
        for child_name, child in self.children.items():
            if not child.is_valid_assignment(assignment.lookup([child_name])):
                return False
        return True


class Config:
    """
    Configuration class with user-specified execution options.
    """
    def __init__(self, max_iter: Optional[int], output: pathlib.Path,
                 preserve_files: List[pathlib.Path], search_class):
        """
        @param max_iter: maximum iterations, or None for indefinite
        @param output: output folder (e.g. arcc-codes/arcc-run-date)
        @param preserve_files: files to preserve
        @param search_class: search strategy class (e.g. MutationSearch)
        """
        self.max_iter = max_iter
        self.output = output
        self.preserve_files = preserve_files
        self.search_class = search_class

    def run_iter(self) -> Iterator[int]:
        """
        Iterate over trials. Either the user-specified number or infinite
        """
        if self.max_iter is None:
            return itertools.count(start=0, step=1)
        else:
            return range(self.max_iter)


def production(assignment_handler, argfile: Path, output: Path) -> TunableArg:
    """
    Production - convert parsed arguments to a root arg
    """

    output.mkdir(parents=True, exist_ok=True)

    # if our argfile is an R-Stream argfile, that means it doesn't exist, and
    # we need to create the file with tunable args. This will then flow through
    # the rest of the logic.
    if isinstance(argfile, RstreamArgFile):
        get_logger().info("no argfile specified; generating one")
        meta_file = produce_rstream_argfile(assignment_handler, output)

        argfile = ClassicArgFile(Path(meta_file))

    if isinstance(argfile, ClassicArgFile):
        assert argfile.file.exists(), \
            f"can't find classic argfile {argfile.file}"
        # parse it using the existing parser
        meta_data = src.meta_data_parser.parse(argfile.file)
        # conversion step
        data = convert_classic_meta_data(meta_data)
        arcc_new_file = output.joinpath("args.json")
        get_logger().info(f"placing converted file in {arcc_new_file}")
        with open(str(arcc_new_file), 'w') as f:
            json.dump(data, f, indent=2)
    elif isinstance(argfile, NewArgFile):
        # new format
        assert argfile.file.exists(), f"can't find args file {argfile.file}"
        shutil.copyfile(str(argfile.file),
                        str(output.joinpath("args.json")))
        # just read the data as-is from the file
        with open(argfile.file) as f:
            data = json.load(f)
    else:
        assert False, "should be unreachable: no argfile specified"
    global_env = data.get("global_env")
    if global_env is not None:
        global_env = {StringFormatter(key): StringFormatter(value)
                      for key, value in global_env.items()}
    else:
        global_env = {}
    # parse the tunable args from the argfile, and add a GLOBAL root arg that
    # contains all of them
    root = TunableArg("GLOBAL", global_env, None,
                      [parse_arg(arg, None) for arg in data["args"]], None)
    return root


def produce_rstream_argfile(assignment_handler, output) -> Path:
    """
    Takes in an assignment an output dir, returns path to metafile.

    This is a bit of a sore spot. Production and consumption are very
    different operations, but R-Stream kinda wants to do both with the same
    workflow. The idea is to "run without an assignment" to generate an argfile,
    but that really isn't possible with most commands. For example, how should
    `cmd --opt1 {opt1} --opt2 {opt2}` be run in production mode? One idea would
    be to set some magic flags that tells `cmd` to run in production mode, but
    then what should be substituted in that command line argument?

    Unfortunately, I think the most reasonable solution would be to require
    the user to generate this argfile, but that would break existing ARCC
    workflow. It might be worth moving this logic out to another command in
    bin/ that is an R-Stream specific way of generating this argfile to
    separate out the R-Stream specific logic, but leaving it here for now.
    """
    meta_file = output.joinpath("arcc.meta")
    production_env = {
        "ARCC_METADATA": str(meta_file),
        "ARCC_OPTIONUSEMODE": "default",
        "ARCC_MODE": "produce",
    }
    # actually perform production. this varies on the assignment handler,
    # since it's like we're running "without an assignment"
    assignment_handler.rstream_production(production_env)
    assert meta_file.exists(), "failed to make the argfile"
    return meta_file


def parse_arg(datum: Any, path: Optional[List[str]]) -> TunableArg:
    """
    Helper method that parses a tunable argument. See the README for the format.
    :param datum: Data constructed from primitive data structures.
    :param path: Path of args to this point. Used for error reporting.
    """
    # clone the path to avoid any weird shenanigans
    if path is not None:
        path = path.copy()

    # helper to get path as a string, for error reporting.
    def get_path() -> str:
        if path is None:
            return "GLOBAL"
        else:
            return '.'.join(path)
    name = datum.get("name")
    assert name is not None, "Missing name specifier in child of " + get_path()
    assert re.fullmatch(r"\w*", name), \
        f"name must only contain word-like characters: " \
        f"{name} in child of {get_path()}"
    if path is None:
        path = [name]
    else:
        path.append(name)

    # check if there are any unexpected keys, and warn if so. Do this first, as
    # it may be an indicator of an error to come.
    possible_keys = ["name", "env", "constraint", "range", "children"]
    for key in datum:
        if key not in possible_keys:
            get_logger().warning(f"unknown key {key} in {get_path()}")

    # environment variable formatter
    env = datum.get("env")
    if env is None:
        env = {}
    env = {StringFormatter(key): StringFormatter(val)
           for key, val in env.items()}

    # constraint formatter
    constraint = datum.get("constraint")
    if constraint is not None:
        constraint = StringFormatter(constraint)

    # range field should specify what type of range, with necessary data
    range_data = datum.get("range")
    if range_data is not None:
        range_data = parse_range_data(get_path, range_data)

    # recursively parse all our children
    children = [parse_arg(child, path) for child in datum.get("children", [])]

    return TunableArg(name, env, constraint, children, range_data)


def parse_range_data(get_path, range_data):
    assert len(range_data) == 1, \
        f"only one range specification allowed in {get_path()}"
    key, value = list(range_data.items())[0]
    # continuous float between lower and upper bound
    if key == "continuous":
        assert isinstance(value, list), "expected list"
        assert len(value) == 2, "expected length 2: lower and upper bound"
        lb = float(value[0])
        ub = float(value[1])
        range_data = ContinuousRange(lb, ub)
    # integer between lower and upper bound
    elif key == "integral":
        assert isinstance(value, list), "expected list"
        assert len(value) == 2, "expected length 2: lower and upper bound"
        lb = int(value[0])
        ub = int(value[1])
        range_data = IntegralRange(lb, ub)
    # discrete list of possibilities
    elif key == "discrete":
        assert isinstance(value, list), "expected list"
        range_data = DiscreteRange(list(map(str, value)))
    else:
        assert False, f"unknown range specifier: {key} in {get_path()}"
    return range_data


def convert_classic_meta_data(
        meta_data: src.meta_data_info.MetaDataInfo) -> Any:
    """
    Convert a meta data info parsed using the older code into a new meta data
    format that we know how to consume.

    Has a fixed two-layer structure - a list of high level name (usually
    corresponding to the function + tactic), then a list of variables inside
    that corresponding to the possible ways to tune.
    """
    args = []
    # returned in a singly linked list structure
    while meta_data is not None:
        prefix = meta_data.ID + "_"
        # the old parser appends the parent variable's name to the child
        # variables for some reason. strip that off to clean up a bit

        def strip_prefix(old: str) -> str:
            assert old.startswith(prefix)
            return old[len(prefix):]
        children = []
        # add all the children variables
        for var, options in zip(meta_data.var_list, meta_data.var_vals_list):
            children.append({
                "name": strip_prefix(var),
                "range": {
                    "discrete": options
                }
            })

        # the existing format wraps with dollar signs (`$VAR$`), and the new
        # format wraps in brackets (`{VAR}`), so use regex to replace
        def convert_old_to_new(old: str) -> str:
            # match any word-like string inside of dollar signs.
            # this will greedy (avoid matching `$VAR1$ $VAR2$`), as desired.
            return re.sub(r"\$(\w+)\$",
                          lambda m: f"{{{strip_prefix(m.group(1))}}}", old)
        arg = {
            "name": meta_data.ID,
            "env": {"ARCC_OPTION_{name}": convert_old_to_new(meta_data.option)},
            "children": children,
        }
        # avoid appending the constraint if it's statically true
        constraint = convert_old_to_new(meta_data.constraint)
        if constraint != "True":
            arg["constraint"] = constraint

        args.append(arg)
        meta_data = meta_data.next

    # Some magic flags that R-Stream uses
    env = {
        "ARCC_MODE": "consume",
        "ARCC_PERF": "rough",
    }
    return {"global_env": env, "args": args}
