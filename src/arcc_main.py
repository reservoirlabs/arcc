#!/usr/bin/env python3
import argparse
import itertools
import json
import logging
import os
import pathlib
import random
import re
import shutil
import subprocess as sp
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Any, Dict, Iterator, Tuple, Union

import src.meta_data_info
import src.meta_data_parser


def get_logger() -> logging.Logger:
    """
    Make sure the logger has been initialized first before calling this!
    """
    return logging.getLogger('arcc')


def initialize_logger(log_file: Optional[str], verbose: bool):
    """
    Build and return the `arcc` logger. Should only be called once! After
    called, can also be accessed with logging.getLogger('arcc').
    """
    # high level logger - for now, no others are used
    logger = logging.getLogger('arcc')
    # custom formatter
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    # DEBUG for verbose, otherwise just info
    log_level = logging.DEBUG if verbose else logging.INFO
    # log level must be set for both
    logger.setLevel(log_level)

    # helper to configure a handler
    def configure_handler(handler):
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # file handler
    if log_file is not None:
        configure_handler(logging.FileHandler(log_file))
    # stream (stdout) handler
    configure_handler(logging.StreamHandler())
    logger.debug("created logger")
    return logger


class Config:
    """
    Configuration class produced during "production", and consumed during
    "consumption".
    """
    def __init__(self, clean: str, build: str, run: str,
                 root: "TunableArg", max_iter: Optional[int],
                 output: pathlib.Path, search_strategy: "SearchStrategy",
                 preserve_files: List[pathlib.Path], fresh_dir: bool):
        # clean/build/run commands
        self.clean = clean
        self.build = build
        self.run = run
        # root of all arguments - uses a tree structure
        self.root = root
        # maximum trials, or none for indefinite
        self.max_iter = max_iter
        # output folder (e.g. arcc-codes/arcc-run-date)
        self.output = output
        # search strategy (random, mutation, etc) class
        self.search_strategy = search_strategy
        # files to preserve
        self.preserve_files = preserve_files
        # whether to switch to a fresh directory
        self.fresh_dir = fresh_dir

    def run_iter(self) -> Iterator[int]:
        """
        Iterate over trials. Either the user-specified number or infinite
        """
        if self.max_iter is None:
            return itertools.count(start=0, step=1)
        else:
            return range(self.max_iter)


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
    def __init__(self, name: str, env_key: Optional[StringFormatter],
                 set_env: Optional[StringFormatter],
                 constraint: Optional[StringFormatter],
                 children: List["TunableArg"],
                 data_range: Optional[DataRange]):
        # name given to this arg. used for setting command-line/environment.
        # NOTE: the "full name" will be ...GRANDPARENT_NAME.PARENT_NAME.MY_NAME
        self.name = name
        # how the environment variable key and value should be formatted
        self.env_key = env_key
        self.set_env = set_env
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
            assert self.set_env is None
            new_path = []
        else:
            new_path = path + [self.name]
            # NOTE: normally, we'd use periods as a separator. However,
            # environment variables don't support periods in their name, so use
            # underscores instead. This shouldn't cause an issue, because even
            # though we are creating ambiguity, we don't need to go in the
            # reverse direction. I'm not sure how this is parsed on the
            # R-Stream side of things.
            if self.set_env:
                # expand the user-given key to get our key
                # NOTE: this could be done in __init__ time
                if self.env_key:
                    env_key = self.env_key.expand_dict(
                        {"name": self.name, "path": '_'.join(new_path)})
                else:
                    env_key = '_'.join(new_path)
                tr[env_key] = self.set_env.expand_assignment(assignment)
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
            "env_key": "ARCC_OPTION_{name}",
            "set_env": convert_old_to_new(meta_data.option),
            "children": children,
        }
        # avoid appending the constraint if it's statically true
        constraint = convert_old_to_new(meta_data.constraint)
        if constraint != "True":
            arg["constraint"] = constraint

        args.append(arg)
        meta_data = meta_data.next
    return {"args": args}


def parse_args() -> Any:
    """
    Parse args and return the result.
    """
    parser = argparse.ArgumentParser(
        description='ARCC is an automated tuning tool designed to tune '
                    'compiler flags for optimal performance.')
    # config group - either the legacy or the new format
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument("--config-classic",
                        help="config with the previous format, as generated by "
                             "R-Stream. See README for more info.")
    config.add_argument("--config",
                        help="configuration file with tunable args, and "
                             "optionally build/run commands. See README "
                             "for more info.")
    # clean build and run command
    parser.add_argument('--clean',
                        help="clean command. ran before building.")
    parser.add_argument('--build',
                        help="build command. use {VAR_NAME} to indicate where "
                             "variables should be placed in the command.")
    parser.add_argument('--run',
                        help="run command. will be run after building and "
                             "timed. this is the main command to test")
    # logging stuff
    parser.add_argument('--verbose', help="enable verbose logging",
                        action='store_true')
    parser.add_argument('--log-file', help="log file location",
                        # with just --log-file, stored in log.txt
                        action='store_const', const="log.txt")
    # iterations
    parser.add_argument('--max-iter', '-n', type=int,
                        help="max number of iterations in the search "
                             "(infinite by default)")
    # output folder
    curr_time = str(datetime.now()).replace(' ', '_')
    default_output = os.path.join("arcc-codes", f"arcc-run-{curr_time}")
    parser.add_argument('--output', help="output location",
                        default=default_output)
    # files to copy to our own directory
    parser.add_argument('--preserve', help="files to preserve in our local dir",
                        nargs="+")
    # whether to switch to a fresh directory to run our command
    parser.add_argument('--fresh-dir',
                        help="switch to a fresh directory to run our command",
                        action='store_true')

    # exclusive group of search strategies
    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument("--dummy", action="store_true",
                                help="dummy strategy that executes once with "
                                     "arbitrary arguments, useful for testing")
    strategy_group.add_argument("--random", action="store_true",
                                help="simple strategy that randomly varies "
                                     "arguments and returns the best one")
    strategy_group.add_argument("--mutation", action="store_true",
                                help="strategy that repeatedly mutates the "
                                     "best known assignment until satisfied")

    return parser.parse_args()


def production(args: Any) -> Config:
    """
    Production - convert parsed arguments to a config that can be executed.
    """
    initialize_logger(args.log_file, args.verbose)
    if args.config_classic:
        # classic format
        config_file = pathlib.Path(args.config_classic)
        assert config_file.exists(), "can't find classic config file " \
                                     + args.config_classic
        # parse it using the existing parser
        meta_data = src.meta_data_parser.parse(config_file)
        # conversion step
        data = convert_classic_meta_data(meta_data)
        # TODO: remove this
        with open('arcc-new.json', 'w') as f:
            json.dump(data, f, indent=2)
    elif args.config:
        # new format
        config_file = pathlib.Path(args.config)
        assert config_file.exists(), "can't find config file " \
                                     + args.config_classic
        # just read the data as-is from the file
        with open(config_file) as f:
            data = json.load(f)
    else:
        # handled by argparse
        assert False, "unreachable: no config specified"
    # load each of the stages from the file/command line
    # NOTE: the "clean" stage is no longer necessary, as each test is ran in a
    # separate folder.
    stages = []
    for stage in ["clean", "build", "run"]:
        file_data = data.get(stage)
        cmd_data = getattr(args, stage)
        # by default, use the cmd line argument
        if cmd_data is not None and file_data is not None:
            stages.append(cmd_data)
            # warn since specified in both
            get_logger().warning(f"{stage} specified in both config file "
                                 f"and command line args. using value in "
                                 f"command file")
        elif cmd_data is not None:
            stages.append(cmd_data)
        elif file_data is not None:
            stages.append(file_data)
        else:
            assert False, f"{stage} not specified in either " \
                          f"config file or command line args"
    # parse the tunable args from the config file, and add a GLOBAL root arg
    # that contains all of them
    root = TunableArg("GLOBAL", None, None, None,
                      [parse_arg(arg, None) for arg in data["args"]], None)

    # choose and initialize the search strategy class with the tunable arguments
    if args.dummy:
        search_strategy = DummySearch
    elif args.random:
        search_strategy = RandomSearch
    elif args.mutation:
        search_strategy = MutationSearch
    else:
        get_logger().info("defaulting to random search strategy")
        search_strategy = RandomSearch
    if args.preserve is None:
        args.preserve = []
    args.preserve = [pathlib.Path(file) for file in args.preserve]
    search_strategy = search_strategy(root)
    return Config(stages[0], stages[1], stages[2], root,
                  args.max_iter, pathlib.Path(args.output), search_strategy,
                  args.preserve, args.fresh_dir)


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
    possible_keys = ["name", "env_key", "set_env",
                     "constraint", "range", "children"]
    for key in datum:
        if key not in possible_keys:
            get_logger().warning(f"unknown key {key} in {get_path()}")

    # environment variable formatter
    set_env = datum.get("set_env")
    if set_env is not None:
        set_env = StringFormatter(set_env)
    env_key = datum.get("env_key")
    if env_key is not None:
        assert set_env is not None, \
            f"env_key specified but not set_env in {get_path()}: " \
            f"can't set environment variable key without value"
        env_key = StringFormatter(env_key)

    # constraint formatter
    constraint = datum.get("constraint")
    if constraint is not None:
        constraint = StringFormatter(constraint)

    # range field should specify what type of range, with necessary data
    range_data = datum.get("range")
    if range_data is not None:
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

    # recursively parse all our children
    children = [parse_arg(child, path) for child in datum.get("children", [])]

    return TunableArg(name, env_key, set_env, constraint, children, range_data)


def consumption(config: Config):
    """
    The core logic. Consumes the config, optimizing arguments to maximize
    performance. Currently, only has one walk strategy -
    """
    get_logger().info(f"starting search, output in {config.output}")
    # make the high level output folder
    config.output.mkdir(parents=True)

    for run_id in config.run_iter():
        while True:
            # get the next assignment to try from the strategy
            assignment = config.search_strategy.get_assignment()
            # the strategy finished early, exit
            if assignment is None:
                break
            # valid assignment! continue on
            if config.root.is_valid_assignment(assignment):
                break
        # must have finished early
        if assignment is None:
            break

        # create the new directory
        fresh_dir = config.output.joinpath(f"run-{run_id}")
        fresh_dir.mkdir()

        # write extra info to log.txt as we run
        with open(str(fresh_dir.joinpath("log.txt")), "x") as out_file:
            # store the runtime and whether an error occurred
            runtime = None
            error_occurred = False
            get_logger().info(f"run {run_id} with assignment:\n{assignment}")
            print(f"assignment: {assignment}", file=out_file)
            for stage in ["clean", "build", "run"]:
                cmd = getattr(config, stage)
                if stage == "build":
                    # update the command and environment using the assignment
                    cmd = StringFormatter(cmd).expand_assignment(assignment)
                    new_env = config.root.build_env(None, assignment)
                else:
                    new_env = {}

                run_env = os.environ.copy()
                run_env.update(new_env)
                # run the stage command, and handle any errors appropriately
                get_logger().debug(f"running stage `{stage}` with cmd `{cmd}`")
                if len(new_env) > 0:
                    get_logger().debug(f"and env {new_env}")
                run_dir = fresh_dir if config.fresh_dir else pathlib.Path.cwd()
                print(f"{fresh_dir}: {cmd}, env={new_env}", file=out_file)
                start = time.time()
                proc = sp.run(cmd, shell=True, cwd=str(run_dir),
                              stdout=sp.PIPE, stderr=sp.STDOUT,
                              encoding='utf-8', env=run_env)
                end = time.time()
                if stage == "run":
                    runtime = end - start
                    print(f"executed in {runtime}", file=out_file)
                print(proc.stdout.strip(), file=out_file)
                # failing any stage isn't a hard error, so just log as such
                if proc.returncode != 0:
                    error_str = f"failed to run {stage} with code " \
                                f"{proc.returncode}"
                    print(error_str, file=out_file)
                    get_logger().debug(proc.stdout)
                    get_logger().error(error_str)
                    error_occurred = True
                    break
            # only record the runtime and notify results if no error occurred.
            if not error_occurred:
                get_logger().info(f"successfully ran in {runtime}s")
                config.search_strategy.notify_results(assignment, runtime)
            # preserve any files the user has requested
            for file in config.preserve_files:
                if not file.exists():
                    get_logger().warning(f"can't find file {file} to preserve")
                else:
                    shutil.copy(str(file), str(fresh_dir.joinpath(file.name)))

    # finished all runs! clean up and print results
    get_logger().info(f"finished executing in {config.output}")
    optimal = config.search_strategy.get_optimal()
    if optimal is None:
        get_logger().info("no optimal assignment found")
    else:
        assignment, runtime = optimal
        get_logger().info(f"optimal assignment is:\n"
                          f"{assignment}\n"
                          f"runtime: {runtime}s")


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


class SearchStrategy(ABC):
    """
    Abstract base class for a strategy that finds an optimal assignment.
    """

    @abstractmethod
    def __init__(self, _root: TunableArg):
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

    def __init__(self, root: TunableArg):
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
    def __init__(self, root: TunableArg):
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
    def __init__(self, root: TunableArg):
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


def main():
    args = parse_args()
    config = production(args)
    consumption(config)
