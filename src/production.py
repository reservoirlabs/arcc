import itertools
import json
import os
import pathlib
import re
import subprocess as sp
import tempfile
from typing import List, Optional, Any, Dict, Iterator, Union

import src.meta_data_info
import src.meta_data_parser
from src.arcc_main import get_logger
from src.assignment import Assignment
from src.data_range import DataRange, ContinuousRange, DiscreteRange, \
    IntegralRange
from src.search import MutationSearch, SearchStrategy, RandomSearch, DummySearch


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


def production(args: Any) -> Config:
    """
    Production - convert parsed arguments to a config that can be executed.
    """
    arcc_tmp_dir = None

    """
    Helper function to lazily create a temporary directory only when necessary.
    """
    def temp_dir():
        nonlocal arcc_tmp_dir
        if arcc_tmp_dir is None:
            arcc_tmp_dir = tempfile.mkdtemp(prefix='arcc-',)
            return arcc_tmp_dir
        else:
            return arcc_tmp_dir

    if args.config_classic is None and args.config is None:
        """
        This is a bit of a sore spot. Production and consumption are very 
        different operations, but R-Stream kinda wants to do both with the same
        workflow. The idea is to "run without an assignment" to generate a 
        configuration, but that really isn't possible with most commands. For 
        example, how should `cmd --opt1 {opt1} --opt2 {opt2}` be run in 
        production mode? One idea would be to set some magic flags that tells
        `cmd` to run in production mode, but then what should be substituted in
        that command line argument?
        
        Unfortunately, I think the most reasonable solution would be to require
        the user to generate this config, but that would break existing ARCC
        workflow. It might be worth moving this logic out to another command in
        bin/ that is an R-Stream specific way of generating this config to 
        separate out the R-Stream specific logic, but leaving it here for now.
        """
        get_logger().info("no config specified; generating one.\n"
                          "this is mostly meant as backward compatibility with "
                          "R-Stream, it is recommended to generate one before "
                          "calling arcc and to pass it in.")
        meta_file = None
        # no need tor run, since just bulding configuration
        for stage in ["clean", "build"]:
            cmd_data = getattr(args, stage)
            assert cmd_data is not None, f"must specify command for `{stage}`"
            run_env = os.environ.copy()
            if stage == "build":
                # add some magic environment variables
                meta_file = os.path.join(temp_dir(), "arcc.meta")
                run_env.update({
                    "ARCC_METADATA": meta_file,
                    "ARCC_OPTIONUSEMODE": "default",
                    "ARCC_MODE": "produce",
                })
            # run, handle errors
            proc = sp.run(cmd_data, shell=True, env=run_env,
                          stdout=sp.PIPE, stderr=sp.STDOUT,
                          encoding='utf-8')
            if proc.returncode != 0:
                get_logger().info(f"stage `{stage}` failed with exit code "
                                  f"{proc.returncode}:\n{proc.stdout}")
                assert False, "failed to generate configuration"
            if stage == "build":
                if not os.path.exists(meta_file):
                    get_logger().info(f"build returned zero exit code, but "
                                      f"failed to generate {meta_file}:\n"
                                      f"{proc.stdout}")
                    assert False, "failed to generate configuration"
                get_logger().info(f"generated config file {meta_file}")
                # set the classic config, and flow through the rest of the logic
                args.config_classic = meta_file

    if args.config_classic:
        # classic format
        config_file = pathlib.Path(args.config_classic)
        assert config_file.exists(), "can't find classic config file " \
                                     + args.config_classic
        # parse it using the existing parser
        meta_data = src.meta_data_parser.parse(config_file)
        # conversion step
        data = convert_classic_meta_data(meta_data)
        arcc_new_file = os.path.join(temp_dir(), "arcc-new.json")
        get_logger().info(f"placing converted file in {arcc_new_file}")
        with open(arcc_new_file, 'w') as f:
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
    global_env = data.get("global_env")
    if global_env is not None:
        global_env = {StringFormatter(key): StringFormatter(value)
                      for key, value in global_env.items()}
    # parse the tunable args from the config file, and add a GLOBAL root arg
    # that contains all of them
    root = TunableArg("GLOBAL", global_env, None,
                      [parse_arg(arg, None) for arg in data["args"]], None)

    # choose and initialize the search strategy class with the tunable arguments
    if args.dummy:
        search_strategy = DummySearch
    elif args.random:
        search_strategy = RandomSearch
    elif args.mutation:
        search_strategy = MutationSearch
    else:
        get_logger().info("defaulting to mutation search strategy")
        search_strategy = MutationSearch
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

    return TunableArg(name, env, constraint, children, range_data)


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
