#!/usr/bin/env python3
import argparse
import itertools
import json
import logging
import os
import pathlib
import random
import subprocess as sp
import time
from abc import ABC, abstractmethod
from datetime import datetime
import sys
from typing import List, Optional, Any, Dict, Iterator, Tuple
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def get_logger() -> logging.Logger:
    """
    Make sure the logger has been initialized first before calling this!
    """
    return logging.getLogger('arcc')


def get_arcc_home() -> pathlib.Path:
    """
    Helper to locate the arcc home directory. Uses environment variable if it
    exists, otherwise recursively searches for it.
    """
    # check env variable
    tfrcc_home = os.environ.get('ARCC_HOME')
    if tfrcc_home is not None:
        return pathlib.Path(tfrcc_home)
    else:
        # recursively walk back
        curr = pathlib.Path(__file__).parent.absolute()
        while curr.name != 'arcc':
            curr = curr.parent
        return curr


class Config:
    """
    Configuration class produced during "production", and consumed during
    "consumption".
    """
    def __init__(self, build: str, run: str, clean: str,
                 tunable_args: List["TunableArg"], max_trials: Optional[int],
                 output: pathlib.Path, search_strategy: "SearchStrategy"):
        # build command
        self.build = build
        # run command
        self.run = run
        # clean command
        self.clean = clean
        # tunable arguments
        self.tunable_args = tunable_args
        # maximum trials, or none for indefinite
        self.max_trials = max_trials
        # output folder (e.g. arcc-codes/arcc-run-date)
        self.output = output
        # search strategy (random, mutation, etc)
        self.search_strategy = search_strategy

    def run_iter(self) -> Iterator[int]:
        """
        Iterate over trials. Either the user-specified number or infinite
        """
        if self.max_trials is None:
            return itertools.count(start=0, step=1)
        else:
            return range(self.max_trials)


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
    def mutate_value(self, value: Any) -> Any:
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

    def mutate_value(self, value: Any) -> Any:
        """
        To mutate, get a new limited range centered at the current location,
        and get a uniform value up or down from that.
        """
        mutation_range = (self.ub - self.lb) / 8
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

    def mutate_value(self, value: Any) -> Any:
        """
        To mutate, get a new limited range centered at the current location,
        and get a uniform value up or down from that.
        """
        # +7 to round up on mutation range
        mutation_range = (self.ub - self.lb + 7) // 8
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

    def mutate_value(self, value: Any) -> Any:
        """
        To mutate, get our current index and mutate it.
        """
        index = self.options.index(value)
        new_index = IntegralRange(0, len(self.options) - 1).mutate_value(index)
        return self.options[new_index]

    def contains(self, val: Any) -> bool:
        assert isinstance(val, str)
        return val in self.options


class TunableArg:
    """
    Class representing an argument that can be tuned
    """
    def __init__(self, name: str, formatter: "FormatValue",
                 data_range: DataRange):
        # name given to this arg
        self.name = name
        # string to format (should contain {key} or {val} where desired)
        self.formatter = formatter
        # possible values for this arg
        self.data_range = data_range

    def __str__(self) -> str:
        return f"{self.name} ({self.data_range})"


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


class FormatValue(ABC):
    """
    Abstract base class for
    """
    pass


class Environment(FormatValue):
    def __init__(self, variable):
        self.variable = variable


class Cmdline(FormatValue):
    def __init__(self, format_str):
        self.format_str = format_str

    def format_with(self, val: Any) -> str:
        return self.format_str.format(val=val)


def get_config() -> Config:
    here = os.getcwd()
    parser = argparse.ArgumentParser(
        description='ARCC is an automated tuning tool designed to tune '
                    'compiler flags for optimal performance.')
    parser.add_argument("--config", required=True,
                        help="configuration file with tunable args, and "
                             "optionally build/run/clean commands.")
    parser.add_argument('--build',
                        help="build command. `{}` is used to denote optional "
                             "arguments, otherwise will be placed at the end.")
    parser.add_argument('--run',
                        help="run command. will be run after building and "
                             "timed.")
    parser.add_argument('--clean', help="clean command. ran after running")
    parser.add_argument('--verbose', help="enable verbose logging",
                        action='store_true')
    parser.add_argument('--log-file', help="log file location",
                        action='store_const', const="log.txt")
    parser.add_argument('--max-trials', '-n', type=int, help="number of trials")
    parser.add_argument('--output', help="output location",
                        default=pathlib.Path(here)
                        .joinpath("arcc-codes", f"arcc-run-{datetime.now()}"))

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

    args = parser.parse_args()

    initialize_logger(args.log_file, args.verbose)
    # try to load the config as a file directly
    if os.path.isfile(args.config):
        config_file = pathlib.Path(args.config)
    else:
        # if it isn't one, search for it in `arcc/configs/{name}.json`
        config_file = get_arcc_home().joinpath("configs", args.config + ".json")
        assert config_file.exists(), f"can't locate config {args.config}"
    with open(str(config_file)) as f:
        get_logger().info(f"loading config from {f.name}")
        data = json.load(f)

    # load each of the stages from the file/command line
    stages = []
    for stage in ["build", "run", "clean"]:
        file_data = data.get(stage)
        cmd_data = getattr(args, stage)
        if file_data is not None:
            stages.append(file_data)
            # if it exists in the file, warn if it's also specified in args
            if cmd_data is not None:
                get_logger().warning(f"{stage} specified in both config file "
                                     f"and command line args. using value in "
                                     f"config file")
        else:
            assert cmd_data is not None, f"{stage} not specified in either " \
                                         f"config file or command line args"
            stages.append(cmd_data)

    # parse the tunable args from the config file
    tunable_args = production(data["args"])
    get_logger().debug(f"tunable args: {list(map(str, tunable_args))}")
    if args.dummy:
        search_strategy = DummySearch
    elif args.random:
        search_strategy = RandomSearch
    elif args.mutation:
        search_strategy = MutationSearch
    else:
        get_logger().info("defaulting to random search strategy")
        search_strategy = RandomSearch
    # initialize the chosen search strategy class with the tunable arguments
    search_strategy = search_strategy(tunable_args)
    return Config(stages[0], stages[1], stages[2], tunable_args,
                  args.max_trials, args.output, search_strategy)


def production(data: List[Dict[str, Any]]) -> List[TunableArg]:
    """
    Generate the meta data for use in auto-tuning. This describes what arguments
    can be tuned and how. This could also be done by querying R-Stream (or
    whatever is being auto-tuned) dynamically; however, this isn't currently
    supported.
    """
    tr = []
    for datum in data:
        # range field should specify what type of range, with necessary data
        range_data = datum["range"]
        assert len(range_data) == 1, "only one range specification allowed"
        key, value = list(range_data.items())[0]

        # consider moving these raw constructors into the class?
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
            assert False, f"unknown key: {key}"

        # name, formatter
        name = datum['name']
        assert isinstance(name, str)
        format_data = datum['format']
        assert isinstance(format_data, dict)
        assert len(format_data) == 1
        format_type, format_data = list(format_data.items())[0]
        assert isinstance(format_type, str)
        if format_type == 'cmd_line':
            assert isinstance(format_data, str)
            formatter = Cmdline(format_data)
        elif format_type == 'environment':
            assert isinstance(format_data, str)
            formatter = Environment(format_data)
        else:
            assert False, "unknown format type: " + format_type
        tr.append(TunableArg(name, formatter, range_data))
    return tr


def consumption(config: Config):
    """
    The core logic. Consumes the config, optimizing arguments to maximize
    performance. Currently, only has one walk strategy -
    """
    # make the high level output folder
    config.output.mkdir(parents=True)

    for run_id in config.run_iter():
        # get the next assignment to try from the strategy
        assignment = config.search_strategy.get_assignment()
        # the strategy finished early, exit
        if assignment is None:
            break
        # create the new directory to run the test in
        out_dir = config.output.joinpath(f"run-{run_id}")
        out_dir.mkdir()

        # write extra info to log.txt as we run
        with open(str(out_dir.joinpath("log.txt")), "x") as out_file:
            # store the runtime and whether an error occurred
            runtime = None
            error_occurred = False
            get_logger().info(f"run {run_id} with assignment "
                              f"{assignment}")
            print(str(assignment), file=out_file)
            for stage in ["build", "run", "clean"]:
                cmd = getattr(config, stage)
                run_env = os.environ.copy()
                if stage == "build":
                    # in build mode, update our command/environment based on the
                    # assignment. environment variables should be updated and
                    # the build command should be run
                    extra_args = ""
                    for formatter, val in assignment.format_value_pairs():
                        # command line formatter, add to command line
                        if isinstance(formatter, Cmdline):
                            extra_args += formatter.format_with(val) + " "
                        # environment formatter, update command line
                        elif isinstance(formatter, Environment):
                            run_env[formatter.variable] = val
                        else:
                            assert False, f"unexpected formatter: {formatter}"

                    extra_args = extra_args.strip()
                    if "{}" in cmd:
                        # if the command has {}, place them there
                        cmd = cmd.replace("{}", extra_args)
                    else:
                        # otherwise, just place them at the end
                        cmd = f"{cmd} {extra_args}"
                # run the stage command, and handle any errors appropriately
                get_logger().debug(f"running {stage} stage with {cmd}")
                print(f"{out_dir}: {cmd}", file=out_file)
                start = time.time()
                proc = sp.run(cmd, shell=True, cwd=str(out_dir),
                              stdout=sp.PIPE, stderr=sp.STDOUT,
                              encoding='utf-8', env=run_env)
                end = time.time()
                if stage == "run":
                    runtime = end - start
                    print(f"executed in {runtime}", file=out_file)
                print(proc.stdout, file=out_file)
                # failing any stage isn't a hard error, so just log as such
                if proc.returncode != 0:
                    error_str = f"failed to run {stage} with code " \
                                f"{proc.returncode}"
                    print(error_str, file=out_file)
                    get_logger().error(error_str)
                    error_occurred = True
                    break
            # only record the runtime and notify results if no error occurred.
            # technically, we're still probably safe to do it if cleanup fails,
            # but that's unlikely
            if not error_occurred:
                get_logger().info(f"successfully ran in {runtime}s")
                config.search_strategy.notify_results(assignment, runtime)

    # finished all runs! clean up and print results
    get_logger().info(f"finished executing")
    optimal = config.search_strategy.get_optimal()
    if optimal is None:
        get_logger().info("no optimal assignment found")
    else:
        assignment, runtime = optimal
        get_logger().info(f"optimal assignment is {assignment} with runtime "
                          f"{runtime}s")


class Assignment:
    """
    Represents an assignment of values to tunable arguments. Helpful to avoid
    hashing oddities and easy pretty printing.
    """
    def __init__(self, items: Dict[TunableArg, Any]):
        """
        Python doesn't have a `frozendict`, so model with a frozenset instead.
        """
        self.items = frozenset(items.items())

    def dict(self) -> Dict[TunableArg, Any]:
        """
        Build and return a dict the user can use.
        """
        return {key: val for key, val in self.items}

    def format_value_pairs(self) -> List[Tuple[FormatValue, Any]]:
        """
        Return the pairs of format values and values
        """
        return [(key.formatter, val) for key, val in self.items]

    def __str__(self) -> str:
        """
        Print the assignments in a human-readable format.
        """
        return str([f"{key.name} -> {val}" for key, val in self.items])

    # forward hash/eq to the frozenset
    def __hash__(self):
        return hash(self.items)

    def __eq__(self, other):
        return self.items == other.items


class SearchStrategy(ABC):
    """
    Abstract base class for a strategy that finds an optimal assignment.
    """

    @abstractmethod
    def __init__(self, tunable_args: List[TunableArg]):
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

    def __init__(self, tunable_args: List[TunableArg]):
        super().__init__(tunable_args)
        self.tunable_args = tunable_args
        self.has_returned = False

    def get_assignment(self) -> Optional[Assignment]:
        """
        Return an arbitrary assignment for each input
        """
        if self.has_returned:
            return None
        else:
            self.has_returned = True
            return Assignment({arg: arg.data_range.get_arbitrary()
                               for arg in self.tunable_args})

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
    def __init__(self, tunable_args: List[TunableArg]):
        super().__init__(tunable_args)
        self.tunable_args = tunable_args
        self.best_observed = None
        self.best_time = None

    def get_assignment(self) -> Optional[Assignment]:
        """
        Return an arbitrary assignment for each input
        """
        return Assignment({arg: arg.data_range.get_arbitrary()
                           for arg in self.tunable_args})

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
    def __init__(self, tunable_args: List[TunableArg]):
        super().__init__(tunable_args)
        # current number of failed mutations/restarts
        self.failed_mutations = 0
        self.restarts = 0
        # time and assignment of the current "head"
        self.curr_time = None
        self.curr_assignment: Optional[Assignment] = None
        # all tunable args
        self.tunable_args = tunable_args
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
            if self.restarts >= 4:
                return None
            return Assignment({arg: arg.data_range.get_arbitrary()
                               for arg in self.tunable_args})
        else:
            mutated = list(self.curr_assignment.dict().items())
            # only if there's anything to mutate
            if len(mutated) > 0:
                # choose a random argument to mutate, and mutate it
                to_mutate = random.randint(0, len(mutated) - 1)
                key, old_val = mutated[to_mutate]
                new_val = key.data_range.mutate_value(old_val)
                mutated[to_mutate] = key, new_val
            # build and return the mutated assignment
            mutated = Assignment({key: val for key, val in mutated})
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
            if self.failed_mutations == 5:
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
    config = get_config()
    consumption(config)


if __name__ == "__main__":
    main()
