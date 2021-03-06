import os
import shutil
import subprocess as sp
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Dict, TextIO, Callable

from src.arcc_main import get_logger
from src.assignment import Assignment
from src.production import Config, StringFormatter, TunableArg
from contextlib import contextmanager


@contextmanager
def add_to_env(kv_pairs: dict):
    orig_dict = os.environ.copy()
    try:
        os.environ.update(kv_pairs)
        yield
    finally:
        os.environ = orig_dict


class AssignmentHandler(ABC):
    """
    An assignment handler is meant to abstract over the actual methods used to
    evaluate an assignment in a context. For example, the programmatic frontend
    will want to handle assignments by calling user-specified python functions,
    while the cmd assignment handler will call user-specified functions.
    """
    @abstractmethod
    def evaluate(self, run_id: int, run_dir: Path, assignment: "Assignment",
                 new_env_pairs: Dict[str, str], log_file: TextIO) \
            -> Tuple[bool, float]:
        """
        The main purpose of an assignment handler: call the user-specified data
        to execute the three stages (clean, build, run), and report the status.
        @param run_id: current run
        @param run_dir: output dir
        @param assignment: current assignment
        @param new_env_pairs: environment keys for the assignment
        @param log_file: log file to dump stuff to
        @return: whether an error occurred and the execution time.
        """
        pass

    @abstractmethod
    def epilogue(self, assignment: Assignment, new_env_pairs: Dict[str, str]):
        """
        Prints the epilogue for the user, e.g. how to rerun the assignment.
        @param assignment: Optimal assignment
        @param new_env_pairs: environment keys for the assignment
        """
        pass

    @abstractmethod
    def rstream_production(self, production_env: Dict[str, str]):
        """
        Evaluate in r-stream production mode
        @param production_env: environment variable flags to set
        """
        pass


class DefaultHandler(AssignmentHandler):
    """
    The default handler works using the command line. Runs clean, build, then
    times the run.

    This is a kind of further specification of the python func handler. That is,
    rather than specifying commands themselves, we could pass in callbacks that
    simply wrap around a subprocess run call. This would remove some of the
    duplicated logic, but made it needlessly complex when I went down that path.
    """
    def __init__(self, fresh_dir: bool, clean: str, build: str, run: str):
        """
        @param fresh_dir: whether to run in a fresh dir
        @param clean: clean cmd
        @param build: build cmd
        @param run: run cmd
        """
        self.fresh_dir = fresh_dir
        self.clean = clean
        self.build = build
        self.run = run

    def evaluate(self, run_id, run_dir, assignment, new_pairs, log_file) \
            -> Tuple[bool, float]:
        """
        See super's comments for more description/typing. Runs each stage,
        logging and writing any relevant output to the log file. Fails if any
        command fails.
        """
        # store the runtime and whether an error occurred
        runtime = None
        error_occurred = False
        for stage in ["clean", "build", "run"]:
            cmd = getattr(self, stage)
            if stage == "build":
                # update the command and environment using the assignment
                cmd = StringFormatter(cmd).expand_assignment(assignment)
                new_env = new_pairs
            else:
                new_env = {}

            run_env = os.environ.copy()
            run_env.update(new_env)
            # run the stage command, and handle any errors appropriately
            get_logger().debug(f"running stage `{stage}` with cmd `{cmd}`")
            if len(new_env) > 0:
                get_logger().debug(f"and env {new_env}")
            run_dir = run_dir if self.fresh_dir else Path.cwd()
            print(f"{run_dir}: {cmd}, env={new_env}", file=log_file)
            start = time.time()
            # the main call
            proc = sp.run(cmd, shell=True, cwd=str(run_dir),
                          stdout=sp.PIPE, stderr=sp.STDOUT,
                          encoding='utf-8', env=run_env)
            end = time.time()
            if stage == "run":
                runtime = end - start
                print(f"executed in {runtime}", file=log_file)
            print(proc.stdout.strip(), file=log_file)
            # failing any stage isn't a hard error, so just log as such
            if proc.returncode != 0:
                error_str = f"failed to run {stage} with code " \
                            f"{proc.returncode}"
                print(error_str, file=log_file)
                get_logger().debug(proc.stdout)
                get_logger().error(error_str)
                error_occurred = True
                break
        return error_occurred, runtime

    def epilogue(self, assignment, new_env_pairs):
        """
        print out what the user should rerun the full process
        """
        build = StringFormatter(self.build).expand_assignment(assignment)
        # make sure to add an `env` command for each key value pair
        if len(new_env_pairs) > 0:
            build = "env " + ' '.join(f'{key}={val}'
                                      for key, val in new_env_pairs.items()) \
                        + " " + build
        get_logger().info(f"rerun with: `{self.clean} && "
                          f"{build} && "
                          f"{self.run}`")

    def rstream_production(self, production_env):
        """
        To do production, run just the clean and build commands, with some extra
        environment variables.
        """
        stages = {"clean": self.clean, "build": self.build}
        # no need tor run, since just building argfile
        for stage in ["clean", "build"]:
            cmd_data = stages[stage]
            assert cmd_data is not None, f"must specify command for `{stage}`"
            run_env = os.environ.copy()
            # update the environment
            if stage == "build":
                run_env.update(production_env)
            get_logger().debug("running " + cmd_data)
            # run, handle errors
            proc = sp.run(cmd_data, shell=True, env=run_env,
                          stdout=sp.PIPE, stderr=sp.STDOUT,
                          encoding='utf-8')
            if proc.returncode != 0:
                get_logger().info(f"stage `{stage}` failed with exit code "
                                  f"{proc.returncode}:\n{proc.stdout}")
                assert False, "failed to generate argfile"


class PythonFuncHandler(AssignmentHandler):
    """
    An alternate assignment handler used for the programmatic frontend. Rather
    than taking in user commands, it takes in user functions. This allows them
    to be run directly in the same process space.

    Currently, this only propagates assignment information to the user via
    environment variables, although this could be updated to also allow passing
    in the assignment directly
    """
    def __init__(self, clean: Callable[[], None],
                 build: Callable[[], Optional[bool]],
                 run: Callable[[], Optional[Tuple[bool, float]]]):
        """
        @param clean: clean callback
        @param build: build callback
        @param run: run callback that optionally returns the runtime to use
        instead (useful if the user wants to time the command themselves)
        """
        self.clean = clean
        self.build = build
        self.run = run

    def evaluate(self, run_id, run_dir, assignment, new_pairs, log_file) \
            -> Tuple[bool, float]:
        """
        See super's comments for typing and more description. Calls the function
        for each stage, ensuring the environment variables are updated for it.
        """
        self.clean()
        with add_to_env(new_pairs):
            res = self.build()
            # res could be None (user didn't report status), False (no error),
            # or True (error occurred)
            if res is not None:
                if res:
                    return True, 0.0
            start = time.time()
            res = self.run()
            end = time.time()
            if res is not None:
                error_occurred, self_time = res
                if error_occurred:
                    return True, 0.0
            else:
                error_occurred = False
                self_time = end - start
            return error_occurred, self_time

    def epilogue(self, assignment, new_pairs):
        """
        no epilogue, since no way for the user to "rerun" it themselves.
        """
        pass

    def rstream_production(self, production_env):
        """
        For production, simply call clean and build with environment set.
        @param production_env: env keys for production
        """
        self.clean()
        with add_to_env(production_env):
            res = self.build()
            if res is not None and res:
                assert res, "failed to build during production"


def consumption(config: Config, root: TunableArg,
                assignment_handler: AssignmentHandler) \
        -> (Optional[Tuple["Assignment", Path]]):
    """
    The core logic. Use the search strategy of the config to generate
    assignments and time them using the assignment handler.
    @param config: configuration of execution
    @param root: root of the tunable arg tree
    @param assignment_handler: how to evaluate assignments
    @return: optimal assignment and Path to its dir
    """
    # initialize search
    search_strategy = config.search_class(root)
    get_logger().info(f"starting search, output in {config.output}")
    # make the high level output folder
    config.output.mkdir(parents=True, exist_ok=True)
    # save the map from assignment to its corresponding run directory
    # at some point, this should probably be replaced by a more general system
    # to evaluate assignments, that also handles detecting duplicates,
    # identifying "adjacent" assignments to estimate evaluation, etc.
    assignment_map = {}
    for run_id in config.run_iter():
        while True:
            # get the next assignment to try from the strategy
            assignment = search_strategy.get_assignment()
            # the strategy finished early, exit
            if assignment is None:
                break
            # valid assignment, done
            if root.is_valid_assignment(assignment):
                break
            # invalid assignment - keep trying
        # no assignment, break
        if assignment is None:
            break
        # create the new directory
        fresh_dir = config.output.joinpath(f"run-{run_id}")
        fresh_dir.mkdir()

        new_env_pairs = root.build_env(None, assignment)
        with open(str(fresh_dir.joinpath("log.txt")), "x") as log_file:
            get_logger().info(f"run {run_id} with assignment:\n{assignment}")
            print(f"assignment: {assignment}", file=log_file)
            # forward the work of evaluating to the actual assignment handler
            error_occurred, runtime = \
                assignment_handler.evaluate(run_id, fresh_dir,
                                            assignment, new_env_pairs, log_file)
        # only record the runtime and notify results if no error occurred.
        if not error_occurred:
            get_logger().info(f"successfully ran in {runtime}s")

            search_strategy.notify_results(assignment, runtime)
            assignment_map[assignment] = fresh_dir
        # preserve any files the user has requested
        for file in config.preserve_files:
            if not file.exists():
                get_logger().warning(f"can't find file {file} to preserve")
            else:
                shutil.copy(str(file), str(fresh_dir.joinpath(file.name)))
        # must have finished early
        if assignment is None:
            break

    # finished all runs! clean up and print results
    get_logger().info(f"finished executing in {config.output}")
    optimal = search_strategy.get_optimal()
    if optimal is None:
        get_logger().info("no optimal assignment found")
        return None
    else:
        assignment, runtime = optimal
        get_logger().info(f"optimal assignment is:\n"
                          f"{assignment}\n"
                          f"with runtime: {runtime}s")
        new_env_pairs = root.build_env(None, assignment)
        # print the handler-specific epilogue
        assignment_handler.epilogue(assignment, new_env_pairs)
        opt_dir = assignment_map[assignment]
        get_logger().info(f"preserved files in {opt_dir}")
        return assignment, opt_dir
