import os
import pathlib
import shutil
import subprocess as sp
import time
from typing import Optional

from src.arcc_main import get_logger
from src.production import Config, StringFormatter

from src.assignment import Assignment


def consumption(config: Config) -> Optional["Assignment"]:
    """
    The core logic. Consumes the config, optimizing arguments to maximize
    performance.
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
        return None
    else:
        assignment, runtime = optimal
        get_logger().info(f"optimal assignment is:\n"
                          f"{assignment}\n"
                          f"runtime: {runtime}s")

        # print out what the user should rerun the full process
        build = StringFormatter(config.build).expand_assignment(assignment)
        env = config.root.build_env(None, assignment)
        if len(env) > 0:
            build = "env " + ' '.join(f'{key}={val}'
                                      for key, val in env.items()) \
                        + " " + build
        get_logger().info(f"rerun with: `{config.clean} && "
                          f"{build} && "
                          f"{config.run}`")
        return assignment
