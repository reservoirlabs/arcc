import json
import os
import subprocess as sp
import tempfile
from pathlib import Path


def test_rstream_production():
    """
    Test R-Stream production successfully generates an argfile, then this
    file is successfully parsed.
    """
    import src.production
    # run rstream production on a matmult file
    arcc_tmp_dir = tempfile.mkdtemp(prefix='arcc-')
    source_file = os.path.join(os.environ["ARCC_HOME"],
                               "tests", "testfiles", "matmult.c")
    out_file = os.path.join(arcc_tmp_dir, "dummy.out")
    build_cmd = "rcc -S -I../include -I$RSTREAM_HOME/runtime/common/include " \
                f"-fopenmp {source_file} -o {out_file}"

    import src.consumption
    generated_file = src.production.produce_rstream_argfile(
        src.consumption.DefaultHandler(False, "", build_cmd, ""),
        Path(arcc_tmp_dir))
    # make sure it exists, and has contents
    assert generated_file.exists()
    assert os.path.getsize(str(generated_file)) > 0
    # make sure it parses
    import src.meta_data_parser
    src.meta_data_parser.parse(generated_file)
    # NOTE: ideally, this would assert some properties on the generated file
    # (e.g., known tunable variables, etc), but don't think that's as useful,
    # and worried about non-determinism and subtle changes in R-Stream breaking
    # the test.


def test_conversion():
    """
    Attempt to convert a "known" legacy format file into a new file type.
    """
    import json
    import src.production

    classic_metadata = os.path.join(os.environ["ARCC_HOME"],
                                    "tests", "testfiles", "matmult.meta")
    import src.meta_data_parser
    meta_data = src.meta_data_parser.parse(classic_metadata)
    res = src.production.convert_classic_meta_data(meta_data)

    expected_result = os.path.join(os.environ["ARCC_HOME"],
                                   "tests", "testfiles", "matmult.json")
    with open(expected_result) as f:
        expected_result = json.load(f)

    assert res == expected_result, \
        "failed to convert from classic to new format"


def test_sleep_consumption():
    """
    Basic test of consumption of a manually-created sleep argfile and args
    """
    import src.consumption
    import src.production
    import src.search

    sleep_data = os.path.join(os.environ["ARCC_HOME"],
                              "tests", "testfiles", "sleep.json")
    with open(sleep_data) as f:
        sleep_data = json.load(f)

    # pretty explicitly create the root and config for consumption to run with
    root = src.production.TunableArg(
        "GLOBAL", {}, None, [src.production.parse_arg(arg, None)
                             for arg in sleep_data["args"]], None)
    arcc_tmp_dir = tempfile.mkdtemp(prefix='arcc-')
    config = src.production.Config(
        1, Path(arcc_tmp_dir).joinpath("dummy"), [],
        src.search.RandomSearch)

    assignment_handler = src.consumption.DefaultHandler(
        False, "rm -f sleep.sh", "echo sleep {sleep_duration} > sleep.sh",
        "bash ./sleep.sh")
    # now, run consumption to get the optimal assignment
    optimal, opt_dir = \
        src.consumption.consumption(config, root, assignment_handler)
    # should only be one variable, which satisfies the constraints in the file
    assert optimal is not None, "no ideal assignment found"
    optimal = optimal.as_json()
    assert len(optimal) == 1, "should be one variable"
    key, val = list(optimal.items())[0]
    assert key == "sleep_duration", "invalid variable"
    assert .3 <= val <= .5, "value doesn't satisfy constraint"


def test_programmatic_sleep():
    """
    Test the programmatic frontend for the sleep example
    """
    # each {stage}_ran makes sure it is actually run, and is ran only once
    clean_ran = False
    sleep_file = os.path.join(tempfile.mkdtemp(prefix='arcc-'), "sleep.sh")

    def clean():
        nonlocal clean_ran
        assert not clean_ran
        clean_ran = True
        # remove the sleep file
        sp.run(f"rm -f {sleep_file}", shell=True, check=True)

    build_ran = False

    def build():
        nonlocal build_ran
        assert not build_ran
        build_ran = True
        sleep_dur = float(os.environ["sleep_duration"])
        # generate a dummy file that sleeps for a short duration
        sp.run(f"echo sleep {sleep_dur} > {sleep_file}", shell=True, check=True)

    run_ran = False

    def run():
        nonlocal run_ran
        assert not run_ran
        run_ran = True
        # run the sleep file
        sp.run(f"bash {sleep_file}", shell=True, check=True)

    import src.arcc_main
    sleep_data = Path(os.path.join(os.environ["ARCC_HOME"],
                                   "tests", "testfiles", "sleep.json"))
    # call the programmatic frontend, executing one time
    # this is a little high-level than might be ideal, but should be thorough
    optimal, opt_dir = \
        src.arcc_main.programmatic_main(
            clean, build, run, max_iter=1, argfile=sleep_data,
            preserve=[Path("sleep.sh")], enable_logging=False)
    # all stages should have run
    assert clean_ran and build_ran and run_ran
    # should only be one variable, which satisfies the constraints in the file
    assert optimal is not None, "no ideal assignment found"
    optimal = optimal.as_json()
    assert len(optimal) == 1, "should be one variable"
    key, val = list(optimal.items())[0]
    assert key == "sleep_duration", "invalid variable"
    assert .3 <= val <= .5, "value doesn't satisfy constraint"
    assert opt_dir
    assert opt_dir.exists()
    assert opt_dir.joinpath("sleep.sh").exists(), \
        "sleep.sh should have been preserved"


def test_matmult():
    """
    Test arcc by running on R-Stream's matmult sample
    @return:
    """
    openmp_dir = Path(os.environ["RSTREAM_HOME"])\
        .joinpath("benchmarks", "micro_kernels", "openmp")
    assert openmp_dir.exists()
    # this is how you'd "actually use" arcc on a real example
    proc = sp.run([
        os.path.join(os.environ["ARCC_HOME"], "bin", "arcc"),
        "--clean", "make clean",
        "--build", "make matmult",
        "--run", "./matmult",
        "--preserve", "matmult.gen.c", "matmult",
        "--max-iter", "5"],
        encoding="utf-8", stdout=sp.PIPE, stderr=sp.STDOUT,
        cwd=str(openmp_dir))
    if proc.returncode != 0:
        print(proc.stdout)
        assert False


def main():
    """
    At some point, I'd like to switch this over to unittest, but worried about
    multithreading or other issues. That's meant more for unit tests, while
    because of how arcc depends so heavily on a client, integration tests are
    easier without extensive mocking.
    """
    import argparse
    import src.arcc_main

    parser = argparse.ArgumentParser(
        description='Testing script for ARCC')
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        src.arcc_main.initialize_logger(None, True)
    test_rstream_production()
    test_conversion()
    test_sleep_consumption()
    test_programmatic_sleep()
    # disabled by default, since relies on the micro kernels, is slow, and
    # disrupts the existing code in the directory
    # test_matmult()
