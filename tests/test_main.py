import os
import tempfile


def test_rstream_production():
    """
    Test R-Stream production successfully generates a config file, then this
    file is successfully parsed.
    """
    import src.production
    # run rstream production on a matmult file
    arcc_tmp_dir = tempfile.mkdtemp(prefix='arcc-')
    source_file = os.path.join(os.environ["ARCC_HOME"],
                               "tests", "configs", "matmult.c")
    out_file = os.path.join(arcc_tmp_dir, "dummy.out")
    build_cmd = "rcc -S -I../include -I$RSTREAM_HOME/runtime/common/include " \
                f"-fopenmp {source_file} -o {out_file}"

    def temp_dir():
        return arcc_tmp_dir
    generated_file = src.production.rstream_production("", build_cmd, temp_dir)
    # make sure it exists, and has contents
    assert os.path.exists(generated_file)
    assert os.path.getsize(generated_file) > 0
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
                                    "tests", "configs", "matmult.meta")
    import src.meta_data_parser
    meta_data = src.meta_data_parser.parse(classic_metadata)
    res = src.production.convert_classic_meta_data(meta_data)

    expected_result = os.path.join(os.environ["ARCC_HOME"],
                                   "tests", "configs", "matmult.json")
    with open(expected_result) as f:
        expected_result = json.load(f)

    assert res == expected_result, \
        "failed to convert from classic to new format"


def test_sleep_consumption():
    import src.consumption
    import src.production
    import json
    import pathlib
    import src.search

    sleep_data = os.path.join(os.environ["ARCC_HOME"],
                              "tests", "configs", "sleep.json")
    with open(sleep_data) as f:
        sleep_data = json.load(f)

    # pretty explicitly create the root and config for consumption to run with
    root = src.production.TunableArg(
        "GLOBAL", {}, None, [src.production.parse_arg(arg, None)
                             for arg in sleep_data["args"]], None)
    arcc_tmp_dir = tempfile.mkdtemp(prefix='arcc-')
    config = src.production.Config(
        sleep_data["clean"], sleep_data["build"], sleep_data["run"],
        root, 1, pathlib.Path(arcc_tmp_dir).joinpath("dummy"),
        src.search.RandomSearch(root), [], True)

    # now, run consumption to get the optimal assignment
    optimal = src.consumption.consumption(config)
    # should only be one variable, which satisfies the constraints in the file
    assert optimal is not None, "no ideal assignment found"
    optimal = optimal.as_json()
    assert len(optimal) == 1, "should be one variable"
    key, val = list(optimal.items())[0]
    assert key == "sleep_duration", "invalid variable"
    assert .3 <= val <= .5, "value doesn't satisfy constraint"


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
