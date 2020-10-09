This is an implementation of the ARCC Protocol, a flexible auto-tuning framework, in Python. It's used to tune compiler options for R-Stream to make the fastest code.

# How it works

The user specifies a clean, build, and run command, as well as arguments to tune with their possible values. The auto-tuner wil try various possible assignments, and use a search strategy to find an optimal assignment and report to the user.

# Running + Testing

Run `bin/arcc --help` for information about how to run arcc. For an example, run `bin/arcc --argfile tests/testfiles/sleep.json` to show arcc "tuning" a dummy compiler that simply generates a script that runs for a tunable amount of time. The `bin` dir can be safely added to your path to make running `arcc` from anywhere easier. 

To run the tests, run `tests/test`. If there is no output, then all the tests passed. 

Use the `--verbose` flag for either command for more detailed run information.

For a more advanced and R-Stream specific test, run `arcc --clean "make clean" --build "make matmult" --run "./matmult" --preserve matmult.gen.c matmult` in `$RSTREAM_HOME/benchmarks/micro_kernels/openmp`.

# Format

There are two formats - one of them is a legacy format used currently by R-Stream, the other is a newer format that's preferred which will be described below. The arguments to be tuned are leaves of a tree, and the elements of that tree can be used to help organize the formatting and constraints for those arguments. See `tests/testfiles` for some examples. 

The format is: 
```
{ 
    "global_env": Dict[str, str], (optional)
    "args": List[Arg],
}
```
The "global_env" contains a dictionary of environment variables to set before running any test (e.g., if the client needs to be notified that it's being called from `arcc`). `args` contains the actual list of arguments to tune.

The format of each `Arg` is:
```
{
    "name": str, 
    "env": Dict[str, str], (optional)
    "constraint": str, (optional)
    "range": DataRange, (optinal)
    "children": List[Arg], (optional)
}
``` 

The `name` is the name of this argument, and the "path" to an argument is defined as a period concatenated list of names of args to transverse. `env` is a dictionary of environment variables to set. This is useful when the client receives arguments through environment variables. The constraint is a string constraint that should be satisfied (interpreted as python). "range" is the possible values for this variable, described below. "children" are any children of this node.

All strings (except name) above may contain the following expressions, which will be substituted before evaluated:

- `{name}` - the short name of this node (e.g. `load_factor`)  
- `{path}` - the full path of this node (e.g., `hash_config.load_factor`)
- `{SELF}` - the value of this node for the assignment (e.g., `0.7`)
- `{child1.child2}` - the value of `child2` of `child1` of this node
 
 The possible `DataRange` formats are:
 - `{ "discrete": List[str] }` explicit list of possible values
 - `{ "continuous": [float, float] }` range of floating values 
 - `{ "integral": [int, int] }` range of integer values (inclusive)
 
# Programmatic frontend
There is also a frontend that can be called more easily and directly from python code, that uses callbacks rather than commands, allowing clean/build/run to be ran in the same process as arcc was called from. See `bin/arcc_main:programmatic_main` for more info.
