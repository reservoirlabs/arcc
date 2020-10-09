#!/usr/bin/env python3
import os
import shutil
import subprocess as sp
rqa_cmd = shutil.which('run_test')
if rqa_cmd is None:
    rqa_cmd = os.path.join(os.environ["RSTREAM_HOME"],
                           'rqa', 'bin', 'run_suite')
    assert os.path.isdir(rqa_cmd)

test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')
sp.run([rqa_cmd, 'arcc', '--tag', 'arcc', test_file, '--rqa'])
