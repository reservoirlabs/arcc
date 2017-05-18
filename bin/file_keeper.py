# DOE-GPU rights
#
# Author      : Albert Hartono <hartonoa@reservoir.com>
# Description : Implementation for storing new and modified files in a subdirectory
#
# $Id$
#

import shutil, os, subprocess, time

#----------------------------------------------------------

class FileKeeper:
    '''The class definition used for storing new and modified files in a subdirectory'''
    
    def __init__(self, reporter, build_cmd, run_cmd, clean_cmd,):
        '''Instantiate a file keeper instance'''

        self.__reporter = reporter  # the reporting tool
        self.__build_cmd = build_cmd  # the command for building the optimized code
        self.__run_cmd = run_cmd  # the command for running the optimized code
        self.__clean_cmd = clean_cmd  # the command for cleaning previously generated codes
        
        # directory names in which the files are stored
        t = time.localtime()
        self.__new_code_dir = '%s-codes' % self.__reporter.moduleName()
        self.__this_run_dir = '%s-run-%s-%s-%s-%sh-%sm-%ss' % (self.__reporter.moduleName(), 
                                                               t.tm_year, t.tm_mon, t.tm_mday, 
                                                               t.tm_hour, t.tm_min, t.tm_sec)

    #------------------------------------------------------

    def __mkdirs(self, dir_path):
        '''
        Create the given directory along with its parent directories
        if they do not exist.
        '''
        
        # Check the given directory path
        assert dir_path.startswith(os.sep), 'given path must be absolute'

        # Create directories
        cur_dir = os.sep
        for d in dir_path.split(os.sep):
            if d == '': continue
            cur_dir = os.path.join(cur_dir, d)
            if not os.path.exists(cur_dir):
                os.mkdir(cur_dir)

    #------------------------------------------------------

    def __copy(self, src_file, dst_dir):
        '''
        Copy source file into destination directory. If the parent
        directories of the source file do not exist, then create the
        parent directories prior to copying the file.
        E.g., copy(a/b/c, /x/y) will create /x/y/a/b/c file
        '''
        
        # Check that the source path 
        assert os.path.isfile(src_file), 'source must be a file'
        assert not src_file.startswith(os.sep), 'source file must not use absolute path'
        assert not src_file.endswith(os.sep), 'source file must not end with a path separator'

        # Create parent directories
        cur_dir = dst_dir
        for d in src_file.split(os.sep)[:-1]:
            assert d != '', 'illegal directory name'
            cur_dir = os.path.join(cur_dir, d)
            if not os.path.exists(cur_dir):
                os.mkdir(cur_dir)
                
        # Copy the file
        shutil.copyfile(src_file, os.path.join(cur_dir, src_file.split(os.sep)[-1]))

    #------------------------------------------------------

    def __codeVariantDir(self, coord):
        '''
        Generate a directory name based on the given code variant
        (i.e., coordinate in the search space)
        '''
        
        d = 'coord'
        for i, c in enumerate(coord):
            d += '-%s' % c
        return d

    #------------------------------------------------------

    def __collectFiles(self, dir_path):
        '''
        Traverse the given directory and store the files in the
        directory and its subdirectory
        '''

        # Exclude collecting certain files/directories
        excluded_files = []
        excluded_files.extend(self.__reporter.logFiles()) # exclude ARCC log files
        excluded_dirs = []
        excluded_dirs.append(os.path.join(os.path.abspath(os.curdir), 
                                          self.__new_code_dir)) # exclude the directory used for storing

        # Filter out excluded files/directories that do not exist
        excluded_files = filter(lambda x: os.path.exists(x), excluded_files)
        excluded_dirs = filter(lambda x: os.path.exists(x), excluded_dirs)

        # Collect all files in the given path
        table = {}
        for top, dirs, files in os.walk(os.path.abspath(dir_path)):

            # Skip excluded directories
            top = top.rstrip(os.sep)
            is_in_excluded_dirs = False
            for ex_d in excluded_dirs:
                if top.startswith(ex_d):
                    is_in_excluded_dirs = True
                    break
            if is_in_excluded_dirs: 
                continue

            # Iterate over each files
            for f in files:
                f = os.path.join(top, f)

                # Skip excluded files
                is_excluded_file = False
                for ex_f in excluded_files:
                    if os.path.samefile(ex_f, f):
                        is_excluded_file = True
                        break
                if is_excluded_file:
                    continue

                # Remember the file
                table[f] = os.stat(f).st_mtime
        return table

    #------------------------------------------------------

    def __createREADME(self, dst_dir, coord, mdata):
        '''
        Create a README file to contain additional useful info (e.g.,
        build command, coordinate, used options)
        '''
        
        # README file name
        fname = os.path.join(dst_dir, 'README')
        
        # The content of the README file
        s = ''
        s += '# \n'
        s += '# %s \n' % fname
        s += '# \n'
        s += '# This directory contains copies of all files generated or modified \n'
        s += '# during the build of the code that corresponds to coordinate %s \n' % coord
        s += '# \n'
        s += '\n'
        s += 'Build command: %s \n' % self.__build_cmd
        s += 'Clean command: %s \n' % self.__clean_cmd
        s += 'Run command: %s \n' % self.__run_cmd
        s += '\n'
        s += 'Coordinate: %s \n' % coord
        s += 'Options: \n' 
        for ID, opt in mdata.getOptions(coord):
            s += '  %s = %s \n' % (ID, opt)
        s += '\n'

        # Write the README file
        try:
            f = open(fname, 'w')
            f.write(s)
            f.close()
        except Exception, e:
            raise self.__reporter.error('Failed to write README file "%s" ' % fname)

    #------------------------------------------------------

    def init(self):
        '''
        Traverse the current directory to remember the old files
        (before build)
        '''

        self.__reporter.logMessage('Remember directory structure before build')
        return self.__collectFiles(os.curdir)

    #------------------------------------------------------

    def store(self, old_files_table, coord, mdata):
        '''
        Traverse the current directory to find and store new files
        and modified files (after build)
        '''

        # Check the input parameters
        assert old_files_table != None, 'old files table cannot be None'
        assert coord != None, 'the coordinate must not be None'
        assert mdata != None, 'the meta data info must not be None'

        # Get a table containing the current files (after build)
        self.__reporter.logMessage('Traverse directory structure after build')
        cur_files_table = self.__collectFiles(os.curdir)

        # Create the directory to store the new files and the modified files
        this_code_dir = self.__codeVariantDir(coord)
        dst_dir = os.path.join(os.path.abspath(os.curdir), self.__new_code_dir, self.__this_run_dir, this_code_dir)
        self.__reporter.logMessage('Store new and modified files in "%s"' % dst_dir)
        self.__mkdirs(dst_dir)        

        # Store the new files and modified files
        prefix = os.path.abspath(os.curdir)
        for f, mtime in cur_files_table.items():
            if f not in old_files_table or mtime > old_files_table[f]:
                assert f.startswith(prefix), 'source file must be prefixed with the absolute current path'
                src = f[len(prefix):].lstrip(os.sep)
                dst = dst_dir
                self.__copy(src, dst)
        # Create a README file to contain additional useful info (e.g., build command, coordinate, used options)
        self.__createREADME(dst_dir, coord, mdata)
