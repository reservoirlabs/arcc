# DOE-GPU rights
#
# Author      : Albert Hartono <hartonoa@reservoir.com>
# Description : Class for building, running code and measuring its performance
#
# $Id$
#

import os, subprocess, time
from . import file_keeper
from functools import reduce

#----------------------------------------------------------

class CodeExecutor:
    '''The class definition used for building, running code and measuring its performance'''

    # ARCC's environment variable names
    __ARCC_OPTION = 'ARCC_OPTION'
    
    #------------------------------------------------------

    def __init__(self, reporter, build_cmd, run_cmd, clean_cmd, keep, perf_count, perf_fname):
        '''Instantiate a code executor instance'''

        self.__reporter = reporter  # the reporting tool
        self.__file_keeper = file_keeper.FileKeeper(reporter, build_cmd, run_cmd, clean_cmd)  # the file keeper

        self.__build_cmd = build_cmd  # the command for building the optimized code
        self.__run_cmd = run_cmd  # the command for running the optimized code
        self.__clean_cmd = clean_cmd  # the command for cleaning previously generated codes
        self.__keep = keep  # store new and modified files after each build in a subdirectory
        self.__perf_count = perf_count  # the performance counting method
        self.__perf_fname = perf_fname # the performance data filename

        if self.__perf_count not in ('rough', 'precise'):
            raise self.__reporter.error('Unrecognized performance counting method: ' % self.__perf_count)

    #------------------------------------------------------

    def isToMinimize(self):
        '''Return true if minimum performance number indicates best result. Otherwise, return false'''
        
        if self.__perf_count in ('rough', 'precise'):
            return True
        else:
            raise self.__reporter.error('Unrecognized performance counting method: ' % self.__perf_count)

    #------------------------------------------------------

    def perfMetric(self):
        '''Return the performance metric (e.g., seconds, Gflops)'''

        if self.__perf_count in ('rough', 'precise'):
            return 'seconds'
        else:
            raise self.__reporter.error('Unrecognized performance counting method: ' % self.__perf_count)

    #------------------------------------------------------

    def __measurePreciseTime(self):
        '''
        Read performance data from the performance data file and
        measure the time.
        '''
        
        # Read the performance data
        try:
            f = open(self.__perf_fname, 'r')
            data = f.read()
            f.close()
        except Exception as e:
            raise self.__reporter.error('Failed to read performance data from file "%s": ' % self.__perf_fname)

        # Measure the performance 
        perf_map = {}
        for line in [x for x in data.split('\n') if x != '' and not x.isspace()]:
            cols = [x for x in line.split(' ') if x != '' and not x.isspace()]
            assert len(cols) == 2, 'a line of performance data must have two columns: an id and a number'
            pid, pnum = cols
            pnum = float(pnum)
            if pid in perf_map:
                perf_map[pid].append(pnum)
            else:
                perf_map[pid] = [pnum]
        perf = 0
        for pnums in list(perf_map.values()):
            perf += reduce(lambda x,y: x+y, pnums, 0) / len(pnums)

        # Delete the performance data file
        os.unlink(self.__perf_fname)

        # Return the measured performance number
        if perf == 0:
            return {True: float('inf'), False: float('-inf')}[self.isToMinimize()]
        return perf


    #------------------------------------------------------

    def __runShellCmd(self, cmd):
        '''Run the given shell command and return the exit status and output'''

        p = subprocess.Popen([cmd], shell=True, executable="bash", close_fds=True,
                             stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        output = p.communicate()[0].rstrip()
        status = p.returncode
        return (status, output)

    #------------------------------------------------------

    def __run(self, cmd):
        '''
        Run the code using the given command and return the status,
        the dump messages, and the measured performance number.  
        '''

        # Run with rough performance measurement
        if self.__perf_count == 'rough':
            start_time = time.time()
            status, output = self.__runShellCmd(cmd)
            end_time = time.time()
            perf = end_time - start_time
            return (status, output, perf)

        # Run with precise performance measurement
        elif self.__perf_count == 'precise':
            status, output = self.__runShellCmd(cmd)
            perf = {True: float('inf'), False: float('-inf')}[self.isToMinimize()]
            if status == 0:
                perf = self.__measurePreciseTime()
            return (status, output, perf)            

        # Unrecognized performance counting method
        else:
            raise self.__reporter.error('Unrecognized performance counting method: ' % self.__perf_count)

    #------------------------------------------------------

    def __cleanBuildRun(self, coord, mdata, skip_run):
        '''
        Clean, build and run the code. "skip_run" indicates if the
        code run is skipped. It returns the status and the measure
        performance number.
        '''

        # Initialize the status
        #   0: successful
        #   1: error during cleaning
        #   2: error during building (e.g., compilation error)
        #   3: error during running (e.g., crash)
        status = 0

        # Clean previously generated codes (if needed)
        if status == 0:
            if self.__clean_cmd != None:
                self.__reporter.logMessage('Cleaning previously generated codes: "%s"' % self.__clean_cmd)
                clean_status, output = self.__runShellCmd(self.__clean_cmd)
                self.__reporter.logMessage('Clean dump messages: \n%s' % output)
                if clean_status != 0:
                    status = 1
                    self.__reporter.logMessage('Cleaning failed')

        # Remember the existing files before build
        files_table = {}
        if status == 0 and not skip_run and self.__keep:
            files_table = self.__file_keeper.init()

        # Build the code
        if status == 0:
            self.__reporter.logMessage('Building the code: "%s"' % self.__build_cmd)
            build_status, output = self.__runShellCmd(self.__build_cmd)
            self.__reporter.logMessage('Build dump messages: \n%s' % output)
            if build_status != 0:
                status = 2
                self.__reporter.logMessage('Building failed')
        
        # Find and store new and modified files (if needed)
        if (status == 0 or status == 2) and not skip_run and self.__keep:
            self.__file_keeper.store(files_table, coord, mdata)

        # Run and measure the performance of the generated code (if wanted)
        perf = {True: float('inf'), False : float('-inf')}[self.isToMinimize()]
        if status == 0:
            if not skip_run:
                self.__reporter.logMessage('Running the code: "%s"' % self.__run_cmd)
                run_status, run_output, run_perf = self.__run(self.__run_cmd)
                self.__reporter.logMessage('Run dump messages: \n%s' % run_output)
                if run_status != 0:
                    status = 3
                    self.__reporter.logMessage('Running failed')
                else:
                    perf = run_perf
                    self.__reporter.logMessage('Code performance: %s %s' % (perf, self.perfMetric()))

        # Return the status and the performance number
        return (status, perf)

    #------------------------------------------------------

    def __cleanBuildRunWithOpts(self, coord, mdata, skip_run):
        '''
        Clean, build and run the code with some options set (using
        environment variables). "skip_run" indicates if the code run
        is skipped. It returns the status and the measure performance
        number.
        '''

        # Set the needed environment variables (for the RCC options)
        evars = []
        if coord != None:
            assert mdata != None, 'meta data info must not be None'
            self.__reporter.logMessage('Sets environment variables:')
            for ID, opt in mdata.getOptions(coord):
                n_ID = '%s_%s' % (self.__ARCC_OPTION, ID)
                os.environ[n_ID] = opt
                evars.append(n_ID)
                self.__reporter.logMessage('  %s=%s ' % (n_ID, opt))

        # Clean, build and run the code
        status, perf = self.__cleanBuildRun(coord, mdata, skip_run)
                
        # Flush the environment variables
        if coord != None:
            assert mdata != None, 'meta data info must not be None'
            s = ''
            s += 'Flushes environment variables: '
            for i, ev in enumerate(evars):
                if i > 0:
                    s += ', '
                s += '%s' % ev
                del os.environ[ev]
            self.__reporter.logMessage(s)

        # Return the status and the performance number
        return status, perf

    #------------------------------------------------------
    
    def __recordError(self, status, coord, mdata):
        '''Record in the log file the error happening during cleaning, building, or running'''
        
        s = ''
        if status == 1:
            s += 'CLEAN ERROR: \n'
        elif status == 2:
            s += 'BUILD ERROR: \n'
        elif status == 3:
            s += 'RUN ERROR: \n'
        else:
            assert status == 0, 'unrecognized status'
        s += ' Coordinate = %s \n' % coord
        s += ' Options = \n'
        for ID, opt in mdata.getOptions(coord):
            s += '   %s = %s \n' % (ID, opt)
        self.__reporter.logError(s)

    #------------------------------------------------------

    def build(self, coord=None, mdata=None):
        '''Build the code and return the status.'''

        status, _ = self.__cleanBuildRunWithOpts(coord, mdata, True)
        return status

    #------------------------------------------------------

    def run(self, coord, mdata):
        '''Build and run the code, and return the measured code performance'''

        assert coord != None and mdata != None, 'coordinate and meta data info must not be None'
        status, perf = self.__cleanBuildRunWithOpts(coord, mdata, False)
        if status != 0:
            self.__recordError(status, coord, mdata)
        return perf

