# DOE-GPU rights
#
# Author      : Albert Hartono <hartonoa@reservoir.com>
# Description : Reporting tool for ARCC
#
# $Id$
#

import os

#----------------------------------------------------------

class MessageReporter:
    '''The class definition of the reporting tool (e.g., logging, error message)'''

    #------------------------------------------------------
    
    def __init__(self, module_name, verbose, log_prefix):
        '''Instantiate a reporting tool'''
        
        self.__module_name = module_name  # the name of the module using this reporting tool
        self.__verbose = verbose  # print details of the running program
        self.__log_prefix = log_prefix  # the prefix used for naming the log files

        # Generate the names of the log files
        prefix = '%s%s%s' % (log_prefix, {True:'-', False:''}[len(log_prefix)!=0], module_name)
        self.__main_log_fname = os.path.abspath('%s-main.log' % prefix)
        self.__result_log_fname = os.path.abspath('%s-result.log' % prefix)
        self.__error_log_fname = os.path.abspath('%s-error.log' % prefix)

        # Delete log files (if exist)
        self.__deleteLogFiles()

    #------------------------------------------------------
    
    def __deleteLogFiles(self):
        '''Delete all log files (if exist)'''

        if os.path.exists(self.__main_log_fname):
            assert os.path.exists(self.__result_log_fname) and os.path.exists(self.__error_log_fname), 'missing other log files'
            os.unlink(self.__main_log_fname)
            os.unlink(self.__result_log_fname)
            os.unlink(self.__error_log_fname)

    #------------------------------------------------------
    
    def __createLogFiles(self):
        '''Create all log files (if not exist)'''

        # Skip if the log files already exist
        if os.path.exists(self.__main_log_fname):
            assert os.path.exists(self.__result_log_fname) and os.path.exists(self.__error_log_fname), 'missing other log files'
            return

        # Create all log files
        self.__createLogFile(self.__main_log_fname)  
        self.__createLogFile(self.__result_log_fname)  
        self.__createLogFile(self.__error_log_fname)

        # Add info header in each log file
        self.__appendToLog('# %s' % self.__main_log_fname, self.__main_log_fname)
        self.__appendToLog('#', self.__main_log_fname)
        self.__appendToLog('# The main log file for ARCC that records all steps of ARCC auto-tuning process', self.__main_log_fname)
        self.__appendToLog('#', self.__main_log_fname)
        self.__appendToLog('', self.__main_log_fname)

        self.__appendToLog('# %s' % self.__result_log_fname, self.__result_log_fname)
        self.__appendToLog('#', self.__result_log_fname)
        self.__appendToLog('# The log file for ARCC that records all best results found at each search-space exploration step', self.__result_log_fname)
        self.__appendToLog('#', self.__result_log_fname)
        self.__appendToLog('', self.__result_log_fname)

        self.__appendToLog('# %s' % self.__error_log_fname, self.__error_log_fname)
        self.__appendToLog('#', self.__error_log_fname)
        self.__appendToLog('# The error log file for ARCC that records all encountered failures', self.__error_log_fname)
        self.__appendToLog('#', self.__error_log_fname)
        self.__appendToLog('', self.__error_log_fname)

    #------------------------------------------------------
    
    def __createLogFile(self, log_fname):
        '''Create a log file'''
        
        try:
            f = open(log_fname, 'w')
            f.write('')
            f.close()
        except Exception, e:
            raise self.error('Failed to create log file: %s' % log_fname)

    #------------------------------------------------------

    def __message(self, msg):
        '''Create a message (with the module name prepended)'''
        
        return '[%s] %s' % (self.__module_name, msg)

    #------------------------------------------------------

    def __errorMessage(self, msg):
        '''Create an error message (with the module name prepended)'''
        
        return self.__message('Error: %s' % msg)

    #------------------------------------------------------
    
    def __appendToLog(self, msg, log_fname):
        '''Append message to log file'''

        # Create log files if they don't exist yet
        self.__createLogFiles()

        # Append message to the specified log file
        try:
            f = open(log_fname, 'a')
            f.write(msg + '\n')
            f.close()
        except Exception, e:
            raise self.error('Failed to write to log file: %s' % log_fname)

    #------------------------------------------------------

    def logFiles(self):
        '''Return paths to all log files'''
        
        return [self.__main_log_fname, self.__result_log_fname, self.__error_log_fname]
        
    #------------------------------------------------------

    def moduleName(self):
        '''Return the module name'''
        
        return self.__module_name
        
    #------------------------------------------------------

    def error(self, msg):
        '''Return an error (that is expected to be caught later)'''
        
        return Exception(self.__errorMessage(msg))
        
    #------------------------------------------------------

    def logMessage(self, msg):
        '''Write the given message into the main log file'''

        msg = self.__message(msg)
        if self.__verbose: print msg
        self.__appendToLog(msg, self.__main_log_fname)
        
    #------------------------------------------------------

    def logError(self, msg):
        '''Write the given error message into the error log file (and the main log file)'''
        
        self.__appendToLog(msg, self.__error_log_fname)
        msg = self.__message(msg)
        if self.__verbose: print msg
        self.__appendToLog(msg, self.__main_log_fname)
        
    #------------------------------------------------------

    def logResult(self, msg):
        '''Write the given result message into the result log file (and the main log file)'''

        self.__appendToLog(msg, self.__result_log_fname)
        msg = self.__message(msg)
        if self.__verbose: print msg
        self.__appendToLog(msg, self.__main_log_fname)
        
