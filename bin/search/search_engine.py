# DOE-GPU rights
#
# Author      : Albert Hartono <hartonoa@reservoir.com>
# Description : Main class for the tuning search engine
#
# $Id$
#

#----------------------------------------------------------

class SearchEngine:
    '''The class definition used for defining the behavior of the tuning search mechanism'''

    def __init__(self, reporter, executor, max_trials):
        '''Instantiate a search engine object'''

        self.reporter = reporter  # the reporting tool
        self.executor = executor  # the code executor
        self.max_trials = max_trials  # the maximum number of search trials

    #------------------------------------------------------

    def recordBestResult(self, best_perf, best_coord, mdata):
        '''Record in the log file the best performance number and its corresponding coordinate'''

        s = ''
        s += 'Found best new performance \n'
        s += 'Performance = %s %s \n' % (best_perf, self.executor.perfMetric())
        s += '  Coordinate = %s \n' % best_coord
        s += '    Options = \n'
        for ID, opt in mdata.getOptions(best_coord):
            s += '      %s = %s \n' % (ID, opt)
        self.reporter.logResult(s)

    #------------------------------------------------------

    def recordBestCoord(self, best_coord, mdata):
        '''Record in the log file the equally best coordinate'''

        s = ''
        s += 'Found new coordinate with equally best performance \n'
        s += '  Coordinate = %s \n' % best_coord
        s += '    Options = \n'
        for ID, opt in mdata.getOptions(best_coord):
            s += '      %s = %s \n' % (ID, opt)
        self.reporter.logResult(s)

    #------------------------------------------------------

    def search(self, mdata):
        '''
        Explore the tuning search space and return a tuple of two
        things: the performance number, and a list of equally best
        coordinates.
        '''

        raise NotImplementedError('%s: unimplemented abstract function "search"' % self.__class__.__name__)
    
        



