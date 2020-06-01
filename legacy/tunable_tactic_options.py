# DOE-GPU rights
#
# Author      : Albert Hartono <hartonoa@reservoir.com>
# Description : Class for tunable tactic options
#
# $Id$
#

#----------------------------------------------------------

class TunableTacticOptions(object):
    '''The class for tunable tactic options'''

    def __init__(self, reporter):
        '''Instantiate tunable tactic options instance'''

        super(TunableTacticOptions, self).__init__()
        self.reporter = reporter

    #------------------------------------------------------

    def listTunableTacticOptions(self):
        '''Return a list of tunable tactic options'''

        raise NotImplementedError('%s: unimplemented abstract function "listTunableTacticOptions"' % self.__class__.__name__)

