# DOE-GPU rights
#
# Author      : Albert Hartono <hartonoa@reservoir.com>
# Description : The main class definition file for ARCC (Auto-tuner
#               for R-Stream C Compiler)
#
# $Id$
#


import getopt, os, re, sys
import code_executor, message_reporter, meta_data_parser, rstream_tunable_tactic_options
import search.exhaustive_search, search.random_search, search.simplex_search

#----------------------------------------------------------

class ARCC:
    '''The main class for ARCC (Auto-tuning R-Stream C Compiler)'''

    # ARCC modes
    __PRODUCE_MODE = 'produce'
    __CONSUME_MODE = 'consume'

    # ARCC's main environment variable names
    __ARCC_MODE = 'ARCC_MODE'
    __ARCC_METADATA = 'ARCC_METADATA'
    __ARCC_OPTIONUSEMODE = 'ARCC_OPTIONUSEMODE'
    __ARCC_PERF = 'ARCC_PERF'
    __ARCC_PERFFILE = 'ARCC_PERFFILE'

    # The helper message
    __usage_msg = '''
 Description: 
   Auto-tuner for RCC compiler (%s)

 Usage:  %s --build=<cmd> --run=<cmd> --clean=<cmd> [ options ]

   --build=<cmd>     The command for building the target program
   --run=<cmd>       The command for running the target program
   --clean=<cmd>     The command for cleaning the generated files

 Options:
   -h | --help           Print this helper message
   -v | --verbose        Print details of the running program
   --produce             Perform produce mode only (production of meta data by RCC)
   --consume             Perform consume mode only (search for best codes)
   --keep                Keep new and modified files in subdir "%s-codes"
   --list                The command for listing all autotunable tactic options
   --tune-all            Autotune all autotunable tactic options
   --meta-data=<file>    Name of meta data file
   --log=<prefix>        Prefix for naming ARCC-generated log files
   --exhaustive          Use exhaustive search
   --random              Use random search
   --simplex             Use Nelder-Mead Simplex search (used by default)
   --random-seed=<num>   Seed for random number generator (default: 12345)
   --max-trials=<num>    Max. limit of search iterations
   --rough-perf          Use rough performance counting method (used by default)
   --precise-perf        Use precise performance counting method 
                         (supported architectures: SMP, CUDA, Tilera, Cell)

 Example:
   %s --build="make foo" --run="./foo input.dat" --clean="make clean"

    '''

    #------------------------------------------------------
    
    def __init__(self, script_name, argv):
        '''Instantiate an ARCC object'''

        # Remember the script name (e.g., arcc)
        self.__script_name = script_name

        # Expand the helper usage message
        self.__usage_msg = self.__usage_msg % tuple([script_name]*4)  # expand message with the ARCC command

        # Parse the options
        (tune_all_tactic_options, build_cmd, run_cmd, clean_cmd, keep, meta_data_fname, verbose, log_prefix, max_trials, 
         random_seed, search_algo, produce_only, consume_only, perf_count) = self.__parseArgs(script_name, argv)

        # Make absolute paths for filenames
        self.__meta_data_fname = os.path.join(os.path.abspath(os.curdir), meta_data_fname)  # meta data filename
        self.__perf_fname = os.path.join(os.path.abspath(os.curdir), '%s-perf.dat' % script_name)  # performance data filename

        # Initialize class fields
        self.__reporter = message_reporter.MessageReporter(script_name, verbose, log_prefix)  # the reporting tool (e.g., for logging)
        self.__executor = code_executor.CodeExecutor(self.__reporter, build_cmd, run_cmd, clean_cmd, 
                                                     keep, perf_count, self.__perf_fname)  # the code executor
        self.__build_cmd = build_cmd  # the command for building the optimized code
        self.__run_cmd = run_cmd  # the command for running the optimized code
        self.__clean_cmd = clean_cmd  # the command for cleaning previously generated codes
        self.__tune_all_tactic_options = tune_all_tactic_options  # autotune all autotunable tactic options
        self.__keep = keep  # store new and modified files after each build in subdirectory
        self.__verbose = verbose  # print details of the running program
        self.__log_prefix = log_prefix  # the prefix used for naming the log files 
        self.__max_trials = max_trials  # the maximum limit of search trials
        self.__random_seed = random_seed  # the random seed number
        if search_algo == 'random':  # the search algorithm
            self.__search_engine = search.random_search.RandomSearch(self.__reporter, self.__executor, max_trials)
            self.__search_engine.initRandomSeed(random_seed)
        elif search_algo == 'simplex': 
            self.__search_engine = search.simplex_search.SimplexSearch(self.__reporter, self.__executor, max_trials)
            self.__search_engine.initRandomSeed(random_seed)
        else: 
            assert search_algo == 'exhaustive', 'unrecognized search algorithm: %s' % search_algo
            self.__search_engine = search.exhaustive_search.ExhaustiveSearch(self.__reporter, self.__executor, max_trials)
        self.__produce_only = produce_only  # only perform the meta data production (i.e., generation of meta data file by RCC)
        self.__consume_only = consume_only  # only perform the meta data consumption (i.e., the tuning process)
        self.__perf_count = perf_count  # the performance counting method

    #------------------------------------------------------

    def __parseArgs(self, script_name, argv):
        '''Parse the command line arguments'''

        # Reporting tool
        reporter = message_reporter.MessageReporter(script_name, False, '')

        # Read the command line arguments
        try:
            opts, _ = getopt.getopt(argv[1:], 
                                    'hv', 
                                    ['help', 'build=', 'run=', 'clean=', 'keep', 'meta-data=', 'verbose', 'log=', 'max-trials=', 
                                     'random-seed=', 'exhaustive', 'random', 'simplex', 'produce', 'consume',
                                     'rough-perf', 'precise-perf', 'list', 'tune-all'])
        except Exception, e:
            raise Exception('Error: %s \n %s' % (e, self.__usage_msg))

        # Set the options to their default values
        list_tactic_options = False
        tune_all_tactic_options = False
        build_cmd = None
        run_cmd = None
        clean_cmd = None
        keep = False
        meta_data_fname = '%s-meta-data.md' % self.__script_name
        verbose = False
        log_prefix = ''
        max_trials = -1
        random_seed = 12345
        search_algo = 'simplex'
        produce_only = False
        consume_only = False
        perf_count = 'rough' #FIXME: should be assigned to 'precise'

        # Get the options
        for opt, arg in opts:
            if opt in ('-h', '--help',):
                print self.__usage_msg
                sys.exit(1)
            elif opt in ('--list',):
                list_tactic_options = True
            elif opt in ('--tune-all',):
                tune_all_tactic_options = True
            elif opt in ('--build',):
                build_cmd = arg
            elif opt in ('--run',):
                run_cmd = arg
            elif opt in ('--clean',):
                clean_cmd = arg
            elif opt in ('--keep',):
                keep = True
            elif opt in ('--meta-data',):
                meta_data_fname = arg
            elif opt in ('-v', '--verbose',):
                verbose = True
            elif opt in ('--log',):
                log_prefix = arg
            elif opt in ('--max-trials',):
                max_trials = arg
            elif opt in ('--random-seed',):
                random_seed = arg
            elif opt in ('--exhaustive',):
                search_algo = 'exhaustive'
            elif opt in ('--random',):
                search_algo = 'random'
            elif opt in ('--simplex',):
                search_algo = 'simplex'
            elif opt in ('--produce',):
                produce_only = True
            elif opt in ('--consume',):
                consume_only = True
            elif opt in ('--rough-perf',):
                perf_count = 'rough'
            elif opt in ('--precise-perf',):
                perf_count = 'precise'

        # List all tunable tactic options and then stop
        if list_tactic_options:
            tactic_options_descriptions = rstream_tunable_tactic_options.RStreamTunableTacticOptions(reporter).listTunableTacticOptions()
            s = ''
            if len(tactic_options_descriptions) == 0:
                s += '[%s] NO autotunable tactic options currently available \n' % script_name
            else:
                s += '[%s] List of all autotunable tactic options: \n' % script_name
                s += '------------------------------------------------------------------------------\n' 
                s += '| No. |    Tactic    |    Option    | Used by | Description                   \n' 
                s += '|     |              |              | default |                               \n' 
                s += '------------------------------------------------------------------------------\n' 
                for i, (t, o, u, d) in enumerate(tactic_options_descriptions):
                    s += '| %2d. | %12s | %12s | %7s | %s \n' % (i+1, t, o, u, d)
                s += '------------------------------------------------------------------------------\n' 
            print s
            sys.exit(1)

        # Semantic checks and evaluations of the options
        if clean_cmd == '':
            clean_cmd = None
        if produce_only and consume_only:
            raise reporter.error('Cannot use --produce and --consume simultaneously \n %s' % self.__usage_msg)
        if build_cmd == '':
            raise reporter.error('Build command cannot be an empty string\n %s' % self.__usage_msg)
        if run_cmd == '':
            raise reporter.error('Run command cannot be an empty string\n %s' % self.__usage_msg)
        if build_cmd == None:
            raise reporter.error('Missing build command \n %s' % self.__usage_msg)
        if run_cmd == None:
            raise reporter.error('Missing run command \n %s' % self.__usage_msg)
        if not produce_only and clean_cmd == None:
            raise reporter.error('Missing clean command \n %s' % self.__usage_msg) 
        if type(max_trials) == str:
            try:
                max_trials = eval(max_trials)
            except Exception, e:
                raise reporter.error('Number of maximum trials must be a positive integer \n %s' % self.__usage_msg)
            if type(max_trials) != int or max_trials <= 0:
                raise reporter.error('Number of maximum trials must be a positive integer \n %s' % self.__usage_msg)
        if type(random_seed) == str:
            try:
                random_seed = eval(random_seed)
            except Exception, e:
                raise reporter.error('Random seed must be an integer \n %s' % self.__usage_msg)
            if type(random_seed) != int:
                raise reporter.error('Random seed must be an integer \n %s' % self.__usage_msg)
        if search_algo == 'random' and max_trials <= 0:
            raise reporter.error(('Random search requires a certain number of maximum trials. \n' + 
                             '  Please set the maximum search trials using --max-trials=<num>. \n %s') % self.__usage_msg)
        if meta_data_fname == '':
            raise reporter.error('Meta data file name cannot be an empty string\n %s' % self.__usage_msg)

        # Return the options
        return (tune_all_tactic_options, build_cmd, run_cmd, clean_cmd, keep, meta_data_fname, verbose, log_prefix, 
                max_trials, random_seed, search_algo, produce_only, consume_only, perf_count)

    #------------------------------------------------------

    def __printArgs(self):
        '''Print the options used'''
        
        s = ''
        s += 'Options used by ARCC: \n'
        s += '  Build command = %s \n' % self.__build_cmd
        s += '  Run command = %s \n' % self.__run_cmd
        s += '  Clean command = %s \n' % {True: self.__clean_cmd, False: ''}[self.__clean_cmd != None]
        s += '  Keep new and modified files = %s \n' % self.__keep
        s += '  Tune all autotunable tactic options = %s \n' % self.__tune_all_tactic_options
        s += '  Meta data file = %s \n' % self.__meta_data_fname
        s += '  Verbose = %s \n' % self.__verbose
        s += '  Log prefix = "%s" \n' % self.__log_prefix
        s += '  Search algorithm = %s \n' % self.__search_engine.__class__.__name__
        s += '  Max. search trials = %s ' % self.__max_trials + {True: '(no limit) \n', False: '\n'}[self.__max_trials < 0]
        s += '  Random seed number = %s \n' % self.__random_seed
        s += '  Produce only = %s \n' % self.__produce_only
        s += '  Consume only = %s \n' % self.__consume_only
        s += '  Performance counting method = %s \n' % self.__perf_count
        s += '  Performance data file = %s \n' % self.__perf_fname
        self.__reporter.logMessage(s)

    #------------------------------------------------------

    def __produceInit(self):
        '''Sets environment variables to begin the produce mode'''

        # Set the needed ARCC environment variables
        self.__reporter.logMessage('Sets environment variables: ')
        self.__reporter.logMessage('  %s=%s ' % (self.__ARCC_MODE, self.__PRODUCE_MODE))
        self.__reporter.logMessage('  %s=%s ' % (self.__ARCC_METADATA, self.__meta_data_fname))
        os.environ[self.__ARCC_MODE] = self.__PRODUCE_MODE
        os.environ[self.__ARCC_METADATA] = self.__meta_data_fname

    #------------------------------------------------------

    def __produceExit(self):
        '''Flushes environment variables to end the produce mode'''

        # Flush the environment variables
        s = 'Flushes environment variables: %s, %s, %s' % (self.__ARCC_MODE, self.__ARCC_METADATA, self.__ARCC_OPTIONUSEMODE)
        self.__reporter.logMessage(s)
        del os.environ[self.__ARCC_MODE]
        del os.environ[self.__ARCC_METADATA]
        del os.environ[self.__ARCC_OPTIONUSEMODE]

    #------------------------------------------------------

    def __build(self):
        '''Build the code and check the status'''

        # Clean previously generated codes (if needed) and build the code using RCC to generate the meta data file
        status = self.__executor.build()
        if status == 1:
            raise self.__reporter.error('Failed to clean previously generated codes (during production mode): "%s"' % self.__clean_cmd)
        elif status == 2:
            raise self.__reporter.error('The build failed to generate meta data file, using command: "%s".' % self.__build_cmd)
        else:
            assert status == 0, 'unrecognized status'

    #------------------------------------------------------

    def __produce(self):
        '''
        Enter the production mode, where ARCC uses RCC to generate the
        meta data file containing all required information
        representing the tuning search space.
        '''

        # Print some opening messages
        self.__reporter.logMessage('ARCC enters the production mode')
        
        # Sets environment variables to begin produce mode
        self.__produceInit()

        # Delete the meta data file if exists
        try:
            if os.path.exists(self.__meta_data_fname):
                self.__reporter.logMessage('Previous meta data file %s exists. Deleting it now.' % self.__meta_data_fname)
                os.unlink(self.__meta_data_fname)
        except Exception, e:
            raise self.__reporter.error('Failed to delete meta data file: ' % self.__meta_data_fname)

        # Determine the use mode of the autotunable tactic options
        if self.__tune_all_tactic_options:
            self.__reporter.logMessage('Set to use *all* autotunable tactic options')
            opt_use_mode = 'all'
        else:
            opt_use_mode = 'user-specified'
        self.__reporter.logMessage('Sets environment variables: %s=%s' % (self.__ARCC_OPTIONUSEMODE, opt_use_mode))
        os.environ[self.__ARCC_OPTIONUSEMODE] = opt_use_mode

        # Build the code
        self.__build()
        
        # Check if the meta data file was generated
        if os.path.exists(self.__meta_data_fname):
            self.__reporter.logMessage('Meta data file generated: %s' % self.__meta_data_fname)
        else:
            # Rebuild the code and enforce RCC to use its default
            # lists of autotuned tactic options
            if not self.__tune_all_tactic_options:
                self.__reporter.logMessage('No meta data file was generated, because no autotunable tactic options were explicitly specified. ' +
                                           'Enforce RCC to use its default lists of autotuned tactic options.')
                self.__reporter.logMessage('Sets environment variables: %s=%s' % (self.__ARCC_OPTIONUSEMODE, 'default'))
                os.environ[self.__ARCC_OPTIONUSEMODE] = 'default'
                
                # Rebuild the code
                self.__build()

            # Again check if meta data file was generated
            if os.path.exists(self.__meta_data_fname):
                self.__reporter.logMessage('Meta data file generated: %s' % self.__meta_data_fname)
            else:
                raise self.__reporter.error(('RCC failed to generate meta data file: %s. \n' +
                                             'No autotunable tactic options were found to be usable. \n' + 
                                             'Please report this error to rstream-support@reservoir.com.') % self.__meta_data_fname)
        
        # Flushes environment variables to end produce mode
        self.__produceExit()

        # Print some closing messages
        self.__reporter.logMessage('ARCC finishes the production mode')

    #------------------------------------------------------

    def __extractMetaData(self):
        '''Read and parse the meta data file to extract meta data information'''
        
        # Check if the meta data file exists
        if not os.path.exists(self.__meta_data_fname):
            raise self.__reporter.error('Missing meta data file: %s' % self.__meta_data_fname)

        # Parse the meta data file to retrieve all meta data info
        mdata = meta_data_parser.MetaDataParser(self.__reporter).parse(self.__meta_data_fname)

        # Return the meta data info
        return mdata

    #------------------------------------------------------

    def __consumeInit(self):
        '''Sets environment variables to begin the consume mode'''

        # Set the needed ARCC environment variables
        self.__reporter.logMessage('Sets environment variables: ')
        self.__reporter.logMessage('  %s=%s ' % (self.__ARCC_MODE, self.__CONSUME_MODE))
        self.__reporter.logMessage('  %s=%s ' % (self.__ARCC_PERF, self.__perf_count))
        self.__reporter.logMessage('  %s=%s ' % (self.__ARCC_PERFFILE, self.__perf_fname))
        os.environ[self.__ARCC_MODE] = self.__CONSUME_MODE
        os.environ[self.__ARCC_PERF] = self.__perf_count
        os.environ[self.__ARCC_PERFFILE] = self.__perf_fname

    #------------------------------------------------------

    def __consumeExit(self):
        '''Flushes environment variables to end the consume mode'''

        # Flush the environment variables
        s = 'Flushes environment variables: %s, %s, %s' % (self.__ARCC_MODE,
                                                           self.__ARCC_PERF,
                                                           self.__ARCC_PERFFILE)
        self.__reporter.logMessage(s)
        del os.environ[self.__ARCC_MODE]
        del os.environ[self.__ARCC_PERF]
        del os.environ[self.__ARCC_PERFFILE]

    #------------------------------------------------------

    def __consume(self, mdata):
        '''
        Enter the consumption mode, where ARCC empirically explores
        the tuning search space to find the best-performing code
        variant.
        '''

        # Print some opening messages
        self.__reporter.logMessage('ARCC enters the consumption mode')

        # Sets environment variables to begin consume mode
        self.__consumeInit()

        # Construct the search engine and start the empirical search
        perfs = self.__search_engine.search(mdata)
        
        # Flushes environment variables to end consume mode
        self.__consumeExit()

        # Print some closing messages
        self.__reporter.logMessage('ARCC finishes the consumption mode')

        # Return the search results
        return perfs

    #------------------------------------------------------

    def __reportResults(self, perfs, mdata):
        '''Make a report about the search results'''

        s = ''
        perf, coords = perfs
        for i, coord in enumerate(coords):
            s += '\n'
            if len(coords) > 1:
                s += ' ------------------\n'
                s += ' Code variant %s: \n' % (i+1)
                s += ' ------------------\n'
            s += ' Performance = %s %s\n' % (perf, self.__executor.perfMetric())
            s += ' Coordinate = %s \n' % coord
            s += ' Options = \n'
            for ID, opt in mdata.getOptions(coord):
                s += '   %s = %s \n' % (ID, opt)
        self.__reporter.logMessage('Search results: \n%s' % s)

    #------------------------------------------------------

    def __genBestCode(self, coord, mdata):
        '''Generate the best-performing code'''

        # Print some opening messages
        self.__reporter.logMessage('Generate the best-performing code (one that corresponds to coordinate %s) ' % coord)

        # Sets environment variables to begin consume mode
        self.__consumeInit()

        # Construct the search engine and start the empirical search
        status = self.__executor.build(coord, mdata)
        assert status == 0, 'build of best code must succeed'
        
        # Flushes environment variables to end consume mode
        self.__consumeExit()

    #------------------------------------------------------

    def run(self):
        '''
        Run the whole chain of R-Stream auto-tuning process, with
        fail-safe environmental variables flushing. Catch any error
        and properly flush ARCC-generated environment variables.
        '''

        # Remove any ARCC-related environment variables that may be
        # existent from previous ARCC runs
        for k in os.environ.keys():
            if k.startswith('ARCC_'):
                self.__reporter.logMessage('Warning: Found ARCC-related environment variable: %s. Flushing it now.' % k)
                del os.environ[k]
                
        # Remember all environment variables before running ARCC
        before_evars = os.environ.keys()

        # Run ARCC, catch any exception, and remove any ARCC-related
        # environment variables (if any)
        try:
            self.__run()
        except Exception, e:

            # Delete any ARCC-related environment variables that have
            # not been flushed. By convention, the variable names of
            # ARCC-related environment variables always have a prefix
            # of "ARCC_".
            after_evars = os.environ.keys()
            for k in after_evars:
                if k not in before_evars:
                    if k.startswith('ARCC_'):
                        self.__reporter.logMessage('Flushes ARCC-related environment variable: %s' % k)
                    else:
                        self.__reporter.logMessage('Flushes non-ARCC-related environment variable: %s' % k)
                    del os.environ[k]
            
            # Re-throw the exception
            raise e

    #------------------------------------------------------        

    def __run(self):
        '''(Internal) Run the whole chain of R-Stream auto-tuning process'''

        # Print some opening messages
        self.__reporter.logMessage('ARCC starts')
        self.__printArgs()

        # Skipping the production mode
        if self.__consume_only:
            self.__reporter.logMessage('Skipping the meta data production')

        # Enter the production mode to retrieve the meta data information
        else:
            self.__produce()
        
        # Skipping the consumption mode
        if self.__produce_only:
            self.__reporter.logMessage('Skipping the meta data consumption (the tuning process)')

        # Enter the consumption mode to empirically search for the best-performing code variants
        else:

            # Get the meta data info and enter the tuning process
            mdata = self.__extractMetaData()
            perfs = self.__consume(mdata)

            # Report the search results
            if perfs != None:
                self.__reportResults(perfs, mdata)
            else:
                self.__reporter.logMessage('No search results found. ' + 
                                        'This may be due to several possible reasons: \n' + 
                                        ' - The search space gets all pruned out (hence, no legal/feasible coordinates exist). \n' + 
                                        ' - The search space is too small (many coordinates get pruned out by constraints) \n' + 
                                        '   and the heuristic search might fail to explore the legal/feasible coordinates. \n' + 
                                        '   Please try to use a different random seed number (to change the behaviors of the heuristic search). \n' + 
                                        ' - All evaluations of legal/feasible coordinates may have some errors during the build or run. \n' + 
                                        '   Please check the error log file. \n' +
                                        ' - etc (some other unknown errors)')

            # Generate the best-performing code variant
            if perfs != None:
                _, coords = perfs
                self.__genBestCode(coords[0], mdata)
            
        # Print some closing messages
        self.__reporter.logMessage('ARCC finishes')
        
