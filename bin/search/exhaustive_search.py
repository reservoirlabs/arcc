# DOE-GPU rights
#
# Author      : Albert Hartono <hartonoa@reservoir.com>
# Description : Exhaustive search algorithm
#
# $Id$
#

import search.search_engine

#----------------------------------------------------------

class ExhaustiveSearch(search.search_engine.SearchEngine):
    '''The class definition for the exhaustive search algorithm'''

    #------------------------------------------------------

    def __nextCoord(self, coord, dim_sizes):
        '''
        Return the next coordinate in the search space we want to
        visit. Return None if all coordinates in the considered search
        space have been visited.
        '''
        
        next_coord = coord[:]
        for i in range(len(dim_sizes)-1, -1, -1):
            c = next_coord[i]
            dsize = dim_sizes[i]
            if c < dsize-1:
                next_coord[i] += 1
                break
            else:
                next_coord[i] = 0
                if i == 0:
                    return None
        return next_coord

    #------------------------------------------------------
        
    def search(self, mdata):
        '''
        Explore the tuning search space and return a tuple of two
        things: the performance number, and a list of equally best
        coordinates.
        '''

        # Print some opening messages
        self.reporter.logMessage('Start the exhaustive search')

        # Record the best performance and the best-performing coordinates so far
        best_perf = {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]
        best_coords = [] 

        # Start the search
        nb_trials = 0
        dim_sizes = mdata.dimSizes()
        coord = [0] * len(dim_sizes)
        while True:

            # Halt if all coordinates have been visited
            if coord == None:
                break
        
            # Check if the current coordinate is in the considered search space
            if mdata.isNotPruned(coord):
            
                # Build and run the code, and empirically measure the performance
                self.reporter.logMessage('Evaluating coordinate: %s' % coord)
                cur_perf = self.executor.run(coord, mdata)

                # Compare the measured performance with those of previously run codes
                if {True: cur_perf < best_perf, False: cur_perf > best_perf}[self.executor.isToMinimize()]:
                    best_perf = cur_perf
                    best_coords = [coord]
                    self.recordBestResult(best_perf, coord, mdata)
                elif cur_perf == best_perf and coord not in best_coords:
                    best_coords.append(coord)
                    self.recordBestCoord(coord, mdata)

                # Increment the number of trials
                nb_trials += 1

                # Halt if the maximum number of trials is reached
                if self.max_trials > 0 and nb_trials >= self.max_trials:
                    break

            # Get the next coordinate in the search space
            coord = self.__nextCoord(coord, dim_sizes)

        # Print some closing messages
        self.reporter.logMessage('Finishes the exhaustive search')

        # Return the search results
        if best_perf == {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]:
            return None
        else:
            return (best_perf, best_coords)
        

