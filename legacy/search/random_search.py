# DOE-GPU rights
#
# Author      : Albert Hartono <hartonoa@reservoir.com>
# Description : Random search algorithm
#
# $Id$
#

import random


#----------------------------------------------------------

class RandomSearch(search_engine.SearchEngine):
    '''The class definition for the random search algorithm'''

    #------------------------------------------------------
        
    def initRandomSeed(self, num):
        '''Initialize basic random generator (to make the search deterministic)'''
        
        random.seed(num)

    #------------------------------------------------------

    def __nextRandomCoord(self, visited_coords, dim_sizes, space_size):
        '''
        Randomly return the next coordinate in the search space we
        want to visit. Return None if all coordinates in the
        considered search space have been visited.
        '''

        while True:

            # Return None if all coordinates in the search space have been visited
            if len(visited_coords) == space_size:
                return None

            # Create a random coordinate
            coord = [-1] * len(dim_sizes)
            for i in range(len(coord)):
                coord[i] = random.randint(0, dim_sizes[i]-1)

            # Continue finding the next random coordinate, if already visited
            coord_str = str(coord).replace(' ', '')
            if coord_str in visited_coords:
                continue

            # Record and return the current random coordinate, if not previously visited
            visited_coords[coord_str] = None
            return coord

    #------------------------------------------------------
        
    def search(self, mdata):
        '''
        Explore the tuning search space and return a tuple of two
        things: the performance number, and a list of equally best
        coordinates.
        '''

        # Print some opening messages
        self.reporter.logMessage('Start the random search')

        # Record the best performance and the best-performing coordinates so far
        best_perf = {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]
        best_coords = [] 

        # Initialize a storage for recording the previously visited coordinates
        visited_coords = {}

        # Start the search
        nb_trials = 0
        dim_sizes = mdata.dimSizes()
        space_size = {True: 0, False: reduce(lambda x, y: x*y, dim_sizes, 1)}[len(dim_sizes) == 0] 
        coord = self.__nextRandomCoord(visited_coords, dim_sizes, space_size)
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

            # Get the next random coordinate in the search space
            coord = self.__nextRandomCoord(visited_coords, dim_sizes, space_size)

        # Print some closing messages
        self.reporter.logMessage('Finishes the random search')

        # Return the search results
        if best_perf == {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]:
            return None
        else:
            return (best_perf, best_coords)
        
        
        
        



