# DOE-GPU rights
#
# Author      : Albert Hartono <hartonoa@reservoir.com>
# Description : Nelder-Mead simplex search algorithm
# Reference: 
#  Jeffrey C. Lagarias, James A. Reeds, Margaret H. Wright,
#  and Paul E. Wright. 1998. Convergence Properties of the
#  Nelder--Mead Simplex Method in Low Dimensions. SIAM J. on
#  Optimization 9, 1 (May 1998).
#
# $Id$
#

import random


#----------------------------------------------------------

class SimplexSearch(search_engine.SearchEngine):
    '''The class definition for the Nelder-Mead simplex search algorithm.'''

    # Simplex-specific coefficients used for determining the next
    # moves in the search algorithm
    __reflection_coef = 1.0
    __expansion_coef = 2.0
    __contraction_coef = 0.5
    __shrinkage_coef = 0.5

    #------------------------------------------------------

    def initRandomSeed(self, num):
        '''Initialize basic random generator (to make the search deterministic)'''

        random.seed(num)

    #------------------------------------------------------
        
    def __evalPerf(self, visited_coord_perfs, coord, mdata, nb_trials):
        '''
        Empirically evaluate the performance cost of the code
        corresponding to the given coordinate.
        '''
        
        # Check if the given coordinate has been previously run
        coord_str = str(coord).replace(' ', '')
        if coord_str in visited_coord_perfs:
            return visited_coord_perfs[coord_str]
        
        # If the coordinate is in the considered search space, then
        # empirically evaluate the given coordinate for its performance
        if (not mdata.isOutsideSpace(coord)) and mdata.isNotPruned(coord):
            perf = self.executor.run(coord, mdata)
            nb_trials[0] += 1

        # Check if the given coordinate is pruned out (not in the
        # considered search space)
        else: 
            perf = {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]

        # Record the performance of the given coordinate
        visited_coord_perfs[coord_str] = perf

        # Return the performance cost of the given coordinate
        return perf

    #------------------------------------------------------

    def __nextRandomCoord(self, dim_sizes):
        '''Randomly return the next coordinate in the search space we want to visit.'''
        
        coord = [-1] * len(dim_sizes)
        for i in range(len(coord)):
            coord[i] = random.randint(0, dim_sizes[i]-1)
        return coord

    #------------------------------------------------------
    
    def __initSimplex(self, dim_sizes, space_size):
        '''
        Form a simplex in the search space using random coordinates. A
        simplex is a N+1-dimensional polytope, where N is the total
        number of dimension of the search space.
        '''

        # Check if the search space is valid for Nelder-Mead simplex
        # search algorithm (the search space can be just too small)
        nb_dims = len(dim_sizes)
        if nb_dims+1 > space_size:
            raise self.reporter.error('The search space is too small for the Nelder-Mead simplex algorithm. \n' + 
                                      'please use the exhaustive search instead.')

        # Generate a random simplex
        simplex = []
        while True:
            coord = self.__nextRandomCoord(dim_sizes)
            if coord not in simplex:
                simplex.append(coord)
                if len(simplex) == nb_dims+1:
                    break
        simplex.sort()
        return simplex

    #------------------------------------------------------

    def __roundInt(self, i):
        '''Proper rounding for an integer'''

        return int(round(i))

    def __subCoords(self, coord1, coord2):
        '''coord1 - coord2'''
        
        return map(lambda x,y: x-y, coord1, coord2)

    def __addCoords(self, coord1, coord2):
        '''coord1 + coord2'''
        
        return map(lambda x,y: x+y, coord1, coord2)

    def __scaleCoord(self, coef, coord):
        '''coef * coord'''
        
        return map(lambda x: self.__roundInt(1.0*coef*x), coord)

    #------------------------------------------------------

    def __centroid(self, coords):
        '''Compute the centroid of the given set of coordinates'''

        nb_coords = len(coords)
        centroid = coords[0]
        for c in coords[1:]:
            centroid = self.__addCoords(centroid, c)
        centroid = self.__scaleCoord((1.0 / nb_coords), centroid)
        return centroid
        
    #------------------------------------------------------

    def __reflection(self, coord, centroid):
        '''Compute reflection coordinate'''

        sub_coord = self.__subCoords(centroid, coord)
        return self.__addCoords(centroid, self.__scaleCoord(self.__reflection_coef, sub_coord))

    def __expansion(self, coord, centroid):
        '''Compute expansion coordinate'''

        sub_coord = self.__subCoords(coord, centroid)
        return self.__addCoords(centroid, self.__scaleCoord(self.__expansion_coef, sub_coord))

    def __contraction(self, coord, centroid):
        '''Compute contraction coordinate'''

        sub_coord = self.__subCoords(coord, centroid)
        return self.__addCoords(centroid, self.__scaleCoord(self.__contraction_coef, sub_coord))

    def __shrinkage(self, coord, rest_coords):
        '''Compute shrinkage simplex'''

        return map(lambda x: self.__addCoords(coord, self.__scaleCoord(self.__shrinkage_coef, self.__subCoords(x, coord))), rest_coords)

    #------------------------------------------------------

    def __compareResults(self, cur_perf, cur_coord, best_perf, best_coords, mdata):
        '''
        Compare the given new performance with previously obtained
        best performances, then return the updated best results.
        '''

        if {True: cur_perf < best_perf, False: cur_perf > best_perf}[self.executor.isToMinimize()]:
            best_perf = cur_perf
            best_coords = [cur_coord]
            self.recordBestResult(best_perf, cur_coord, mdata)
        elif (cur_perf == best_perf and cur_coord not in best_coords and 
              best_perf != {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]):
            best_coords.append(cur_coord)
            self.recordBestCoord(cur_coord, mdata)
        return (best_perf, best_coords)

    #------------------------------------------------------
        
    def search(self, mdata):
        '''
        Explore the tuning search space and return a tuple of two
        things: the performance number, and a list of equally best
        coordinates.
        '''

        # Print some opening messages
        self.reporter.logMessage('Start Nelder-Mead simplex search')

        # Record the best performance and the best-performing coordinates so far
        best_perf = {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]
        best_coords = [] 

        # Initialize the number of trials (use a list to mimic a pointer)
        nb_trials = [0]

        # Initialize a storage for recording the performances of
        # previously visited coordinates
        visited_coord_perfs = {}

        # The record of several last simplexes. This is for detecting
        # the presence of infinite looping (needed for termination of
        # the algorithm)
        last_N_simplexes = []

        # Initialize the first simplex (randomly)
        dim_sizes = mdata.dimSizes()
        space_size = {True: 0, False: reduce(lambda x, y: x*y, dim_sizes, 1)}[len(dim_sizes) == 0]
        simplex = self.__initSimplex(dim_sizes, space_size)

        # Get the performance costs for each coordinate in the obtained simplex
        simplex_perfs = []
        for coord in simplex:

            # Build and run the current coordinate, and empirically measure the performance
            perf = self.__evalPerf(visited_coord_perfs, coord, mdata, nb_trials)
            simplex_perfs.append(perf)

            # Update results then halt if the maximum number of trials is reached
            best_perf, best_coords = self.__compareResults(perf, coord, best_perf, best_coords, mdata)
            if self.max_trials > 0 and nb_trials[0] >= self.max_trials:
                self.reporter.logMessage('Finishes the Nelder-Mead simplex search')
                if best_perf == {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]:
                    return None
                else:
                    return (best_perf, best_coords)

        # Start the Nelder-Mead simplex search
        while True:
            
            # Sort the simplex coordinates based on their performance costs
            sorted_simplex_coords = zip(simplex, simplex_perfs)
            if self.executor.isToMinimize():
                sorted_simplex_coords.sort(lambda x,y: cmp(x[1],y[1]))
            else:
                sorted_simplex_coords.sort(lambda x,y: -cmp(x[1],y[1]))

            # Unbox the coordinate-performance tuples
            simplex, simplex_perfs = zip(*sorted_simplex_coords)
            simplex = list(simplex)
            simplex_perfs = list(simplex_perfs)
            
            # Print some info about the current simplex
            self.reporter.logMessage('Current simplex: %s' % simplex)
            self.reporter.logMessage('Performance numbers: %s' % simplex_perfs)
            
            # Termination criterion: when a looping is detected in the
            # search steps, then halt
            if str(simplex) in last_N_simplexes:
                self.reporter.logMessage('Convergence reached. Simplex: %s' % simplex)
                break
            
            # Record the current simplex
            last_N_simplexes.append(str(simplex))
            while len(last_N_simplexes) > 10:
                last_N_simplexes.pop(0)
            
            # Best coordinate
            cur_best_coord = simplex[0]
            cur_best_perf = simplex_perfs[0]

            # Worst coordinate
            cur_worst_coord = simplex[-1]
            cur_worst_perf = simplex_perfs[-1]
                
            # 2nd worst coordinate
            cur_2nd_worst_coord = simplex[-2]
            cur_2nd_worst_perf = simplex_perfs[-2]
            
            # Compute the centroid
            centroid = self.__centroid(simplex[:-1])

            # Reflection
            refl_coord = self.__reflection(cur_worst_coord, centroid)
            refl_perf = self.__evalPerf(visited_coord_perfs, refl_coord, mdata, nb_trials)
            
            # Update results then halt if the maximum number of trials is reached
            best_perf, best_coords = self.__compareResults(refl_perf, refl_coord, best_perf, best_coords, mdata)
            if self.max_trials > 0 and nb_trials[0] >= self.max_trials:
                self.reporter.logMessage('Finishes the Nelder-Mead simplex search')
                if best_perf == {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]:
                    return None
                else:
                    return (best_perf, best_coords)

            # The next new coordinate
            next_coord = None
            next_perf = None

            # If cost(cur_best) <= cost(cur_reflection) < cost(cur_2nd_worst) 
            # (We assume to minimize cost)
            if {True: cur_best_perf <= refl_perf < cur_2nd_worst_perf,
                False: cur_best_perf >= refl_perf > cur_2nd_worst_perf}[self.executor.isToMinimize()]: 
                next_coord = refl_coord
                next_perf = refl_perf
                self.reporter.logMessage('Reflection to %s' % next_coord)

            # If cost(reflection) < cost(cur_best)
            # (We assume to minimize cost)
            elif {True: refl_perf < cur_best_perf, False: refl_perf > cur_best_perf}[self.executor.isToMinimize()]:
                
                # Expansion
                exp_coord = self.__expansion(refl_coord, centroid)
                exp_perf = self.__evalPerf(visited_coord_perfs, exp_coord, mdata, nb_trials)

                # Update results then halt if the maximum number of trials is reached
                best_perf, best_coords = self.__compareResults(exp_perf, exp_coord, best_perf, best_coords, mdata)
                if self.max_trials > 0 and nb_trials[0] >= self.max_trials:
                    self.reporter.logMessage('Finishes the Nelder-Mead simplex search')
                    if best_perf == {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]:
                        return None
                    else:
                        return (best_perf, best_coords)

                # If cost(expansion) < cost(reflection)
                # (We assume to minimize cost)
                if {True: exp_perf < refl_perf, False: exp_perf > refl_perf}[self.executor.isToMinimize()]:
                    next_coord = exp_coord
                    next_perf = exp_perf
                    self.reporter.logMessage('Expansion to %s' % next_coord)
                else:
                    next_coord = refl_coord
                    next_perf = refl_perf
                    self.reporter.logMessage('Reflection to %s' % next_coord)

            # If cost(reflection) < cost(cur_worst)
            # (We assume to minimize cost)
            elif {True: refl_perf < cur_worst_perf, False: refl_perf > cur_worst_perf}[self.executor.isToMinimize()]: 
                
                # Outer contraction
                cont_coord = self.__contraction(refl_coord, centroid) 
                cont_perf = self.__evalPerf(visited_coord_perfs, cont_coord, mdata, nb_trials)

                # Update results then halt if the maximum number of trials is reached
                best_perf, best_coords = self.__compareResults(cont_perf, cont_coord, best_perf, best_coords, mdata)
                if self.max_trials > 0 and nb_trials[0] >= self.max_trials:
                    self.reporter.logMessage('Finishes the Nelder-Mead simplex search')
                    if best_perf == {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]:
                        return None
                    else:
                        return (best_perf, best_coords)
                
                # If cost(contraction) < cost(reflection)
                # (We assume to minimize cost)
                if {True: cont_perf < refl_perf, False: cont_perf > refl_perf}[self.executor.isToMinimize()]:
                    next_coord = cont_coord
                    next_perf = cont_perf
                    self.reporter.logMessage('Outer contraction to %s' % next_coord)
            
            # If cost(reflection) >= cost(cur_worst)
            # (We assume to minimize cost)
            else:
                
                # Inner contraction
                cont_coord = self.__contraction(cur_worst_coord, centroid) 
                cont_perf = self.__evalPerf(visited_coord_perfs, cont_coord, mdata, nb_trials)

                # Update results then halt if the maximum number of trials is reached
                best_perf, best_coords = self.__compareResults(cont_perf, cont_coord, best_perf, best_coords, mdata)
                if self.max_trials > 0 and nb_trials[0] >= self.max_trials:
                    self.reporter.logMessage('Finishes the Nelder-Mead simplex search')
                    if best_perf == {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]:
                        return None
                    else:
                        return (best_perf, best_coords)
                
                # If cost(contraction) < cost(cur_worst)
                # (We assume to minimize cost)
                if {True: cont_perf < cur_worst_perf, False: cont_perf > cur_worst_perf}[self.executor.isToMinimize()]:
                    next_coord = cont_coord
                    next_perf = cont_perf
                    self.reporter.logMessage('Inner contraction to %s' % next_coord)
            
            # If shrinkage is needed
            if next_coord == None and next_perf == None:
                
                # Shrinkage
                simplex = self.__shrinkage(cur_best_coord, simplex)
                self.reporter.logMessage('Shrinkage to %s' % cur_best_coord)
                simplex_perfs = []
                for coord in simplex:

                    # Build and run the current coordinate, and empirically measure the performance
                    perf = self.__evalPerf(visited_coord_perfs, coord, mdata, nb_trials)
                    simplex_perfs.append(perf)

                    # Update results then halt if the maximum number of trials is reached
                    best_perf, best_coords = self.__compareResults(perf, coord, best_perf, best_coords, mdata)
                    if self.max_trials > 0 and nb_trials[0] >= self.max_trials:
                        self.reporter.logMessage('Finishes the Nelder-Mead simplex search')
                        if best_perf == {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]:
                            return None
                        else:
                            return (best_perf, best_coords)

            # Delete the worst coordinate in the current simplex, and replace it with the obtained coordinate
            else:
                simplex.pop()
                simplex_perfs.pop()
                simplex.append(next_coord)
                simplex_perfs.append(next_perf)
        
        # Print some closing messages
        self.reporter.logMessage('Finishes the Nelder-Mead simplex search')

        # Return the search results
        if best_perf == {True: float('inf'), False: float('-inf')}[self.executor.isToMinimize()]:
            return None
        else:
            assert best_perf == simplex_perfs[0], 'unmatched performance'
            return (best_perf, best_coords)
        





