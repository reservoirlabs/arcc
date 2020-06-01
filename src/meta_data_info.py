# DOE-GPU rights
#
# Author      : Albert Hartono <hartonoa@reservoir.com>
# Description : Meta data information
#
# $Id$
#

import re


class MetaDataInfo:
    '''The class that contains meta data information'''

    def __init__(self, line_no, ID, option, var_list, var_vals_list, constraint,
                 next):
        """
        Instantiate a meta data information
        """

        self.line_no = line_no  # the line position of this meta data info (in the meta data file)
        self.ID = ID  # the string identifier of this meta data
        self.option = option  # the option string 
        self.var_list = var_list  # the list of the variable names
        self.var_vals_list = var_vals_list  # the list of all variable values under consideration (same order as var_list)
        self.constraint = constraint  # the constraint expression
        self.next = next  # link to the next meta data 
        self.global_constraint = 'True'  # the global constraint expression

        # Evaluate the option string
        self.__evalOptionRHS()

        # Evaluate the variable values expressions
        self.__evalVarRHS()

        # Rename the variable names by mangling each name with the ID name
        self.__mangleVarNames()

    # ------------------------------------------------------

    def __evalOptionRHS(self):
        '''Evaluate the option string'''

        try:
            n_opt = eval(self.option)
        except Exception as e:
            raise Exception('Invalid option string for "%s" in "%s"' % (
                self.option, self.ID))
        if type(n_opt) != str:
            raise Exception(
                'Option value in "%s" must be of a string type' % self.ID)
        self.option = n_opt

    # ------------------------------------------------------

    def __evalVarRHS(self):
        '''Evaluate the RHS expressions of the variables'''

        # Check for redundant variable names
        for v in self.var_list:
            if self.var_list.count(v) > 1:
                raise Exception(
                    'Variable name "%s" defined multiple times in "%s"' % (
                        v, self.ID))

        # Check and evaluate the variable values (must be a list)
        n_var_vals_list = []
        for v, vals in zip(self.var_list, self.var_vals_list):
            try:
                assert type(
                    vals) == str, 'variable values must be initially of string types'
                n_vals = eval(vals)
                n_var_vals_list.append(n_vals)
            except Exception as e:
                raise Exception(
                    'Invalid variable values expression for "%s" in "%s"' % (
                        v, self.ID))
            if type(n_vals) != list:
                raise Exception(
                    'Variable values expression for "%s" in "%s" must be of a list type' % (
                        v, self.ID))
        self.var_vals_list = n_var_vals_list

    # ------------------------------------------------------

    def __mangleVarNames(self):
        '''Rename the variable names by mangling each name with the ID name'''

        # Rename the option string
        for v in self.var_list:
            self.option = re.sub(r'(\W)\$(%s)\$(\W)' % v,
                                 r'\1$%s_\2$\3' % self.ID,
                                 ' %s ' % self.option).strip()

        # Rename the constraint expression
        for v in self.var_list:
            self.constraint = re.sub(r'(\W)(%s)(\W)' % v,
                                     r'\1%s_\2\3' % self.ID,
                                     ' %s ' % self.constraint).strip()

        # Rename the variable names
        self.var_list = ['%s_%s' % (self.ID, x) for x in self.var_list]

    # ------------------------------------------------------

    def __coordToVarVals(self, coord):
        '''Return the size of the dimension i'''

        assert not self.isOutsideSpace(
            coord), 'given coordinate is outside the search space'
        this_coord = coord[:len(self.var_list)]
        rest_coord = coord[len(self.var_list):]
        vals = []
        for i, c in enumerate(this_coord):
            vals.append(self.var_vals_list[i][c])
        if self.next is None:
            return vals
        else:
            return vals + self.next.__coordToVarVals(rest_coord)

    # ------------------------------------------------------

    def __createEnv(self, coord):
        '''
        Create an environment that maps each variable name to its
        corresponding value (based on the given coordinate)
        '''

        assert not self.isOutsideSpace(
            coord), 'given coordinate is outside the search space'
        vars = []
        mdata = self
        while mdata is not None:
            vars.extend(mdata.var_list)
            mdata = mdata.next
        vals = self.__coordToVarVals(coord)
        return dict(list(zip(vars, vals)))

    # ------------------------------------------------------

    def __dimSizeAt(self, i):
        '''Return the size of the dimension i'''

        if i < 0:
            assert False, 'invalid negative dimension index'
        elif i < len(self.var_list):
            return len(self.var_vals_list[i])
        elif self.next is not None:
            return self.next.__dimSizeAt(i - len(self.var_list))
        else:
            assert False, 'invalid given dimension index'

    # ------------------------------------------------------

    def __totalDimNum(self):
        '''Return the total dimension'''

        if self.next == None:
            return len(self.var_list)
        else:
            return len(self.var_list) + self.next.__totalDimNum()

    # ------------------------------------------------------

    def globalConstraint(self, e):
        '''
        Add new expression to the global constraint. Note: the
        variables used in the global constraint are assumed to be
        already mangled with their meta data IDs.
        '''

        assert e is not None, 'the given constraint cannot be None'
        if self.global_constraint == 'True':
            self.global_constraint = e
        else:
            self.global_constraint = '(%s) and (%s)' % (
                self.global_constraint, e)

    # ------------------------------------------------------

    def add(self, mdata):
        '''Append a new meta data info to the end of this meta data list'''

        # Assert check
        assert mdata is not None, 'the given meta data must not be None'

        # Look for an empty next link
        cur_mdata = self
        while cur_mdata.next is not None:
            cur_mdata = cur_mdata.next

        # Append the new meta data
        cur_mdata.next = mdata

    # ------------------------------------------------------

    def dimSizes(self):
        '''Return the dimension sizes'''

        dsizes = []
        for i in range(0, self.__totalDimNum()):
            dsizes.append(self.__dimSizeAt(i))
        return dsizes

    # ------------------------------------------------------

    def isOutsideSpace(self, coord):
        '''
        Check if the given coordinate is outside the "non-pruned"
        search space (i.e., it's invalid)
        '''

        assert len(coord) == self.__totalDimNum(), 'invalid coordinate'
        for i, c in enumerate(coord):
            if c < 0 or c >= self.__dimSizeAt(i):
                return True
        return False

    # ------------------------------------------------------

    def isNotPruned(self, coord):
        '''
        Determine if the given coordinate is in the considered search
        space (i.e., not pruned out)
        '''

        assert not self.isOutsideSpace(
            coord), 'given coordinate is outside the search space'
        env = self.__createEnv(coord)
        is_not_pruned = True
        mdata = self
        while mdata is not None:
            try:
                is_not_pruned = is_not_pruned and eval(mdata.constraint, env)
            except Exception as e:
                raise Exception('Invalid constraint for %s: %s' % (
                    mdata.ID, mdata.constraint))
            try:
                is_not_pruned = is_not_pruned and eval(mdata.global_constraint,
                                                       env)
            except Exception as e:
                raise Exception(
                    'Invalid global constraint: "%s"' % mdata.global_constraint)
            mdata = mdata.next
        return is_not_pruned

    # ------------------------------------------------------

    def getOptions(self, coord):
        '''
        Get the options for the given coordinate. Returns a list of
        tuples. Each tuple contains two elements: the ID and its
        corresponding option.
        '''

        assert not self.isOutsideSpace(coord) and self.isNotPruned(
            coord), 'given coordinate is outside the search space'
        env = self.__createEnv(coord)
        opts = []
        mdata = self
        while mdata is not None:
            n_opt = mdata.option
            for v in mdata.var_list:
                v_re = '\$%s\$' % v
                if len(re.findall(v_re, n_opt)) == 0:
                    raise Exception(
                        'No variable "%s" found in the option of %s' % (
                            v, mdata.ID))
                n_opt = re.sub(v_re, str(env[v]), n_opt)
            for v in re.findall('\$\w+\$', n_opt):
                raise Exception(
                    'Undefined variable "%s" in the option of %s' % (
                        v, mdata.ID))
            opts.append((mdata.ID, n_opt))
            mdata = mdata.next
        return opts

    # ------------------------------------------------------

    def semantCheck(self, meta_data_map=None):
        '''Check the semantic correctness of all meta data info'''

        if meta_data_map is None:
            meta_data_map = {}
        # Check for redundant IDs
        if self.ID in meta_data_map:
            raise Exception('Meta data with ID "%s" already defined' % self.ID)

        # Take one valid coordinate
        coord = [0] * self.__totalDimNum()

        # Check the validity of the constraint expressions (by
        # evaluating it), with no care about the result
        self.isNotPruned(coord)

        # Check for variable usage in the options, with no care about
        # the results
        self.getOptions(coord)

        # Update the meta data map
        meta_data_map[self.ID] = self

        # Recurse to the next meta data info
        if self.next is not None:
            self.next.semantCheck(meta_data_map)

    # ------------------------------------------------------

    def __repr__(self):
        '''Return a string representation of this meta data info'''

        s = ''
        s += '%s {\n' % self.ID
        s += '  option = %s; \n' % self.option
        for (var, val) in zip(self.var_list, self.var_vals_list):
            s += '  var %s = %s; \n' % (var, val)
        s += '  constraint = %s; \n' % self.constraint
        s += '}\n'
        if self.global_constraint != 'True':
            s += '\n'
            s += 'global constraint = %s; \n' % self.global_constraint
        s += '%s' % {True: self.next, False: ''}[self.next is not None]
        return s
