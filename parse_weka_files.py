#!/usr/bin/python3

import sys
import os
import math

from os.path import join
from operator import itemgetter

# Maxime's functions
from config_file import get_config

# Weka's parameters.
list_of_string_params = [
    'reduced_error_pruning',
    'unpruned',
    'binary_splits',
    'save_instance_data',
    'subtree_raising',
    'config_laplace',
    'confidence_factor',
    'min_num_obj',
    'num_folds'
]


class ParseWekaFiles:
    def __init__(self, entry, element, n, order):
        """ ParseWekaFiles is a class that will be usefull to parse all the
        output files created by weka, in order to visualise and retrieve all the
        expected values. Thus, it will allow us to draw some conclusion.
        
        - entry: Init the class with an entry point, where the program will open
        in order to retrieve all the files created by weka.
        - Element: Select which type of the DecisionTree's element we are focused,
        which correspond to LEAVES or TREES
        - n: Corresponds to the numbers of elements we are actually focused.
        - order: if True then the BEST values will be proceed else WORST will be.
        - avoid: Internal attribute to process files only once.
        """
        # Save entries to attributes..
        self.entry = entry
        self.element = element
        self.n = n
        self.order = order
        self.flag = False
        # Contains all the filenames retrieved from repository 'entry'.
        self.list_of_files = os.listdir(str(entry))
        """ Contains all the expected values as list from each file contained in
        'self.list_of_files'.
        """
        self.list_of_values = []
        # Contains all the filename in a map for further purposes.
        self.map_of_files = {}


    def fetch_values_from_files(self):
        """ Fetch all the values from files."""
        """ 'id' allows us to a kind of One2Many relation
        from 'self.map_of_files' to 'self.list_of_values'.
        """
        id = 1
        # Get each file contained in 'self.list_of_files' as f.
        for f in self.list_of_files:
            # Create entries for 'self.map_of_files' and 'self.list_of_values'.
            self.map_of_files[id] = str(f)
            self.list_of_values.append([id])
            # Open the file in read mode as file.
            with open(join(self.entry, str(f)), 'r') as file:
                # Temporary list that will be added to the list 'self.list_of_files'.
                list_of_values_file = []
                # Read each line contained in 'file' in order to find expected ones.
                for line in file:
                    # If 'line' match with those to find then retrieve the value.
                    if (line.find("Number of Leaves") >= 0
                          or line.find("Size of the tree") >= 0):
                        list_of_values_file.append(int(line.split()[-1]))
                self.list_of_values[id - 1] += [int(v) for v in list_of_values_file]
            # Increment by 1 in order to create a new entry for next iteration.
            id += 1


    def sort_list_based_on_leaves(self):
        """ Perform a sort based on the size of leaves fetched from files."""
        return sorted(self.list_of_values, key=itemgetter(1))


    def sort_list_based_on_trees(self):
        """ Perform a sort based on the size of trees fetched from files."""
        return sorted(self.list_of_values, key=itemgetter(2))


    def sort_list_based_on_diff_lt(self):
        """ Perform a sort based on the size of leaves fetched from files.
        'lt' in the function name simply refer to: leaves and trees.

        KEEP IN MIND THAT THIS METHOD ISN'T ACTUALLY USED AND FIGURES ACTUALLY
        JUST AS AN EXPIREMENT.
        """
        # Create a temporary dict
        dic_of_diff = {}
        """ Fill dic_of_diff with:
        - Key: create relation with 'self.map_of_files'.
        - diff: simple difference between numbers of LEAVES and TREES.
        Since the sign of numbers are useless, here, we use abs()
        """
        for i in range(len(self.list_of_values)):
            dic_of_diff[self.list_of_values[i][0]] \
              = abs(self.list_of_values[i][1] - self.list_of_values[i][2])
        # Return the sorted dict based on diff
        return sorted(dic_of_diff.items(), key=itemgetter(1))


    def retrieve_params_from_each_file(self, list):
        """ Iterate through the list and retrieve params for each file.
        """
        # List that contains all params related to LEAVES and TREES' values.
        list_of_files_param = []
        for i in list:
            # Append each file's params
            list_of_files_param.append(
                get_config(self.map_of_files[int(i[0])], self.entry)
            )
        return list_of_files_param


    def find_suit_or_unsuit_configs(self, flag=0):
        """ This function will allow us to get a better overview
        of which type of configuration. The fetch works either for LEAVES and TREES.
        """
        # Retrieve all values from files JUST ONCE.
        if (not self.flag):
            self.fetch_values_from_files()
            self.flag = True
        # Simple if else to know if we are focus on LEAVES or TREES.
        if (self.element):
            l = self.sort_list_based_on_trees()
        else:
            l = self.sort_list_based_on_leaves()
        """ Ternary based on 'self.order' attribute in order to decide
        if we use the BEST or WORST values:
        """
        new_l = l[:self.n] if self.order else l[-self.n:]
        if (flag):
            list_of_files_param = self.retrieve_params_from_each_file(l)
        else:
            list_of_files_param = self.retrieve_params_from_each_file(new_l)
        return (new_l, list_of_files_param)


    def epurate_list(self, list):
        """ Given a list, this function will simply epurate the list by avoiding
        None values.
        """
        return [item for item in list if item is not None]


    def do_average_value(self, list):
        """ Perform avegare value.
        """
        average_value = 0
        for i in list:
            """ If it's a boolean type,
            then it's interpreted as 0 for False and 1 for True."""
            if isinstance(i, bool):
                average_value += 0 if not i else 1
            # Classic sum
            else:
                average_value += i
        # Div by the len of the list
        average_value /= len(list)
        return average_value


    def average_values(self, values, params):
        """" Perform the requested average value based on args.
        """
        # Initialise the list that will contain all the average values.
        list_of_average_values = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        """ This 'list' will be used for each weka's parameters,
        in order to retrieve all the values set for this parameter."""
        list = []
        # Iterate for each Weka's parameter.
        for i in range(len(list_of_string_params)):
            # Retrieve all values from the actual weka's parameter.
            for j in range(len(params)):
                list.append(params[j][list_of_string_params[i]])
            # Epurates the list of None value.
            list = self.epurate_list(list)
            # If the list is empty, e.g. full of None, then we simply set to 0.
            if (not len(list)):
                list_of_average_values[i] = 0
            # Perform the average value of the chosen weka's parameter.
            else:
               list_of_average_values[i] = self.do_average_value(list)
            # Once the average value done, we flush the list for the next iteration.
            del list[:]
        # All the important values are returned to be printed.
        return values, list_of_average_values


    def get_specific_average_values(self):
        """" Retrieve the average values on all weka's parameters',
        based on args.
        """
        # Retrieve our specific 'values' and their respective 'params'.
        values, params = self.find_suit_or_unsuit_configs()
        # Inform the user which type of values have been proceed.
        print("Specific run:", "LEAVES" if not self.element else "TREES")
        v, avg = self.average_values(values, params)
        return (v, avg)

    
    def get_all_average_values(self):
        """" Retrieve the average values on all weka's parameters' values.
        """
        # Retrieve all the 'values' and their respective 'params'.
        values, params = self.find_suit_or_unsuit_configs(1)
        v, avg = self.average_values(values, params)
        return (v, avg)
        
        
    def print_retrieved_configs_from_args(self, values, avg_values):
        """ Print all important values
        """
        print("%s %s %s %s" % ("THOSE ARE ACTUALLY THE", str(self.n),
                               "BEST" if self.order else "WORST", "values!\n"))
        for i in values:
            print("Filename: %s" % self.map_of_files[i[0]])
            print("Number of LEAVES: %s and number of TREES: %s\n" %
                  (str(i[1]), str(i[2])))
        for i in range(len(list_of_string_params)):
            print("%s %s" % (list_of_string_params[i], str(avg_values[i])))


    def print_retrieved_configs_from_all(self, values, avg_values):
        print("\n\nTHIS IS THE FULL RUN FOR ALL VALUES:")
        for i in range(len(list_of_string_params)):
            print("%s %s" % (list_of_string_params[i], str(avg_values[i])))


    def dump_all_attributes(self):
        """ Temporary method.
        """
        print(self.entry)
        print(self.element)
        print(self.n)
        print(self.order)
        print(self.flag)
        print(self.map_of_files)
        print(self.list_of_values)
        print(self.list_of_files)


if __name__ == '__main__':
    # Instatiatation of the class.
    p = ParseWekaFiles(
        str(sys.argv[1]), # Absolute path to the repository.
        int(sys.argv[2]), # Sort based on either LEAVES or TREES.
        int(sys.argv[3]), # Decide the numbers of elements to be proceed.
        int(sys.argv[4])  # Take either the BEST or WORST values in terms of order.
    )
    # Perform a specific average values, then a full average values
    v, avg = p.get_specific_average_values()
    p.print_retrieved_configs_from_args(v, avg)
    v, avg = p.get_all_average_values()
    p.print_retrieved_configs_from_all(v, avg)
