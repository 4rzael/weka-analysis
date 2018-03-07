#!/usr/bin/python3

from os.path import join

def command_line_to_config(line):
   config = {}
   config['reduced_error_pruning'] = '-R' in line
   config['unpruned'] = '-U' in line
   config['binary_splits'] = '-B' in line
   config['save_instance_data'] = '-L' in line
   config['subtree_raising'] = '-S' in line
   config['config_laplace'] = '-A' in line

   config['confidence_factor'] = float(line[line.index('-C') + 1]) if '-C' in line else None
   config['min_num_obj'] = int(line[line.index('-M') + 1]) if '-M' in line else None
   config['num_folds'] = int(line[line.index('-N') + 1]) if '-N' in line else None
   
   return config

def get_config(filename, folder):
   with open(join(folder, filename), 'r') as file:
       while True:
           line = file.readline()
           if line.find('Options:') >= 0:
               break
       
       return command_line_to_config(line.split()[1:])
