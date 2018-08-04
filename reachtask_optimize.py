#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Example of an Inventory Control problem for the documentation

This code closely matches the code chunks of example_inventory.rst

Plots are in seperate file (for embedded plot generation)
 * example_inventory_plot_policy.py
 * example_inventory_plot_simulation.py

Pierre Haessig — July 2013
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import scipy.stats as stats
import matplotlib.pylab as plt

### DP Solver ##################################################################
from stodynprog import DPSolver

import sys
try:
    import stodynprog
except ImportError:
    sys.path.append('..')
    import stodynprog

class ReachMovementTask():
  minimum_movement_length=-5
  maximum_movement_length=5
  c=10
  target_position = 40
  number_of_iterations = 30

from stodynprog import SysDescription
movement_sys = SysDescription((1,1,1), name='Movement system')

def optimize(crtReachMovementTask):
  minimum_movement_length=crtReachMovementTask.minimum_movement_length
  maximum_movement_length=crtReachMovementTask.maximum_movement_length
  c=crtReachMovementTask.c
  target_position = crtReachMovementTask.target_position
  number_of_iterations = crtReachMovementTask.number_of_iterations

  #functions defined here inside the optimizer
  def dyn_inv(x, u,w):
    'dynamical equation of the inventory stock `x`. Returns x(k+1).'
    return (x + u + w,)

  def op_cost(x,u,w):
     'energy cost'

     cost = abs(target_position-x)+c*u

     #cost = abs(20-x)+c*u
     # print(cost)
     return cost

  def admissible_movements(x):
         'interval of allowed orders movements(x_k)'
         U1 = (minimum_movement_length,maximum_movement_length)
         return (U1, ) # tuple, to support several controls
  # Attach it to the system description.

  # Attach the dynamical equation to the system description:
  movement_sys.dyn = dyn_inv

  mov_values = [-6,-4,-2,0,2,4,6]
  mov_proba  = [0.1,0.1,0.15,0.3,0.15,0.1,0.1]
  demand_law = stats.rv_discrete(values=(mov_values, mov_proba))
  demand_law = demand_law.freeze()

  demand_law.pmf([0, 3]) # Probality Mass Function
  demand_law.rvs(10) # Random Variables generation

  movement_sys.perturb_laws = [demand_law] # a list, to support several perturbations
  movement_sys.control_box = admissible_movements
  movement_sys.cost = op_cost

  print('Movement Control with (c={:.1f})'.format(c))

  dpsolv = DPSolver(movement_sys)

  # discretize the state space
  xmin, xmax = (0,100)
  N_x = xmax-xmin+1 # number of states
  dpsolv.discretize_state(xmin, xmax, N_x)

  # discretize the perturbation
  N_w = len(mov_values)
  dpsolv.discretize_perturb(mov_values[0], mov_values[-1], N_w)
  # control discretization step:
  dpsolv.control_steps=(1,) #

  #dpsolv.print_summary()

  ### Value iteration
  J_0 = np.zeros(N_x)
  # first iteration
  J,u = dpsolv.value_iteration(J_0)
  print(u[...,0])
  for i in xrange(number_of_iterations):
    J,u = dpsolv.value_iteration(J)

  # print(u[...,0])

  return [movement_sys,dpsolv,u]