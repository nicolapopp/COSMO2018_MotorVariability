#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Simulate & Plot a trajectory of the Inventory Control Problem
(see example_inventory.py)
"""

from pylab import *

import reachtask_optimize
from reachtask_optimize import ReachMovementTask,optimize

crtReachMovementTask=ReachMovementTask()

crtReachMovementTask.minimum_movement_length=-5
crtReachMovementTask.maximum_movement_length=5
crtReachMovementTask.c=10
crtReachMovementTask.target_position = 40
crtReachMovementTask.number_of_iterations = 30

[movement_sys,dpsolv,u]=optimize(crtReachMovementTask)

# Number of instants to simulate
N = 400

mov_law = movement_sys.perturb_laws[0]

xvals = np.zeros(N+1)
uvals = np.zeros(N)
np.random.seed(0)
demand = mov_law.rvs(N)

x_grid = dpsolv.state_grid[0]
def order_pol(x):
    'ordering policy, for a given stock level `x`'
    x_ind = np.where(x_grid==x)[0]
    return u[x_ind, 0]

# Simpler alternative:
order_pol = dpsolv.interp_on_state(u[...,0])

x0 = 0
for k in range(N):
    uvals[k] = order_pol(xvals[k])
    xvals[k+1] = movement_sys.dyn(xvals[k], uvals[k], demand[k])[0]

### Plot
t = arange(N)
t1 = arange(N+1)

figure(0, figsize=(8,3))

plot(t1, xvals, '-d', label='$x_k$')
plot(t, uvals, '-x', label='$u_k$')
plot(t, demand, '-+', label='$w_k$')
hlines(0, 0, N, colors='gray')

title('Simulated trajectory of the Invertory Control (c={:.1f})'.format(crtReachMovementTask.c))
xlabel('time $k$')
ylabel('Number of items')
xlim(-0.5, N+5.5)
ylim(-10,200)
legend(loc='upper right')

tight_layout()
show()