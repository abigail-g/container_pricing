#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:46:40 2024

@author: Group 11
"""

# %% Import required packages

# Import numpy package and names it np
import numpy as np

# imports pandas package and names it pd
import pandas as pd

# Load Pyomo Modelling Environment
from pyomo.environ import *

# %% Import dataset

file_name = 'WCG_DataSetV1.xlsx'
df = pd.read_excel(file_name, 'WCG Data')

# Converts df from dictionary to numpy
df = df.to_numpy()

# %% Given data

# Number of containers
numofCon = 300

# Cost per container
cost = 15000

# Week number (except holiday weeks)
week_num = df[8:58, 1]

# Lease options (1, 4, 8, 16-weeks)
lease = [1, 4, 8, 16]

# Price of leasing one container
price = pd.DataFrame({
    '1-Week': df[8:58, 4],
    '4-Weeks': df[8:58, 8],
    '8-Weeks': df[8:58, 12],
    '16-Weeks': df[8:58, 16]})

# Expected demand
demand = pd.DataFrame({
    '1-Week': df[8:58, 5],
    '4-Weeks': df[8:58, 9],
    '8-Weeks': df[8:58, 13],
    '16-Weeks': df[8:58, 17]})

# Number of returned containers from last year
ret = pd.DataFrame({
    '1-Week': np.append(df[8:9, 7], [0] * 15),
    '4-Weeks': np.append(df[8:12, 11], [0] * 12),
    '8-Weeks': np.append(df[8:16, 15], [0] * 8),
    '16-Weeks': df[8:24, 19]})

# Initial inventory
inventory = numofCon - ret.sum().sum()

# %% Decision variables & objective function

# Create model
model = ConcreteModel()

# Decision variables
model.x = Var(range(len(week_num)), range(len(lease)), domain = NonNegativeIntegers)

# Objective function
def obj_rule(model):
    return sum(7 * lease[j] * price.iloc[i, j] * model.x[i, j] for i in range(len(week_num)) for j in range(len(lease)))
    
model.obj = Objective(rule = obj_rule, sense = maximize)

# %% Inventory constraints

def inventory_func(i):
    if i == 0:
        return inventory + ret.iloc[i].sum()
    elif 1 <= i <= 3:
        return inventory_func(i - 1) - sum(model.x[i-1, j] for j in range(len(lease))) + model.x[i-1, 0] + ret.iloc[i].sum()
    elif 4 <= i <= 7:
        return inventory_func(i - 1) - sum(model.x[i-1, j] for j in range(len(lease))) + model.x[i-1, 0] + model.x[i-4, 1] + ret.iloc[i].sum()
    elif 8 <= i <= 15:
        return inventory_func(i - 1) - sum(model.x[i-1, j] for j in range(len(lease))) + model.x[i-1, 0] + model.x[i-4, 1] + model.x[i-8, 2] + ret.iloc[i].sum()
    else:
        return inventory_func(i - 1) - sum(model.x[i-1, j] for j in range(len(lease))) + model.x[i-1, 0] + model.x[i-4, 1] + model.x[i-8, 2] + model.x[i-16, 3]    

def inventory_rule(model, i):
    return sum(model.x[i, j] for j in range(len(lease))) <= inventory_func(i)

model.inventory = Constraint(range(len(week_num)), rule = inventory_rule)

# %% Demand constraints

def demand_rule(model, i, j):
    return model.x[i, j] <= demand.iloc[i, j]

model.demand = Constraint(range(len(week_num)), range(len(lease)), rule = demand_rule)

# %% Total constraints

def total_rule(i):
    return inventory_func(i) <= numofCon

# %% Solve the model
                  
solver = SolverFactory('glpk')
results = solver.solve(model)

# %% Output results

# Optimal revenue
if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print('Total revenue is:', model.obj())
    
    # store the revenue with initial capacity
    initial_rev = model.obj()

# Optimal values of decision variables
for j in range(len(lease)):
    for i in range(len(week_num)):
        print('For', lease[j], 'Week(s) lease,', model.x[i, j](), 'container(s) should be accepted in Week', week_num[i])

# Return on investment
roi = model.obj() / numofCon / cost
print('Return on investment is:', roi)

# Load factor: Leases per container per year
lease_per_con = sum(value(model.x[i, j]) for i in range(len(week_num)) for j in range(len(lease))) / numofCon
print('Leases per container per year is:', lease_per_con)

# Revenue per container per year
revenue_per_con = model.obj() / numofCon
print('Revenue per container per year is:', revenue_per_con)

# %% Answer checking

df_1_week = pd.DataFrame({
    'Demand (1-Week)': demand.loc[:,'1-Week'].to_numpy(),
    'Accepted (1-Week)': model.x[:, 0](),
    'Returned (1-Week)': np.append(ret.loc[0, '1-Week'], [model.x[i, 0]() for i in range(len(week_num) - 1)])})

df_4_weeks = pd.DataFrame({
    'Demand (4-Weeks)': demand.loc[:,'4-Weeks'].to_numpy(),
    'Accepted (4-Weeks)': model.x[:, 1](),
    'Returned (4-Weeks)': np.append(ret.loc[0:3, '4-Weeks'], [model.x[i, 1]() for i in range(len(week_num) - 4)])})

df_8_weeks = pd.DataFrame({
    'Demand (8-Weeks)': demand.loc[:,'8-Weeks'].to_numpy(),
    'Accepted (8-Weeks)': model.x[:, 2](),
    'Returned (8-Weeks)': np.append(ret.loc[0:7, '8-Weeks'], [model.x[i, 2]() for i in range(len(week_num) - 8)])})

df_16_weeks = pd.DataFrame({
    'Demand (16-Weeks)': demand.loc[:,'16-Weeks'].to_numpy(),
    'Accepted (16-Weeks)': model.x[:, 3](),
    'Returned (16-Weeks)': np.append(ret.loc[0:15, '16-Weeks'], [model.x[i, 3]() for i in range(len(week_num) - 16)])})

inventory = np.append(inventory, [value(inventory_func(i)) for i in range(len(week_num))])

df_result = pd.concat([pd.DataFrame({'Inventory': inventory[1:]}), df_1_week, df_4_weeks, df_8_weeks, df_16_weeks], axis = 1)

# %%
# ==================================================================================
# We now increase the number of containers by one to see how revenue will be changed
# ==================================================================================

# Number of containers
numofCon = 301

# Initial inventory
inventory = numofCon - ret.sum().sum()

# %% Decision variables & objective function

# Create model
model = ConcreteModel()

# Decision variables
model.x = Var(range(len(week_num)), range(len(lease)), domain = NonNegativeIntegers)

# Objective function
def obj_rule(model):
    return sum(7 * lease[j] * price.iloc[i, j] * model.x[i, j] for i in range(len(week_num)) for j in range(len(lease)))
    
model.obj = Objective(rule = obj_rule, sense = maximize)

# %% Inventory constraints

def inventory_func(i):
    if i == 0:
        return inventory + ret.iloc[i].sum()
    elif 1 <= i <= 3:
        return inventory_func(i - 1) - sum(model.x[i-1, j] for j in range(len(lease))) + model.x[i-1, 0] + ret.iloc[i].sum()
    elif 4 <= i <= 7:
        return inventory_func(i - 1) - sum(model.x[i-1, j] for j in range(len(lease))) + model.x[i-1, 0] + model.x[i-4, 1] + ret.iloc[i].sum()
    elif 8 <= i <= 15:
        return inventory_func(i - 1) - sum(model.x[i-1, j] for j in range(len(lease))) + model.x[i-1, 0] + model.x[i-4, 1] + model.x[i-8, 2] + ret.iloc[i].sum()
    else:
        return inventory_func(i - 1) - sum(model.x[i-1, j] for j in range(len(lease))) + model.x[i-1, 0] + model.x[i-4, 1] + model.x[i-8, 2] + model.x[i-16, 3]    

def inventory_rule(model, i):
    return sum(model.x[i, j] for j in range(len(lease))) <= inventory_func(i)

model.inventory = Constraint(range(len(week_num)), rule = inventory_rule)

# %% Demand constraints

def demand_rule(model, i, j):
    return model.x[i, j] <= demand.iloc[i, j]

model.demand = Constraint(range(len(week_num)), range(len(lease)), rule = demand_rule)

# %% Total constraints

def total_rule(i):
    return inventory_func(i) <= numofCon

# %% Solve the model
                  
solver = SolverFactory('glpk')
results = solver.solve(model)

# %% Output results

# Optimal revenue
if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print('Total revenue with an increase in the number of containers is:', model.obj())
    
    # Store the revenue with initial capacity
    increased_rev = model.obj()
    
    # Display shadow price
    print('The shadow price is:', increased_rev - initial_rev)