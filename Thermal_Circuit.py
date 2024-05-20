"""
Change: -ACH (air chainges per hour)
        -Tm (mean Temperature for radiation)
        -To, Ti_sp
Need to save TC as .csv like in inputs?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem

from Inputs import l
#from Inputs import Ti_sp_ss as Ti_sp
Sg = l**2           # m² surface area of the glass wall
Sc = Si = 5 * Sg    # m² surface area of concrete & insulation of the 5 walls

air         = {'Density': 1.2,                  # kg/m³
               'Specific heat': 1000}           # J/(kg·K)

concrete    = {'Conductivity': 1.400,            # W/(m·K)
               'Density': 2300.0,                # kg/m³
               'Specific heat': 880,             # J/(kg⋅K)
               'Width': 0.2,                     # m
               'Surface': Sc}                    # m²

insulation  = {'Conductivity': 0.027,            # W/(m·K)
              'Density': 55.0,                   # kg/m³
              'Specific heat': 1210,             # J/(kg⋅K)
              'Width': 0.08,                     # m
              'Surface': Si}               # m²

glass       = {'Conductivity': 1.4,              # W/(m·K)
               'Density': 2500,                  # kg/m³
               'Specific heat': 1210,            # J/(kg⋅K)
               'Width': 0.04,                    # m
               'Surface': Sg}                    # m²

pd.DataFrame(air, index=['Air'])
wall = pd.DataFrame.from_dict({'Layer_out': insulation,
                               'Layer_in': concrete,
                               'Glass': glass},
                              orient='index')

# radiative properties
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass

# from Inputs import wall_out
# ε_wLW = wall_out.loc[wall_out['ID'] == 'ws', 'ε_lw'].values[0]
# ε_gLW = wall_out.loc[wall_out['ID'] == 'window', 'ε_lw'].values[0]
# α_wSW = wall_out.loc[wall_out['ID'] == 'ws', 'α_i'].values[0]
# α_gSW = wall_out.loc[wall_out['ID'] == 'window', 'α_o'].values[0]
# τ_gSW = wall_out.loc[wall_out['ID'] == 'window', 'τ'].values[0]

σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant

#convection coefficents
h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)
# convection
Gw = h * wall['Surface'][0]     # wall
Gg = h * wall['Surface'][2]     # glass


# conduction
G_cd = wall['Conductivity'] / wall['Width'] * wall['Surface']
pd.DataFrame(G_cd, columns=['Conductance'])


# view factor wall-glass
Fwg = glass['Surface'] / concrete['Surface']


# long wave radiation
Tm = 293.15   # K, mean temp for radiative exchange

GLW1  = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) *   wall['Surface']['Layer_in']
GLW12 = 4 * σ * Tm**3 * Fwg                 *   wall['Surface']['Layer_in']
GLW2  = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) *   wall['Surface']['Glass']

GLW = 1 / (1 / GLW1 + 1 / GLW12 + 1 / GLW2) #conductance for radiative long-wave heat exchange between wall and glass window

## Advection

# ventilation flow rate
from Inputs import ACH
Va = l**3                   # m³, volume of air
Va_dot = ACH / 3600 * Va    # m³/s, air infiltration

# ventilation & advection
Gv = air['Density'] * air['Specific heat'] * Va_dot  #conductance for advection by ventilation/infiltration

# P-controler gain          
from Inputs import controller
if controller:  
    Kp = 1e3        # Kp -> ∞, almost perfect controller
else: Kp=0

## Conductances in series and/or parallel
# glass: convection outdoor & conduction
Ggs = float(1 / (1 / Gg.loc['h', 'out'] + 1 / (2 * G_cd['Glass'])))


## Thermal capacities
# Walls
C = wall['Density'] * wall['Specific heat'] * wall['Surface'] * wall['Width']
pd.DataFrame(C, columns=['Capacity'])
# Air
C['Air'] = air['Density'] * air['Specific heat'] * Va
pd.DataFrame(C, columns=['Capacity'])

## Matrices
# temperature nodes
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7']

# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11']

#incidence matrix
A = np.zeros([len(q), len(θ)])       # n° of branches X n° of nodes
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1    # branch 3: node 2 -> node 3
A[4, 3], A[4, 4] = -1, 1    # branch 4: node 3 -> node 4
A[5, 4], A[5, 5] = -1, 1    # branch 5: node 4 -> node 5
A[6, 4], A[6, 6] = -1, 1    # branch 6: node 4 -> node 6
A[7, 5], A[7, 6] = -1, 1    # branch 7: node 5 -> node 6
A[8, 7] = 1                 # branch 8: -> node 7
A[9, 5], A[9, 7] = 1, -1    # branch 9: node 5 -> node 7
A[10, 6] = 1                # branch 10: -> node 6
A[11, 6] = 1                # branch 11: -> node 6

pd.DataFrame(A, index=q, columns=θ)

#conductance matrix
G = np.array(np.hstack(
    [Gw['out'],
     2 * G_cd['Layer_out'], 2 * G_cd['Layer_out'],
     2 * G_cd['Layer_in'], 2 * G_cd['Layer_in'],
     GLW,
     Gw['in'],
     Gg['in'],
     Ggs,
     2 * G_cd['Glass'],
     Gv,
     Kp]))

# np.set_printoptions(precision=3, threshold=16, suppress=True)
# pd.set_option("display.precision", 1)
pd.DataFrame(G, index=q)


#capacity matrix
from Inputs import neglect_air_glass_capacity
if neglect_air_glass_capacity:
    C = np.array([0, C['Layer_out'], 0, C['Layer_in'], 0, 0,
                  0, 0])
else:
    C = np.array([0, C['Layer_out'], 0, C['Layer_in'], 0, 0,
                  C['Air'], C['Glass']])

# pd.set_option("display.precision", 3)
pd.DataFrame(C, index=θ)

#temperature source vector
b = pd.Series(['To', 0, 0, 0, 0, 0, 0, 0, 'To', 0, 'To', 'Ti_sp'],      
              index=q)

#heat flow source vector
f = pd.Series(['Φo', 0, 0, 0, 'Φi', 0, 'Qa', 'Φa'],     #Φo: solar radiation absorbed by the outdoor surface of the wall
              index=θ)                                  #Φi: solar radiation absorbed by the indoor surface of the wall
                                                        #Φa: solar radiation absorbed by the window
                                                        #Qa: heat source inside
#output vector
y = np.zeros(8)           # nodes 
y = np.ones(8)            # nodes (temperatures) of interest
pd.DataFrame(y, index=θ)

# thermal circuit
A = pd.DataFrame(A, index=q, columns=θ)
G = pd.Series(G, index=q)
C = pd.Series(C, index=θ)
b = pd.Series(b, index=q)
f = pd.Series(f, index=θ)
y = pd.Series(y, index=θ)

TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}

