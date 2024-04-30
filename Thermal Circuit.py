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

l = 3               # m length of the cubic room
Sg = l**2           # m² surface area of the glass wall
Sc = Si = 5 * Sg    # m² surface area of concrete & insulation of the 5 walls

air =           {'Density': 1.2,                 # kg/m³
                'Specific heat': 1000}           # J/(kg·K)


concrete    = {'Conductivity': 1.400,            # W/(m·K)
               'Density': 2300.0,                # kg/m³
               'Specific heat': 880,             # J/(kg⋅K)
               'Width': 0.2,                     # m
               'Surface': 5 * l**2}              # m²

insulation  = {'Conductivity': 0.027,            # W/(m·K)
              'Density': 55.0,                   # kg/m³
              'Specific heat': 1210,             # J/(kg⋅K)
              'Width': 0.08,                     # m
              'Surface': 5 * l**2}               # m²

glass       = {'Conductivity': 1.4,              # W/(m·K)
               'Density': 2500,                  # kg/m³
               'Specific heat': 1210,            # J/(kg⋅K)
               'Width': 0.04,                    # m
               'Surface': l**2}                  # m²

pd.DataFrame(air, index=['Air'])
wall = pd.DataFrame.from_dict({'Layer_out': concrete,
                               'Layer_in': insulation,
                               'Glass': glass},
                              orient='index')


# radiative properties
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass

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

##Why do we need this??
"""
T_int = 273.15 + np.array([0, 40])
coeff = np.round((4 * σ * T_int**3), 1)
print(f'For 0°C < (T/K - 273.15)°C < 40°C, 4σT³/[W/(m²·K)] ∈ {coeff}')

T_int = 273.15 + np.array([10, 30])
coeff = np.round((4 * σ * T_int**3), 1)
print(f'For 10°C < (T/K - 273.15)°C < 30°C, 4σT³/[W/(m²·K)] ∈ {coeff}')

T_int = 273.15 + 20
coeff = np.round((4 * σ * T_int**3), 1)
print(f'For (T/K - 273.15)°C = 20°C, 4σT³ = {4 * σ * T_int**3:.1f} W/(m²·K)')
"""

# long wave radiation
Tm = 20 + 273.15   # K, mean temp for radiative exchange

GLW1  = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) *   wall['Surface']['Layer_in']
GLW12 = 4 * σ * Tm**3 * Fwg                 *   wall['Surface']['Layer_in']
GLW2  = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) *   wall['Surface']['Glass']

GLW = 1 / (1 / GLW1 + 1 / GLW12 + 1 / GLW2) #conductance for radiative long-wave heat exchange between wall and glass window

## Advection

# ventilation flow rate
Va = l**3                   # m³, volume of air
ACH = 1                     # 1/h, air changes per hour
Va_dot = ACH / 3600 * Va    # m³/s, air infiltration

# ventilation & advection
Gv = air['Density'] * air['Specific heat'] * Va_dot     #conductance for advection by ventilation/infiltration

# P-controler gain
# Kp = 1e4          # almost perfect controller Kp -> ∞
# Kp = 1e-3         # no controller Kp -> 0
Kp = 0              #to be set later 

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
neglect_air_glass = False

if neglect_air_glass:
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
                                                        #Qa: heat source inside     
                                                        #Φa: solar radiation absorbed by the glass
#output vector
y = np.zeros(8)         # nodes
y[[6]] = 1              # nodes (temperatures) of interest
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

# TC['G']['q11'] = 1e3  # Kp -> ∞, almost perfect controller
TC['G']['q11'] = 0      # Kp -> 0, no controller (free-floating)

print(TC)

