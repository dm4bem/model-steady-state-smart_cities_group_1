from Inputs import θ0
from Inputs import input_data_set
from Thermal_Circuit import TC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem

#print(input_data_set)
explicit_Euler = True
imposed_time_step = False
Δt = 3600    # s, imposed time step 

# MODEL
# =====
# Thermal circuits
# TC = dm4bem.file2TC('TC.csv', name='', auto_number=False)

# State-space
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
# dm4bem.print_TC(TC)

λ = np.linalg.eig(As)[0]    # eigenvalues of matrix As
dtmax = 2 * min(-1. / λ)    # max time step for Euler explicit stability
dt = dm4bem.round_time(dtmax)

if imposed_time_step:
    dt = Δt
dm4bem.print_rounded_time('dt', dt)

# INPUT DATA SET

input_data_set = input_data_set.resample(
    str(dt) + 'S').interpolate(method='linear')
input_data_set.head()

# Input vector in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)
u.head()

# Initial conditions
θ0 = 20.0                   # °C, initial temperatures
θ = pd.DataFrame(index=u.index)
θ[As.columns] = θ0          # fill θ with initial valeus θ0


I = np.eye(As.shape[0])     # identity matrix

if explicit_Euler:
    for k in range(u.shape[0] - 1):
        θ.iloc[k + 1] = (I + dt * As) @ θ.iloc[k] + dt * Bs @ u.iloc[k]
else:
    for k in range(u.shape[0] - 1):
        θ.iloc[k + 1] = np.linalg.inv(
            I - dt * As) @ (θ.iloc[k] + dt * Bs @ u.iloc[k])
        
# outputs
y = (Cs @ θ.T + Ds @  u.T).T

Kp = TC['G']['q11']     # controller gain
S = 9                   # m², surface area of the toy house
q_HVAC = Kp * (u['q11'] - y['θ6']) / S  # W/m²
y['θ6']

data = pd.DataFrame({'To': input_data_set['To'],
                     'θi': y['θ6'],
                     'Etot': input_data_set['Etot'],
                     'q_HVAC': q_HVAC})

t = dt * np.arange(data.shape[0])   # time vector

fig, axs = plt.subplots(2, 1)
# plot outdoor and indoor temperature
axs[0].plot(t / 3600 / 24, data['To'], label='$θ_{outdoor}$')
axs[0].plot(t / 3600 / 24, input_data_set['Ti_sp'], label='$θ_{sp}$')
axs[0].plot(t / 3600 / 24, data['θi'], label='$θ_{indoor}$')
axs[0].set(ylabel='Temperatures, $θ$ / °C',
           title='Simulation for weather')
axs[0].legend(loc='upper right')

# plot total solar radiation and HVAC heat flow
axs[1].plot(t / 3600 / 24, data['Etot'], label='$E_{tot}$')
axs[1].plot(t / 3600 / 24, q_HVAC, label='$q_{HVAC}$')
axs[1].set(xlabel='Time, $t$ / day',
           ylabel='Heat flows, $q$ / (W·m⁻²)')
axs[1].legend(loc='upper right')

fig.tight_layout()

q_HVAC_ignored = q_HVAC.iloc[20:]
total_q_HVAC = np.abs(q_HVAC_ignored).sum() * dt * S   # total heat flow in Joules (W·s)
total_q_HVAC_kWh = total_q_HVAC / 3600 / 1000  # convert to kWh

print(f'Total HVAC heat flow: {total_q_HVAC_kWh:.2f} kWh')
