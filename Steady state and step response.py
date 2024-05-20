import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem
from Thermal_Circuit import TC

"""steady state"""
from Inputs import To, Φo, Φi, Φa, Qa, θ0

Ti_sp=20

Φo, Φi, Φa = Φo.mean(), Φi.mean(), Φa.mean()
To=To.mean()
Qa=Qa.mean()

bss = bss_n = bss_d = np.zeros(12)        # temperature sources b for steady state
bss[[0, 8, 10]] = To                      # outdoor temperature
bss[[11]] = Ti_sp                         # indoor set-point temperature

fss = np.zeros(8)                         # flow-rate sources f for steady state
fss[[6]]= Qa
fss[[0]]= Φo
fss[[4]]= Φi
fss[[7]]= Φa

#state space
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

bT = np.array([To, To, To, Ti_sp])  # [To, To, To, Ti_sp]
fQ = np.array([Φo, Φi, Qa, Φa])     # [Φo, Φi, Qa, Φa]
uss = np.hstack([bT, fQ])           # input vector for state space

inv_As = pd.DataFrame(np.linalg.inv(As),
                      columns=As.index, index=As.index)

yss = (-Cs @ inv_As @ Bs + Ds) @ uss

#DAE equations
A = TC['A']
G = TC['G']
diag_G = pd.DataFrame(np.diag(G), index=G.index, columns=G.index)

θss = np.linalg.inv(A.T @ diag_G @ A) @ (A.T @ diag_G @ bss + fss)
qss = np.diag(G) @ (-A @ θss + bss)

print(f"inside Temperature(steady state): {np.around(yss['θ6'],2)} °C")
print(f'Temperature in nodes(DAE, θss):\n{np.around(θss,2)}')
print(f'Thermal load controller: q= {np.around(qss[11])} W')
print(f'Thermal loads: q= {np.around(qss)} W')


###############################################################################
"""step response"""
imposed_time_step = False
Δt = 56                        #s, imposed time step

# Eigenvalues analysis
λ = np.linalg.eig(As)[0]        # eigenvalues of matrix As

# time step
dtmax = 2 * min(-1. / λ)        # max time step for Euler explicit stability

if imposed_time_step:
    dt = Δt
else:
    dt = dm4bem.round_time(dtmax)

if dt < 10:
    raise ValueError("Time step is too small. Stopping the script.")
    
# settling time
t_settle = 4 * max(-1 / λ)

# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle/ 3600) * 3600

# dm4bem.print_rounded_time('dtmax', dtmax)
# dm4bem.print_rounded_time('dt', dt)
# dm4bem.print_rounded_time('t_settle', t_settle)
# dm4bem.print_rounded_time('duration', duration)


""" Create input_data_set"""
# time vector
n = int(np.floor(duration / dt))    # number of time steps

# DateTimeIndex starting at "00:00:00" with a time step of dt
from Inputs import start_date

time = pd.date_range(start=start_date,
                            periods=n, freq=f"{int(dt)}S")

To = To * np.ones(n)
Ti_sp = Ti_sp * np.ones(n)
Φa = Φa * np.ones(n)
Qa = Qa * np.ones(n)
Φo = Φo * np.ones(n)
Φi = Φi * np.ones(n)

data = {'To': To, 'Ti_sp': Ti_sp, 'Φo': Φo, 'Φi': Φi, 'Qa': Qa, 'Φa': Φa}
input_data_set = pd.DataFrame(data, index=time)

# inputs in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)


""""""
# Initial conditions
θ_exp = pd.DataFrame(index=u.index)     # empty df with index for explicit Euler
θ_imp = pd.DataFrame(index=u.index)     # empty df with index for implicit Euler

θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0

I = np.eye(As.shape[0])     # identity matrix, what for??
for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As)\
        @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k])

# outputs
y_exp = (Cs @ θ_exp.T + Ds @  u.T).T
y_imp = (Cs @ θ_imp.T + Ds @  u.T).T

# plot results
y = pd.concat([y_exp['θ6'], y_imp['θ6']], axis=1, keys=['Explicit', 'Implicit'])
# Flatten the two-level column labels into a single level

y.columns = y.columns.get_level_values(0)

ax = y.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Indoor temperature, $\\theta_i$ / °C')
ax.set_title(f'Time step: $dt$ = {dt:.0f} s; $dt_{{max}}$ = {dtmax:.0f} s')
plt.show()

