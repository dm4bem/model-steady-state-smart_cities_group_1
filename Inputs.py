"""
Why is latitude important? Shouldn't that come from the weather data?'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem

start_date = '02-01 01:00:00'
end_date = '02-01 22:00:00'

start_date = '2000-' + start_date
end_date = '2000-' + end_date
print(f'{start_date} \tstart date')
print(f'{end_date} \tend date')

# loading the EPW-file
filename = 'FRA_AR_Grenoble.074850_TMYx.epw'
[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data

weather.index = weather.index.map(lambda t: t.replace(year=2000))
weather = weather.loc[start_date:end_date]

To = weather['temp_air']

wall_out = pd.read_csv('walls_out.csv')

total_wall_radiation = {}
absorbed_wall_radiation = {}

#calculating the solar irradiance for every wall except the window
for wall_id in ['we', 'ws', 'ww', 'roof']:
    wall_data = wall_out[wall_out['ID'] == wall_id]
    wall_orientation = {
        'slope': wall_data['β'].values[0],
        'azimuth': wall_data['γ'].values[0],
        'latitude': 45
    }

    #total solar irradiance
    rad_surf = dm4bem.sol_rad_tilt_surf(weather, wall_orientation, wall_data['albedo'].values[0])
    Etot = rad_surf.sum(axis=1)
    total_wall_radiation[wall_id] = Etot

    # absorbed solar irradiance
    Φo = wall_data['α1'].values[0] * wall_data['Area'].values[0] * Etot
    absorbed_wall_radiation[wall_id] = Φo

""""""""
# window glass properties
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass
S_g = 9         # m², surface area of glass

# total solar irradiance window
window = wall_out[wall_out['ID'] == 'wn']

surface_orientation = {'slope': window['β'].values[0],
                       'azimuth': window['γ'].values[0],
                       'latitude': 45}

rad_surf = dm4bem.sol_rad_tilt_surf(
    weather, surface_orientation, window['albedo'].values[0])

Etot_window = rad_surf.sum(axis=1)

# solar radiation absorbed by the indoor surface of the wall
window_data = wall_out[wall_out['ID']=='wn']
Φi = τ_gSW * window_data['α0'].values[0] * S_g * Etot

# solar radiation absorbed by the glass
Φa = α_gSW * S_g * Etot_window

""""""
#Plot der totalen und absorbierten Wandstrahlung für jede Wand
plt.figure(figsize=(12, 8))

# Plot der totalen Wandstrahlung
plt.subplot(2, 1, 1)
for wall_id, Etot in total_wall_radiation.items():
    plt.plot(Etot.index, Etot.values, label=f'{wall_id} - Total Radiation')
plt.plot(Etot_window.index, Etot_window.values, label="Window - Total Radiation")
plt.xlabel('Time')
plt.ylabel('Total Radiation (W/m²)')
plt.title('Total Solar Radiation for Each Wall')
plt.legend()

# Plot der absorbierten Wandstrahlung
plt.subplot(2, 1, 2)
for wall_id, Φo in absorbed_wall_radiation.items():
    plt.plot(Φo.index, Φo.values, label=f'{wall_id} - Absorbed Radiation')
plt.plot(Φi.index, Φi, label="Φi - Absorbed Radiation inside")
plt.plot(Φa.index, Φa, label="Φa - Absorbed Radiation Window")

plt.xlabel('Time')
plt.ylabel('Absorbed Radiation (W)')
plt.title('Absorbed Solar Radiation for Each Wall')
plt.legend()

plt.tight_layout()
plt.show()
""""""
#Schedules

# indoor air temperature set-point
Ti_sp = pd.Series(20, index=To.index)

Ti_day, Ti_night = 20, 16
Ti_sp = pd.Series(
    [Ti_day if 6 <= hour <= 22 else Ti_night for hour in To.index.hour],
    index=To.index)

# auxiliary (internal) sources
Qa = 0 * np.ones(weather.shape[0])

# Input data set
input_data_set = pd.DataFrame({'To': To, 'Ti_sp': Ti_sp,
                               'Φo': Φo, 'Φi': Φi, 'Qa': Qa, 'Φa': Φa,
                               'Etot': Etot})

input_data_set.to_csv('input_data_set.csv')