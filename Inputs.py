import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem
import matplotlib.dates as mdates

l = 3               # m length of the cubic room

# start_date = '1999-04-01 00:00:00'
# end_date   = '2000-09-30 00:00:00'

# start_date = '1999-10-01 00:00:00'
# end_date   = '2000-03-31 00:00:00'

start_date = '2000-01-01 00:00:00'
end_date   = '2000-03-01 00:00:00'

controller= True
neglect_air_glass_capacity = False
θ0 = 0                                  # initial temperatures

Ti_day, Ti_night = 20, 16

ACH = 0.5                               # 1/h, air changes per hour


# loading the EPW-file
filename = 'FRA_AR_Grenoble.074850_TMYx.epw'
[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data

weather.index = weather.index.map(lambda t: t.replace(year=2000))
weather = weather.loc[start_date:end_date]

To = weather['temp_air']

# β     = slope
# γ     = azimuth (0=south, 90=east)
# h0    = convection coefficent inside
# h1    = convection coefficent outside
# α_i   = short wave absortivity inside (white smooth surface)
# α_o   = short wave absortivity outside 
# ε_lw  = long wave emmisivity (concrete)
# τ     = short wave transmitance
columns = ['ID',  'Area', 'β', 'γ', 'h0', 'h1', 'α_i', 'α_o', 'ε_lw', 'τ', 'albedo']
wall_data = [
          ['wn',     9,   90,  180,   25,   8,   0.25,  0.30, 0.85,     0,   0.25],
          ['we',     9,   90,   90,   25,   8,   0.25,  0.30, 0.85,     0,   0.25],
          ['ww',     9,   90,  270,   25,   8,   0.25,  0.30, 0.85,     0,   0.25],
          ['window', 9,   90,    0,   25,   8,   0.00,  0.38, 0.90,  0.30,   0.25],
          ['roof',   9,    0,    0,   25,   8,   0.25,  0.30, 0.85,     0,   0.25],
          ['floor',  0,    0,    0,   25,   8,   0.25,  0.30, 0.85,     0,   0.25]
]
wall_out = pd.DataFrame(wall_data, columns=columns)


radiation = {}
absorbed_radiation = {}
Etot = pd.Series(0, index=weather.index)


#calculating the solar irradiance for every wall except the window
for wall_id in ['wn', 'we', 'ww', 'window', 'roof', 'floor']:
    wall_data = wall_out[wall_out['ID'] == wall_id]
    wall_orientation = {
        'slope': wall_data['β'].values[0],
        'azimuth': wall_data['γ'].values[0],
        'latitude': 45
    }

    #total solar irradiance walls
    rad_surf = dm4bem.sol_rad_tilt_surf(weather, wall_orientation, wall_data['albedo'].values[0])
    
    E = rad_surf.sum(axis=1)
    radiation[wall_id] = E

    Etot += E
    
    # absorbed solar irradiance walls
    absorbed_radiation[wall_id] = wall_data['α_o'].values[0] * wall_data['Area'].values[0] * E

# solar radiation absorbed by the indoor surface
window_data = wall_out[wall_out['ID']=='window']

Φo = absorbed_radiation['wn']+absorbed_radiation['we']+absorbed_radiation['ww']+absorbed_radiation['roof']
Φi = window_data['τ'].values[0] * window_data['Area'].values[0] * radiation['window']* 0.25  #wrong!!
Φa = absorbed_radiation['window']

#Schedules
# indoor air temperature set-point
Ti_sp = pd.Series(20, index=To.index)
Ti_sp = pd.Series(
    [Ti_day if 6 <= hour <= 22 else Ti_night for hour in To.index.hour],
    index=To.index)

# auxiliary (internal) sources
Qa = 368 * np.ones(weather.shape[0])

#Input data set
input_data_set = pd.DataFrame({'To': To, 'Ti_sp': Ti_sp,
                               'Φo': Φo, 'Φi': Φi, 'Qa': Qa, 'Φa': Φa,
                               'Etot': Etot})

""""daily"""
plt.figure(figsize=(10, 6))
plt.title('Absorbed Solar Radiation')

wall_ids= wall_labels=wall_out['ID'].tolist()

for i, wall_id in enumerate(wall_ids):
    plt.plot(absorbed_radiation[wall_id], label=wall_labels[i])
    
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))

plt.plot(Etot,label='Etot (W/m^2)')
plt.plot(Φo, label='Φo (outside walls)')
plt.plot(Φi, label='Φi (inside walls)')

plt.xlabel('Time')
plt.ylabel('Absorbed Solar Radiation (W)')
plt.legend()
plt.grid(True)
plt.show()
######################################
plt.figure(figsize=(10, 6))


plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))

plt.plot(To,label='Outside Temperatur')


plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

""""monthly"""
# plt.figure(figsize=(10, 6))

# daily_total_absorbed = input_data_set[['Etot']].resample('D').sum()
# monthly_total_absorbed = input_data_set[['Etot']].resample('M').sum()

# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# plt.plot(daily_total_absorbed.index, daily_total_absorbed['Etot']/1000, label='Etot_daily')
# plt.scatter(monthly_total_absorbed.index, monthly_total_absorbed['Etot']/(30*1000), label='Etot_monthly/30', color='red')
# plt.ylabel('Solar Radiation (kW/m^2)')
# plt.legend()
# plt.grid(True)


# plt.xticks(rotation=45)
# plt.show()
# #######################################
# plt.figure(figsize=(10, 6))

# daily_mean_To = input_data_set[['To']].resample('D').mean()
# monthly_mean_To = input_data_set[['To']].resample('M').mean()

# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# plt.plot(daily_mean_To.index, daily_mean_To['To'], label='To daily mean')
# plt.scatter(monthly_mean_To.index, monthly_mean_To['To'], label='To monthly mean', color='red')
# plt.ylabel('Temperature (°C)')
# plt.legend()
# plt.grid(True)


# plt.xticks(rotation=45)
# plt.show()
# """"""



