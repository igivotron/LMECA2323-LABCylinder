import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp




ambiant_p = np.load("CylindersWake-20260225/ambient_p.npy")
ambiant_T = np.load("CylindersWake-20260225/ambient_T.npy")
inflow_delta_p = pd.read_csv("CylindersWake-20260225/inflow_delta_p.CSV")
inflow_delta_p = inflow_delta_p.iloc[:, 2].values
profile_pinf_p_bar = np.load("CylindersWake-20260225/profile_pinf_p_bar.npy")
profile_U_bar = np.load("CylindersWake-20260225/profile_U_bar.npy")
profile_U_p2_bar = np.load("CylindersWake-20260225/profile_U_p2_bar.npy")
profile_y = np.load("CylindersWake-20260225/profile_y.npy")
strain_calibration = pd.read_csv('data/strain_calibration.csv', delimiter=';')
mutlimeter = pd.read_csv('data/multimeter.csv', delimiter=';')

R = 287.05
D = 50e-3
g = 9.81

"""
profile_pinf_p_bar: Profile de pression en fonction de y derrière le cylindre
profile_U_bar: Profile de vitesse en fonction de y derrière le cylindre
profile_U_p2_bar: Profile de vitesse p2 (?) en fonction de y derrière le cylindre
"""

def get_pressure(pressures):
    p_mean = np.mean(pressures)
    p_std = np.std(pressures)
    N = len(pressures)

    error_device = 0.001 * p_mean + 0.03
    error_reading = 0.005
    error_random = 1.96 * p_std / np.sqrt(N)

    error = np.sqrt(error_device**2 + error_reading**2 + error_random**2)
    return p_mean, error

def interpolate_pressure(datas):
    x_inter = np.linspace(0, 31, 100).reshape(-1, 1)
    y_inter = sp.interpolate.interp1d(x, y, kind='cubic')(x_inter)
    return x_inter, y_inter

def get_rho():
    error_device = 4
    error_reading = (0.01e2) /2
    error_p = np.sqrt(error_device**2 + error_reading**2)
    error_T = 1/2

    rho = ambiant_p / (R * ambiant_T)
    error_rho = rho * np.sqrt((error_p / ambiant_p)**2 + (error_T / ambiant_T)**2)
    return rho, error_rho

def get_freestream_velocity(pressure, error_pressure, rho, error_rho):
    U = np.sqrt(2 * pressure / rho)
    U_error = U * np.sqrt((error_pressure / pressure)**2 + (error_rho / rho)**2)
    return U, U_error

def get_kinematic_viscosity(T):
    # Sutherland's formula for air viscosity
    error_T = 1/2
    mu = 1.458e-6 * T**(3/2) / (T + 110.4)

    dmu_dT = 1.458e-6  * ((1.5 * T**0.5 * (T + 110.4) - T**1.5)/ (T + 110.4)**2)
    error_mu = abs(dmu_dT) * error_T
    return mu, error_mu

def get_ReD(U, error_U, D, mu, error_mu, rho, error_rho):
    ReD = (rho * U * D) / mu
    ReD_error = ReD * np.sqrt((error_rho / rho)**2 + (error_U / U)**2 + (error_mu / mu)**2)
    return ReD, ReD_error

def strain_interpolation_coefs():
    """
    On interpole la calibration de la jauge de contrainte pour trouver les coefficients 
    d'une relation linéaire entre la tension et le poids.
    """
    mass = strain_calibration['mass'].values
    tension = strain_calibration['tension'].values
    weight = mass * g
    coeffs = np.polyfit(weight, tension, 1) 
    return coeffs

def tension_to_force(tension):
    a, b = strain_interpolation_coefs()
    return (tension - b) / a

### Mise en forme des données
N = 0
datas = []
for i in range(1, 33):
    data = pd.read_csv(f'data/data{i}.csv')
    pressures = data.iloc[:, 2].values
    p_mean, error = get_pressure(pressures)
    datas.append((p_mean, error))

datas = np.array(datas)
p = datas[:, 0]
p_err = datas[:, 1]

### Conditions ambiantes
rho, error_rho = get_rho()
freestream_pressure, freestram_pressure_error = p[0], p_err[0]
freestream_velocity, freestream_velocity_error = get_freestream_velocity(freestream_pressure, freestram_pressure_error, rho, error_rho)
mu, error_mu = get_kinematic_viscosity(ambiant_T)
ReD, error_ReD = get_ReD(freestream_velocity, freestream_velocity_error, D, mu, error_mu, rho, error_rho)
strainTension = (mutlimeter.iloc[:, 0].values).mean()
DragForce_measured = tension_to_force(strainTension)

print(f"Rho : {rho:.2f} ± {error_rho:.2f} kg/m³")
print(f"Freestream Pressure : {freestream_pressure:.2f} ± {freestram_pressure_error:.2f} Pa")
print(f"Freestream Velocity : {freestream_velocity:.2f} ± {freestream_velocity_error:.2f} m/s")
print(f"Kinematic Viscosity : {mu:.2e} ± {error_mu:.2e} m²/s")
print(f"Reynolds Number : {ReD:.2e} ± {error_ReD:.2e}")
print(f"strain tension : {strainTension:.2f} V")
print(f"Drag Force measured : {DragForce_measured:.2f} N")
