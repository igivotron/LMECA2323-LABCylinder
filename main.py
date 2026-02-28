import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp




ambiant_p = np.load("CylindersWake-20260225/ambient_p.npy")
ambiant_T = np.load("CylindersWake-20260225/ambient_T.npy")
inflow_delta_p = pd.read_csv("CylindersWake-20260225/inflow_delta_p.CSV", header=None)
inflow_delta_p = inflow_delta_p.iloc[:, 2].values
profile_pinf_p_bar = np.load("CylindersWake-20260225/profile_pinf_p_bar.npy")
profile_U_bar = np.load("CylindersWake-20260225/profile_U_bar.npy")
profile_U_p2_bar = np.load("CylindersWake-20260225/profile_U_p2_bar.npy")
profile_y = np.load("CylindersWake-20260225/profile_y.npy")
strain_calibration = pd.read_csv('data/strain_calibration.csv', delimiter=';', header=0)
mutlimeter = pd.read_csv('data/multimeter.csv', delimiter=';', header=None)
strainTension = mutlimeter.iloc[:, 0].values

R = 287.05
D = 50e-3
b = 425e-3
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
    tension_mean = np.mean(tension)
    tension_std = np.std(tension)
    N = len(tension)
    z = 1.96

    error_random = z * tension_std / np.sqrt(N)
    error_reading = 0.01 /2 

    a, b = strain_interpolation_coefs()
    force = (tension_mean - b) / a
    error_force = np.sqrt((error_random / a)**2 + (error_reading / a)**2) # Plus l'erreur de la calibration, mais on suppose que c'est négligeable
    return force, error_force

def get_Cd(DragForce, DragForce_error, rho, error_rho, U, error_U, D):
    Cd = DragForce / (0.5 * rho * U**2 * D * b)
    error_Cd = Cd * np.sqrt((error_rho / rho)**2 + (2 * error_U / U)**2 + (DragForce_error / DragForce)**2)
    return Cd, error_Cd

"""
Mise en forme des données
p[0], p_err[0] : inflow avant le cylindre
1 à 4: Différence de pression autour du point de stagnation (-10, -5, 5, 10)°
"""
datas = []
for i in range(1, 33):
    data = pd.read_csv(f'data/data{i}.csv', header=None)
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
DragForce, DragForce_error = tension_to_force(strainTension)
Cd, Cd_error = get_Cd(DragForce, DragForce_error, rho, error_rho, freestream_velocity, freestream_velocity_error, D)



# Affichage des résultats
print(f"Rho : {rho:.6f} ± {error_rho:.6f} kg/m³")
print(f"Freestream Pressure : {freestream_pressure:.6f} ± {freestram_pressure_error:.6f} Pa")
print(f"Freestream Velocity : {freestream_velocity:.6f} ± {freestream_velocity_error:.6f} m/s")
print(f"Kinematic Viscosity : {mu:.6e} ± {error_mu:.6e} m²/s")
print(f"Reynolds Number : {ReD:.6e} ± {error_ReD:.6e}")
print(f"Drag Force : {DragForce:.6f} ± {DragForce_error:.6f} N")
print(f"Drag Coefficient : {Cd:.6f} ± {Cd_error:.6f}")
