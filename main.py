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

def get_Cd_straingauge(DragForce, DragForce_error, rho, error_rho, U, error_U, D):
    Cd = DragForce / (0.5 * rho * U**2 * D * b)
    error_Cd = Cd * np.sqrt((error_rho / rho)**2 + (2 * error_U / U)**2 + (DragForce_error / DragForce)**2)
    return Cd, error_Cd


def get_Cp_theta(p, error_p, rho, rho_error, U, U_error):
    Cptheta = p / (0.5 * rho * U**2)
    error_Cptheta = Cptheta * np.sqrt((error_p / p)**2 + (rho_error / rho)**2 + (2 * U_error / U)**2)

    return Cptheta, np.abs(error_Cptheta)

def get_D_add(rho, U_inf, U, up2, p, y):

    ym = y * 1e-3
    Y = ym / D

    term1 = (U/U_inf) * (1 - (U/U_inf))
    term2 = (up2 / U_inf**2)
    term3 = (p / (rho * U_inf**2))

    integrand = term1 - term2 + term3

    integral = np.trapezoid(integrand, Y)

    return rho * U_inf**2 * D * integral

def interpolate_Cp(coefs, angles):
    a = coefs[0]
    b = coefs[1]
    c = coefs[2]
    p_inter = a * angles**2 + b * angles + c
    return get_Cp_theta(p_inter, 0, rho, 0, freestream_velocity, 0)[0]

def get_p_interpolation_coefficients(angles, p):
    a, b, c = np.polyfit(angles[0:5], p[1:6], 2)
    return [a, b, c]

def get_max_pressure_angle(coeffs):
    a, b, c = coeffs
    return -b / (2 * a)

def integral_cptheta(angles, Cp):
    angles_rad = np.radians(angles)
    return np.trapezoid(Cp * np.cos(angles_rad), angles_rad)

def get_Cp_potential_flow(theta):
    return 1 - 4 * np.sin(np.radians(theta))**2


def clip_additional_data(U_inf, treshold_U=0.1, treshold_p=0.1):
    """
    On cut les données si U_inf est proche de U_bar et que profile_pinf_p_bar est proche de 0
    """
    index = np.where((np.abs(profile_U_bar - U_inf) < treshold_U) & (np.abs(profile_pinf_p_bar) < treshold_p))[0]
    low_bound = index[0]
    upper_bound = index[1]
    # for i in index: plt.axvline(profile_y[i], color='r', linestyle='--', label=f'Clipping at y={profile_y[i]:.2f} mm')

    mask = (profile_y >= profile_y[low_bound]) & (profile_y <= profile_y[upper_bound])
    y = profile_y[mask]
    U = profile_U_bar[mask]
    up2 = profile_U_p2_bar[mask]
    p = profile_pinf_p_bar[mask]


    # plt.plot(y, U, label='U')
    # plt.plot(y, up2, label='U_p2')
    # plt.plot(y, p, label='p_inf_p')
    # plt.xlabel('y (mm)')
    # plt.ylabel('Values')
    # plt.title('Profiles after clipping')
    # plt.legend()
    # plt.grid()
    # plt.show()

    return U, up2, p, y

"""
Mise en forme des données
p[0], p_err[0] : inflow avant le cylindre
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
# Suppression de la double mesure
p = np.concatenate((p[0:26], p[27:]))
p_err = np.concatenate((p_err[0:26], p_err[27:]))
N = len(p)

### Conditions ambiantes
rho, error_rho = get_rho()
freestream_pressure, freestram_pressure_error = p[0], p_err[0]
freestream_velocity, freestream_velocity_error = get_freestream_velocity(freestream_pressure, freestram_pressure_error, rho, error_rho)
mu, error_mu = get_kinematic_viscosity(ambiant_T)
ReD, error_ReD = get_ReD(freestream_velocity, freestream_velocity_error, D, mu, error_mu, rho, error_rho)
DragForce_straingauge, DragForce_straingauge_error = tension_to_force(strainTension)
Cd_straingauge, Cd_straingauge_error = get_Cd_straingauge(DragForce_straingauge, DragForce_straingauge_error, rho, error_rho, freestream_velocity, freestream_velocity_error, D)

### Mesure stagnation point
angles1 = np.arange(-10, 90, 5)
angles2 = np.arange(90, 190, 10)
angles = np.concatenate((angles1, angles2))

Cp = []
Cp_error = []
for i in range(1, N):
    p_i, error_p_i = p[i], p_err[i]
    Cp_i, error_Cp_i = get_Cp_theta(p_i, error_p_i, rho, error_rho, freestream_velocity, freestream_velocity_error)
    Cp.append(Cp_i)
    Cp_error.append(error_Cp_i)

# interpolation du Cp pour trouver le point de stagnation
coefs = get_p_interpolation_coefficients(angles, p)
angles_inter = np.linspace(-10, 10, 100)
Cp_inter = interpolate_Cp(coefs, angles_inter)
stagnation_angle = get_max_pressure_angle(coefs)
Cp_total = integral_cptheta(angles, Cp)

angles_potential_flow = np.linspace(0, 90, 100)
Cp_potential_flow = get_Cp_potential_flow(angles_potential_flow)
angles_potential_flow += stagnation_angle

# plt.errorbar(angles, Cp, yerr=Cp_error, fmt='o', label='Data with error bars')
# plt.plot(angles_inter, Cp_inter, label='Interpolation')
# plt.plot(angles_potential_flow, Cp_potential_flow, label='Potential Flow')
# plt.axvline(stagnation_angle, color='g', linestyle='--', label=f'Stagnation Point at {stagnation_angle:.2f}°')

# # plt.axhline(1, color='r', linestyle='--', label='Cp = 1')
# plt.xlabel('Angle (degrees)')
# plt.ylabel('Cp')
# plt.title('Cp vs Angle with Interpolation')
# plt.legend()
# plt.grid()
# plt.show()



### Additional Analysis
U, up2, p, y = clip_additional_data(freestream_velocity)
Drag_add = get_D_add(rho, freestream_velocity, U, up2, p, y)
Cd_additional = Drag_add / (0.5 * rho * freestream_velocity**2 * D)



# Affichage des résultats
print(f"Rho : {rho:.6f} ± {error_rho:.6f} kg/m³")
print(f"Freestream Pressure : {freestream_pressure:.6f} ± {freestram_pressure_error:.6f} Pa")
print(f"Freestream Velocity : {freestream_velocity:.6f} ± {freestream_velocity_error:.6f} m/s")
print(f"Kinematic Viscosity : {mu:.6e} ± {error_mu:.6e} m²/s")
print(f"Reynolds Number : {ReD:.6e} ± {error_ReD:.6e}")
print(f"Drag Force from the Strain gauge: {DragForce_straingauge:.6f} ± {DragForce_straingauge_error:.6f} N")
print(f"Drag Coefficient from the strain gauge: {Cd_straingauge:.6f} ± {Cd_straingauge_error:.6f}")
print(f"Cd from additional analysis: {Cd_additional:.6f}")
print(f"Stagnation Point Angle : {stagnation_angle:.2f}°")
print(f"Pressure Coefficient (Cp theta) Integral : {Cp_total:.6f}")
