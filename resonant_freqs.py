import RigolWFM.wfm as rigol
import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass

# Rigol DS1054Z oscilloscope parameters
@dataclass
class scope_params:
    n_divisions: int = 12
    trigger_position: int = 1
    time_per_division: int = 0.02
    sample_rate: int = 5e6
    name: str = 'DS1000Z'

def decaying_exp(t, a, omegar, omegai, phi):
    return a * np.exp(-omegai * t) * np.sin(omegar * t + phi)

def fit_signal(vol, n_rep, omega_Rayleigh):
    
    trigger_position = scope_params.trigger_position
    n_divisions = scope_params.n_divisions
    scope = scope_params.name

    nV = len(vol)
    fit_params = np.zeros((nV, n_rep, 4))
    for i in range(nV):
        for j in range(n_rep):
            
            # load data from wfm file
            infile = f'wfm-files/v{vol[i]}_rep{j}.wfm'
            scope_data = rigol.Wfm.from_file(infile, scope)
            t = scope_data.channels[0].times # time
            V = scope_data.channels[0].volts # voltage
            
            eps = 10 # 1e-10 second time resolution
            
            # extract the impact signal from the entire oscilloscope measurement
            ts = (t[-1] - t[0]) * trigger_position / n_divisions
            dt = np.round(t[1] - t[0], eps)
            ts_ind = int(ts / dt)
            t_sig, V_sig, n_sig = save_signal(t, V, dt, ts_ind, omega_Rayleigh)
            
            # initial guess for fitting parameters
            if j == 0:
                params_0 = [V_sig[int(n_sig/3)], omega_Rayleigh, 50, -10]
            else:
                params_0 = fit_params[j-1, :]
    
            # fit a decaying exponential function to signal
            fit_params[i, j, :], pcov = curve_fit(decaying_exp, t_sig[int(n_sig/3):], V_sig[int(n_sig/3):], p0=params_0)

    return fit_params[:, :, 1:3]

def save_signal(t, V, dt, ts_ind, omega_Rayleigh):

    t_Rayleigh = 2 * np. pi / omega_Rayleigh
    ind_jump = int(t_Rayleigh / dt) # convert to jump in index
    V_p = V - np.mean(V)
    V_std = np.std(V)
    
    # find signal beginning
    i = ts_ind
    back_jump = int(ind_jump/10)
    while np.std(V_p[i-back_jump : i+back_jump]) > 1.2*V_std:
        i -= back_jump
        
    # find signal end
    j = ts_ind + ind_jump
    while np.std(V_p[j-ind_jump : j+ind_jump]) > 1.05*V_std:
        j += ind_jump

    # return time series of t,V and its length
    return t[i:j] - t[i], V_p[i:j], j-i

####### Main code ###########

# experiment droplet volumes (values are the volume in nanoliters divided by 10)
vol = np.arange(30, 110, 5)
nV = len(vol)
n_rep = 20 # number of repetitions for each droplet volume

alpha = 80 * np.pi / 180 # microphone-water static contact angle
R0 = (3 * vol/100 / (4 * np.pi)) ** (1/3) # droplet radius [mm]
R = R0 * ((2 + np.cos(alpha)) / np.sin(alpha)**3 * np.sin(alpha/2)**4) ** (-1/3) # droplet spherical cap radius [mm]

sigma = 0.072 # air-water surface tension [kg/s^2]
rho = 1000 # water density [kg/m^3]

omega_scaling = (sigma / (rho * (1e-3 * R)**3)) ** (1/2) # inverse of the inviscid scaling for time
omega_Rayleigh = 8**(1/2) * omega_scaling # first-mode resonant frequency of a free spherical droplet 

freqs = fit_signal(vol, n_rep, omega_Rayleigh)
np.savetxt("drop_data.csv", np.reshape(freqs, (nV * n_rep, 2)), delimiter=",")