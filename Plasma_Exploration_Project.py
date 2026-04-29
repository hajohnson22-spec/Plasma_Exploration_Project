
"""
@author: Henry Johnson
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

def dispersion(vars,k,u0=1.0,wp=1.0,alpha=1.0):
    x,y=vars #real and imaginary parts of omega
    omega=x+1j*y #assembles omega
    
    omega_bar=omega/wp
    k_bar=k*u0/wp
    
    #epsilon=1e-6 #provides small regularization so that if k=\pm omega we don't divide by zero
    D=1-1/(omega_bar-k_bar)**2-alpha/(omega_bar+k_bar)**2 #dispersion relation between k and omega
    
    return [np.real(D),np.imag(D)] #returns the real and imaginary parts of the dispersion relation

def gamma_ana(k):
    term=k**2+1-np.sqrt(1+4*k**2)
    
    if term<0:
        return np.sqrt(-term)
    else:
        return 0
    
def omega_ana(k):
    term=k**2+1+np.sqrt(1+4*k**2)
    
    if term>0:
        return np.sqrt(term)
    else:
        return 0

k_vals=np.linspace(0.0,1.75,200) #sets an array of the range of k values

omega_r = [] #empty arrays for omega and gamma
gamma = []

guesses = [
    [0.0,0.1],  # unstable branch
    [1.0,0.0],  # upper real branch
    [-1.0,0.0], # lower real branch
    [0.0,-0.1]] # conjugate branch

guess_real=[1.0,0.0]

for k in k_vals:
    sqrt_term = np.sqrt(1 + 4*k**2)
    
    # Upper branch (always real)
    omega_sq_plus = k**2 + 1 + sqrt_term
    omega_r.append(np.sqrt(omega_sq_plus))
    
    # Lower branch (can be negative)
    omega_sq_minus = k**2 + 1 - sqrt_term
    
    if omega_sq_minus < 0:
        gamma.append(np.sqrt(-omega_sq_minus))
    else:
        gamma.append(0)

omega_r = np.array(omega_r) #turns the arrays into np arrays
gamma = np.array(gamma)
unstable = gamma > 1e-5
k_unstable = k_vals[unstable] #finds the k values where the instability is greater than 0 and 
        
maxgam=max(gamma) #finds the maximum instability value
kmin=k_unstable.min() #finds the maximum and minimum unstable k values
kmax=k_unstable.max()

print(f"Unstable band: k in [{kmin:.3f}, {kmax:.3f}]")

print("The maximum gamma value is ",maxgam,".")
    
gammas=np.array([gamma_ana(k) for k in k_vals]) #turns arrays into np arrays
omega_reals=np.array([omega_ana(k) for k in k_vals])

plt.figure(3) #plots
plt.plot(k_vals,gamma,label="Growth rate")
plt.axhline(0, linestyle='--')
plt.axvspan(kmin,kmax, alpha=.2, label = "Unstable band")
plt.xlabel("k (normalized)")
plt.ylabel("Growth rate (γ)")
plt.legend()
plt.grid()
plt.savefig("Instability_Range.png")
plt.show()

u0_vals = [0.5, 1.0, 2.0] #intializes various background velocity values
k_vals_2 = np.linspace(0.0, 3.5, 400) 

plt.figure(1)

for u0 in u0_vals:
    gamma_2 = []
    
    for k in k_vals_2: #this loop evaluates the analytical solution for gamma for each k value and appends that value to the list of gamma values
        k_eff = k * u0
        sqrt_term = np.sqrt(1 + 4*k_eff**2)
        
        omega_sq_minus = k_eff**2 + 1 - sqrt_term
        
        if omega_sq_minus < 0:
            gamma_2.append(np.sqrt(-omega_sq_minus))
        else:
            gamma_2.append(0)
    
    plt.plot(k_vals_2, gamma_2, label=f'u0={u0}') #plots each individual line
    
plt.xlabel("k (normalized)")
plt.ylabel("Growth rate (γ)")
plt.title("Beam Velocity Scan")
plt.legend()
plt.grid() 
plt.savefig("Beam_Velocity.png")  

k_vals_3 = np.linspace(0.0, 1.0, 400) #these values are chosen to show the general alpha values
alpha_vals=[1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4,5e-5] 

#k_vals_3 = np.linspace(0.0, 3.5, 400) #these values are chosen to explore when alpha is much less than 1
#alpha_vals=[10.0,5.0,2.0,0.99,.5,.2,.1,.05]

gamma_max=[]
k_max=[]

plt.figure(2)

for alpha in alpha_vals:
    gamma_alpha=[]
    omega_r_alpha=[]
    guess=[0.0,0.1]
    
    for i,k in enumerate(k_vals_3):
        sol=root(dispersion,guess,args=(k,1.0,1.0,alpha)) #finds the root of the dispersion relation for the specific k value
        
        if sol.success:
            x,y=sol.x   #finds real and imaginary parts of the root
            omega_sol=x+1j*y
            
            gamma_val=np.imag(omega_sol)    #appends the imaginary part to gamma
            omega_r_val=np.real(omega_sol)  #appends the real part to omega_r
            
            if gamma_val > 1e-6:            #filters data. if gamma is positive, we add it to our array and update our guess value, but if not we append zero and reset the guess value
                gamma_alpha.append(gamma_val)
                omega_r_alpha.append(omega_r_val)
                guess = sol.x 
            else:
                gamma_alpha.append(0)
                omega_r_alpha.append(omega_r_val)
        else:                               #if a solution cannot be found, we do not append a nonzero value to our arrays for that k
            gamma_alpha.append(0)
            omega_r_alpha.append(np.nan)
    
    gamma_alpha=np.array(gamma_alpha)   #finds the maximum value of gamma
    idx=np.argmax(gamma_alpha)          #finds the index of that max value
    gamma_max_val=gamma_alpha[idx]
    #k_max=k_vals_2[idx]
    
    gamma_max.append(gamma_max_val)     #appends that maximum value to a list of max values.
    #k_max.append(k_max)
    
    plt.plot(k_vals_3,gamma_alpha,label=f'α={alpha}')

for i in range(len(k_vals_3)-len(k_vals)):
    gamma=np.append(gamma,0)

#plt.plot(k_vals_3,gamma,label='α=1.0')
plt.xlabel("k (normalized)")
plt.ylabel("Growth rate (γ)")
plt.title("Alpha Scan")
plt.legend()
plt.grid() 
plt.savefig("Alpha.png")

log_alpha=np.log(alpha_vals)
log_gamma=np.log(gamma_max)
m,b=np.polyfit(log_alpha,log_gamma,1)
    
##################################################################### 
# PLOTS !!!
##################################################################### 

plt.figure(4)
plt.plot(k_vals,omega_reals,'--', label="Ananlytical", linewidth=2)
plt.xlabel("k")
plt.ylabel("ω_r")
plt.title("Real Frequency")
plt.savefig("Real_Frequency.png")
plt.grid()
plt.show()

plt.figure(5)
plt.loglog(alpha_vals, gamma_max, 'o-',label=f'slope={m:.2f}')
plt.xlabel("α = n₂/n₁")
plt.ylabel("γ_max")
plt.title("Maximum Growth Rate vs Density Ratio")
plt.legend()
plt.grid()
plt.show()

