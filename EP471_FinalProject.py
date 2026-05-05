#This code is authored by Henry Johnson 05/05/2026
#Assignment: Final Project

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, lil_matrix
from scipy.sparse.linalg import eigs

# parameters
N = 300
L = 1.0
dx = L/(N-1)

u = [1.0, -1.0] #Change these values to remove the symmetry of the problem
wp = [1.0, 1.0]

# derivative matrix (central diff)
main = np.zeros(N)
upper = np.ones(N-1)/(2*dx)
lower = -np.ones(N-1)/(2*dx)

D = diags([lower, main, upper], offsets=[-1, 0, 1], format='lil')

# second derivative matrix (central diff)
main = -2*np.ones(N)/dx**2
upper = np.ones(N-1)/dx**2
lower = np.ones(N-1)/dx**2

D2 = diags([lower, main, upper], [-1,0,1], format='lil')

I = eye(N,format='lil')

def build_AB():
    size=5*N
    A=lil_matrix((size,size),dtype=complex)
    B=lil_matrix((size,size),dtype=complex)
    
    def idx(block):
        return slice(block*N,(block+1)*N)
    
    phi,g1,h1,g2,h2 = 0,1,2,3,4
    
    # A matrix 
    
    # D2 * phi - wp^2 (h1 + h2) = 0
    A[idx(phi), idx(phi)] = D2
    A[idx(phi), idx(h1)] = -wp[0]**2 * I
    A[idx(phi), idx(h2)] = -wp[1]**2 * I
    
    #u1 * D * g1 - D2*phi = i omega g1
    A[idx(g1),idx(g1)]=u[0]*D
    A[idx(g1),idx(phi)]=-D2
    
    #u1*D*h1-g1=i * omega * h1
    A[idx(h1),idx(h1)]=u[0]*D
    A[idx(h1),idx(g1)]=-I
    
    #u2 * D * g2 - D2 * phi = i omega g2
    A[idx(g2),idx(g2)]=u[1]*D
    A[idx(g2),idx(phi)]=-D2
    
    #u2*D*h2-g2=i * omega * h2
    A[idx(h2),idx(h2)] = u[1]*D
    A[idx(h2),idx(g2)]=-I
    
    # B matrix
    
    B[idx(g1),idx(g1)]=I
    B[idx(h1),idx(h1)]=I
    B[idx(g2),idx(g2)]=I
    B[idx(h2),idx(h2)]=I
    
    return A.tocsr(), B.tocsr()

def apply_bc(A,B):
    A=A.tolil()
    B=B.tolil()
    
    # phi(0) = 0
    A[0,:] = 0 
    A[0,0] = 1
    B[0,:] = 0
    
    # phi(L) = 0
    A[N-1,:] = 0
    A[N-1,N-1] = 1 
    B[N-1,:] = 0
    
    return A.tocsr(), B.tocsr()

A, B = build_AB()
A, B = apply_bc(A, B)

# Sparsity patterns of A and B

plt.figure(figsize=(6,6))
plt.spy(A,markersize=1)
plt.title("Sparsity pattern of A for N=300")
plt.show()

plt.figure(figsize=(6,6))
plt.spy(B,markersize=1)
plt.title("Sparsity pattern of B for N=300")
plt.show()

np.random.seed(0)                               # makes every run solution have the same value
v0=np.ones(5*N,dtype=complex)

vals, vecs = eigs(A, M=B, k=6, sigma=0,v0=v0)  # changing k changes the number of generated eigenvalues
                                                # changing sigma changes where the eigenvalues are centered at
omega = -1j * vals
idx=np.argsort(np.imag(omega))[::-1]
omega=omega[idx]
vecs=vecs[:,idx]
print("Eigenvalues ω: ",omega)

mode = 0                                       # selects which mode we look at
X = vecs[:,mode]

#stores desired values from eigenfunction calculation into the vectors
phi = X[0*N:1*N]
g1 = X[1*N:2*N]
h1 = X[2*N:3*N]
g2 = X[3*N:4*N]
h2 = X[4*N:5*N]

# normalizes the functions
def normalize(v):
    return v / np.max(np.abs(v)) if np.max(np.abs(v)) > 0 else v

phi = normalize(phi)
g1 = normalize(g1)
h1 = normalize(h1)
g2 = normalize(g2)
h2 = normalize(h2)

x=np.linspace(0,L,N)

def smooth(v):
    return .5*(v[:-1] + v[1:])

x_mid = .5*(x[:-1] + x[1:])

#plots the real and imaginary parts of the electrostatic potential
plt.figure(figsize=(7,4))
plt.plot(x,np.real(phi),label="Re(φ)")
plt.plot(x,np.imag(phi),label="Im(φ)")
plt.title(f"Electrostatic potential (mode {mode})")
plt.xlabel("x")
plt.legend()
plt.grid()
plt.show()

#Plots the h fields (not discussed in report, just a diagnostic)
plt.figure(figsize=(7,4))
plt.plot(x_mid,np.real(smooth(h1)),label="Re(h1)")
plt.plot(x_mid,np.imag(smooth(h2)),label="Re(h2)")
plt.title("Species response (h fields)")
plt.xlabel("x")
plt.legend()
plt.grid()
plt.show()

#Plots the g fields (not discussed in report, just a diagnostic)
plt.figure(figsize=(8,5))
plt.plot(x_mid,np.real(smooth(g2)),label="g1")
plt.plot(x_mid,np.imag(smooth(g2)),label="g2")
plt.title("Species response (g fields)")
plt.xlabel("x")
plt.legend()
plt.grid()
plt.show()