# -*- coding: utf-8 -*-
"""


#PINN Burgers
Code modified from https://github.com/314arhaam/burger-pinn

Supporting implementations for the paper 
Optimal time sampling in physics-informed neural networks
by
Gabriel Turinici
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
#
from scipy.stats import qmc # upgrade scipy for LHS !pip install scipy --upgrade

tf.keras.backend.set_floatx("float64")

### generating data

# number of boundary and initial data points
# value `Nd` in the reference paper:
# Nd = number_of_ic_points + number_of_bc1_points + number_of_bc1_points
number_of_ic_points = 50
number_of_bc1_points = 25
number_of_bc2_points = 25

# Latin Hypercube Sampling (LHS) engine ; to sample random points in domain,
# boundary and initial boundary
engine = qmc.LatinHypercube(d=1)

# temporal data points
t_d = engine.random(n=number_of_bc1_points + number_of_bc2_points)
temp = np.zeros([number_of_ic_points, 1]) # for IC ; t = 0
t_d = np.append(temp, t_d, axis=0)
# spatial data points
x_d = engine.random(n=number_of_ic_points)
x_d = 2 * (x_d - 0.5)
temp1 = -1 * np.ones([number_of_bc1_points, 1]) # for BC1 ; x = -1
temp2 = +1 * np.ones([number_of_bc2_points, 1]) # for BC2 ; x = +1
x_d = np.append(x_d, temp1, axis=0)
x_d = np.append(x_d, temp2, axis=0)

# view randomly sampled boundary and initial points
plt.scatter(t_d, x_d, marker="x", c="k")
plt.xlabel("t")
plt.ylabel("x")
plt.title("Data points (BCs & IC)")

# output values for data points (boundary and initial)
y_d = np.zeros(x_d.shape)

def initial_condition_function(x):
  return -np.sin(np.pi * x)

@tf.function
def initial_condition_function(x):
  return -tf.math.sin(np.pi * x)


# for initial condition: IC = -sin(pi*x)
y_d[ : number_of_ic_points] = -np.sin(np.pi * x_d[:number_of_ic_points])

# all boundary conditions are set to zero
y_d[number_of_ic_points : number_of_bc1_points + number_of_ic_points] = 0
y_d[number_of_bc1_points + number_of_ic_points : number_of_bc1_points + number_of_ic_points + number_of_bc2_points] = 0

# number of collocation points
Nc = 10000

# LHS for collocation points
engine = qmc.LatinHypercube(d=2)
data = engine.random(n=Nc)
# set x values between -1. and +1.
data[:, 1] = 2*(data[:, 1]-0.5)
# change names
t_c = np.expand_dims(data[:, 0], axis=1)
x_c = np.expand_dims(data[:, 1], axis=1)

# convert all data and collocation points to tf.Tensor
x_d, t_d, y_d, x_c, t_c = map(tf.convert_to_tensor, [x_d, t_d, y_d, x_c, t_c])
print(t_d.shape,x_d.shape,t_c.shape,x_c.shape,y_d.shape)

input_dim=1
Tfinal=1.0

def sample_truncated(size=(number_of_bc1_points,input_dim),rate=0.0):
  """
  Utility function : sampling from truncated exponential.
  This distribution has support in [0,T] and a density proportional to exp(-rate*x) in this domain (zero outside).

  inputs : size is the shape of the output
  output : samples from the distribution

  Computation formula : if U is [0,1] uniform then Y= -ln(1-U+U*exp(-rate*T))/rate
  follows the target distribution.
  When rate*T is small enough use use a first order approximation and output U*T
  otherwise division by the rate would be unstable
  """
  U_samples=np.random.uniform(size=size)
  if(np.abs(rate*Tfinal)<1.e-6):
    return U_samples*Tfinal
  else:
    return -np.log(1-U_samples+U_samples*np.exp(-rate*Tfinal))/rate

def quantiles_truncated(size=(number_of_bc1_points,input_dim),rate=0.0):
  """
  Utility function : return 'quantiles' of the truncated exponential.
  This distribution has support in [0,T] and a density proportional to exp(-rate*x) in this domain (zero outside).

  inputs : size is the shape of the output
  output : samples from the distribution

  Computation formula : if U is [0,1] uniform then Y= -ln(1-U+U*exp(-rate*T))/rate
  follows the target distribution.
  When rate*T is small enough use use a first order approximation and output U*T
  otherwise division by the rate would be unstable
  """
  U_samples=np.linspace(0,1,size[0],endpoint=True).reshape(size)
  if(np.abs(rate*Tfinal)<1.e-6):
    return U_samples*Tfinal
  else:
    return -np.log(1-U_samples+U_samples*np.exp(-rate*Tfinal))/rate

t_d.shape,x_d.shape,x_c.shape,t_c.shape,y_d.shape
# y_d : IC + boundary condition + BC2: 50 +25 +25
# take a grid with 50 * 25 in time x space
use_grid=True
if(use_grid):
  lambda_sample=1.0
  grid_x = np.linspace(-1,1,number_of_ic_points,endpoint=True)
  grid_t = quantiles_truncated(rate=lambda_sample)[:,0]
  print(grid_x.shape,grid_t.shape)
  #data is in this order: first number_of_ic_points correspond to initial data x values i.e. t_d[0:number_of_ic_points]=0
  # then bc1 then bc2 values
  x_d = np.hstack((grid_x,-np.ones(number_of_bc1_points),np.ones(number_of_bc1_points)))
  t_d= np.hstack((np.zeros(number_of_ic_points),grid_t,grid_t))
  y_d= x_d*0
  # for initial condition: IC = -sin(pi*x)
  y_d[ : number_of_ic_points] = -np.sin(np.pi * x_d[:number_of_ic_points])
  # all boundary conditions are set to zero
  y_d[number_of_ic_points : number_of_bc1_points + number_of_ic_points] = 0
  y_d[number_of_bc1_points + number_of_ic_points : number_of_bc1_points + number_of_ic_points + number_of_bc2_points] = 0
  t_c,x_c= np.meshgrid(grid_t,grid_x)
  t_c=t_c.flatten()
  x_c= x_c.flatten()
  # view sampled boundary and initial points
  plt.scatter(t_d, x_d, marker="x", c="k")
  plt.xlabel("t")
  plt.ylabel("x")
  plt.title("Data points (BCs & IC)")
  x_d, t_d, y_d, x_c, t_c = map(tf.convert_to_tensor, [v[:,None] for v in [x_d, t_d, y_d, x_c, t_c]])
  print(t_d.shape,x_d.shape,t_c.shape,x_c.shape,y_d.shape)

### model design
#
no_hidden_layers=9
neuron_per_layer = 20
# activation function for all hidden layers
actfn = "tanh"
random_seed=123

# input layer
input_layer = tf.keras.layers.Input(shape=(2,))
layer_result=input_layer
for _ in range(no_hidden_layers):
  layer_result = tf.keras.layers.Dense(neuron_per_layer, activation=actfn,
      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_seed))(layer_result)

# output layer
output_layer = tf.keras.layers.Dense(1, activation=None,
                                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_seed))(layer_result)

model = tf.keras.Model(input_layer, output_layer)

model.summary()

# u(t, x) just makes working with model easier and the whole code looks more
# like its mathematical backend
@tf.function
def u(t, x):
    # model input shape is (2,) and `u` recieves 2 arguments with shape (1,)
    # to be able to feed those 2 args (t, x) to the model, a shape (2,) matrix
    # is build by simply concatenation of (t, x)
    u = model(tf.concat([t, x], axis=1)) # note the axis ; `column`
    u = model(tf.concat([t, x], axis=1))- model(tf.concat([t*0, x], axis=1))+initial_condition_function(x) # note the axis ; `column`

    return u

# the physics informed loss function
# IMPORTANT: this loss function is used for collocation points
@tf.function
def f(t, x):
    u0 = u(t, x)
    u_t = tf.gradients(u0, t)[0]
    u_x = tf.gradients(u0, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    F = u_t + u0*u_x - (0.01/np.pi)*u_xx
    return tf.reduce_mean(tf.square(F))

# MSE loss function
# IMPORTANT: this loss function is used for data points
@tf.function
def mse(y, y_):
    return tf.reduce_mean(tf.square(y-y_))

import time

epochs = 500
#epochs=100
loss_list = []

# L-BFGS optimizer was used in the reference paper
opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
start = time.time()

# training loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # model output/prediction
        y_ = u(t_d, x_d)
        # physics-informed loss for collocation points
        L1 = f(t_c, x_c)
        # MSE loss for data points
        L2 = mse(y_d, y_)
        loss = L1 + L2
    # compute gradients
    g = tape.gradient(loss, model.trainable_weights)
    loss_list.append(loss)
    # log every 10 epochs
    if (not epoch%10) or (epoch == epochs-1):
        print(f"{epoch:4} {loss.numpy():.3f}")
    # apply gradients
    opt.apply_gradients(zip(g, model.trainable_weights))

end = time.time()
print(f"{end - start:.3} (s)")

plt.semilogy(range(len(loss_list)), loss_list)
plt.ylabel("loss")
plt.xlabel("epoch")

#Analytical solution to Burgers equation using a Hopf-cole transformation
from scipy import integrate
nu = 0.01/np.pi

def f_cole(y):
    return np.exp(-np.cos(np.pi*y)/(2*np.pi*nu))

def part_1_of_integral(eta, x, t):
    return np.sin(np.pi*(x-eta))*f_cole(x-eta)*np.exp(-eta**2/(4*nu*t))

def part_2_of_integral(eta, x, t):
    return f_cole(x-eta)*np.exp(-eta**2/(4*nu*t))

def u_exact(t, x):
#  t=t+1.e-3
  if t < 1.e-1:
      return -np.sin(np.pi*x)
  else:
      I1 = integrate.quad(part_1_of_integral, -np.inf, np.inf, args=(x,t))[0]
      I2 = integrate.quad(part_2_of_integral, -np.inf, np.inf, args=(x,t))[0]
      return -I1/I2
#  return -(integrate.quad(part_1_of_integral, -np.inf, np.inf, args=(x,t))[0]) / \
#          (integrate.quad(part_2_of_integral, -np.inf, np.inf, args=(x,t))[0])

### plot
n, m = 51, 101
Xrange = np.linspace(-1, +1, m)
Trange = np.linspace(0, 1, n)
X0, T0 = np.meshgrid(Xrange, Trange)
Xmesh = X0.reshape([n*m, 1])
Tmesh = T0.reshape([n*m, 1])
X = tf.convert_to_tensor(Xmesh)
T = tf.convert_to_tensor(Tmesh)
X.shape, T.shape
S = u(T, X)
S = S.numpy().reshape(n, m)

if(False):#use exact formula ... not too stable
  exactS=np.array([u_exact(Tmesh[ii,0],Xmesh[ii,0]) for ii in range(Xmesh.shape[0])])
  exactS=exactS.reshape(n,m)

plt.figure("solution",figsize=(6, 2), dpi=150)

plt.subplot(1,2,1)
plt.pcolormesh(T0, X0, S, cmap=cm.rainbow)
plt.colorbar()
plt.xlim(0., +1)
plt.ylim(-1, +1)
plt.title("u(x, t) (numerical)")
plt.ylabel("x")
plt.xlabel("t")
plt.scatter(t_d, x_d, marker="x", c="k")
if(False):
  plt.subplot(1,2,2)
  plt.pcolormesh(T0, X0, exactS, cmap=cm.rainbow)
  plt.colorbar()
  plt.xlim(0., +1)
  plt.ylim(-1, +1)
  plt.title("u(x, t) (exact)")
  plt.ylabel("x")
  plt.xlabel("t")
  plt.scatter(t_d, x_d, marker="x", c="k")
plt.tight_layout()
plt.show()

plt.figure("solution at different times",figsize=(8, 4), dpi=150)
plt.xlim(-1, +1)
plt.xlabel("x")
plt.plot(X0[::10,:].T,S[::10,:].T)
plt.legend([ 't='+str(np.round(tt,2)) for tt in Trange[::10] ])
plt.tight_layout()
plt.savefig('burgers_lambda'+str(lambda_sample)+"evolution.pdf")

"""
solve by finite differences Burgers equation
u_t + u u_x = nu u_xx
nu = 0.01/pi
x in [-1,1]
t in [0,1]

use a combination of finite difference schemes for the Burgers' equation:
    the FTCS (Forward-Time Centered-Space) scheme for the viscosity term
    and the Lax-Friedrichs scheme for the nonlinear term.
    This combination can provide stability and accuracy improvements over
    purely explicit schemes.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Parameters
nu = 0.01/np.pi  # Viscosity coefficient
nx = 1001   # Number of spatial grid points
Tfinal=1.
nt = 2000   # Number of time steps
dx = 2.0 / (nx - 1)  # Spatial step size
dt = Tfinal/nt  # Time step size

# Spatial grid
x = np.linspace(-1.0, 1.0, nx)

# Initial condition (sinusoidal)
u0 = -np.sin(np.pi * x)
solution=np.zeros((nt+1,nx))
solution[0,:]=u0.copy()

# Initialize solution array
usol = np.copy(u0)

# Perform time stepping using explicit finite differences
for n in range(nt):
    un = np.copy(usol)  # Copy the solution from the previous time step

    # Apply FTCS scheme for viscosity term (nu * d^2u/dx^2)
    usol[1:-1] = un[1:-1] + nu * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2])

    # Apply Lax-Friedrichs scheme for nonlinear term (u * du/dx)
    f_plus = 0.5 * (un[2:]**2)
    f_minus = 0.5 * (un[:-2]**2)
    usol[1:-1] = 0.5 * (un[2:] + un[:-2]) - 0.5 * dt / dx * (f_plus - f_minus)

    # Apply homogeneous boundary conditions (u(-1, t) = u(1, t) = 0)
    usol[0] = 0.0
    usol[-1] = 0.0
    solution[n+1,:]=usol.copy()

final_solution=interp1d(x,solution[-1,:],fill_value="extrapolate")

pinn_sol=S[-1,:]
fd_sol=np.array([final_solution(xx) for xx in Xrange])
plt.figure('solutions')
plt.plot(Xrange,pinn_sol,label='PINN solution')
plt.plot(Xrange,fd_sol,label='finite differences solution')
plt.xlabel('x')
plt.legend()
plt.savefig('burgers_lambda'+str(lambda_sample)+"final_sol.pdf")
plt.figure('error')
plt.plot(Xrange,pinn_sol-fd_sol)
plt.xlabel('x')
plt.savefig('burgers_lambda'+str(lambda_sample)+"final_diff.pdf")
L1_err=np.mean(np.abs(pinn_sol-fd_sol))
L2_err=np.sqrt(np.mean(np.abs(pinn_sol-fd_sol)**2))
print('L1err=',L1_err,' lambda=',lambda_sample)
print('L2err=',L2_err,' lambda=',lambda_sample)

nx=500 #mesh size
Nmax=epochs
alpha=0.9
dx=5/nx
dt=alpha*dx
lam=dt/dx

T=1.
x=np.linspace(-1,1,nx+1)
nu = 0.01/np.pi


def u_0(x): return -np.sin(np.pi*x)
u0=u_0(x)

#Function for the Lax-Friedrichs scheme
def h(u,v):
    return 0.25*(u**2 + v**2)- (0.5/lam)*(v-u)

u_fd=[]
u_fd.append(u0.copy()) #We copy u0 to do not modify it

t,n=1e-16,0 # We take t and n very close to 0 but not 0 to prevent from a dimension problem

for i in range(int(T/dt)):
    dt=alpha*dx/np.amax(np.abs(u_fd))
    lam=dt/dx

    u1=np.zeros(nx+1)
    u1[1:nx+1]=[u_fd[i][j] for j in range(nx)]
    u1[0]=u_fd[i][nx-1]
    u2=np.zeros(nx+1)
    u2[0:nx]=[u_fd[i][j] for j in range(1,nx+1)]
    u2[nx]=u_fd[i][1]
    #scheme formula
    u_fd.append(u_fd[i] - lam*(h(u_fd[i],u2) - h(u1,u_fd[i])) - nu*lam*(u1-2*u_fd[i]+u2)/dx)
    t=t+dt

print(len(u_fd))

#Plot #4
mse=[]
x = np.expand_dims(np.linspace(-1, +1, nx+1), axis=1)
time_points = [0, 0.25, 0.5, 0.75, 1]
time_points = [Tfinal]

plt.figure(figsize=(9,len(time_points)*3), dpi=300)
for i, time_point in enumerate(time_points, start=1):
    t = np.ones_like(x) * time_point
    diff=np.zeros(nx+1)
    for j in range(nx):
      diff[j] = (float(u_fd[(i-1)*25][j] - u(t,x)[j]))**2
    mse.append(np.mean(diff))
    plt.subplot(len(time_points), 1, i)
    plt.title(f"t = {time_point}")
    plt.plot(x, diff, 'r')
    plt.ylabel(f"u(t, x)")
    plt.xlabel(f"x")
    plt.xlim(-1, +1)
    plt.ylim(-1, +1)
    plt.grid(True)
plt.tight_layout()
plt.show()
print('MSE = ', mse)
