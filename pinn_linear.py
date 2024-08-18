# -*- coding: utf-8 -*-
"""pinn_time_sampling_v1_5.ipynb

#Time sampling for Physics-Informed Neural Networks#


Supporting implementations for the paper 
Optimal time sampling in physics-informed neural networks
by
Gabriel Turinici
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from scipy import integrate
from scipy.integrate import odeint
from scipy.stats import truncexpon

tf.keras.backend.set_floatx("float64")
#random_seed= np.random.choice(1000)
random_seed=964
print('We set seed to obtain reproducible results seed=',random_seed)

tf.__version__

"""##Parameters and problem definition##

###Parameters
"""

backup_filename_prefix="time_sampling"

input_dim=1
#output_dim=2
output_dim=1
test_case='dx/dt=lambda*x(t)'
test_case +='_seed'+str(random_seed)

init_val_loss_coeff=1.0
test_case +='_IVcoef'+str(init_val_loss_coeff)

sampling_fixed=[True,False][0]
print('sampling_fixed=',sampling_fixed)
test_case +='_fixed_'+str(sampling_fixed)

# Model and problem architecture
nb_hidden_layers=5
test_case +='_NbL'+str(nb_hidden_layers)
neuron_per_layer = 10
test_case +='_Nbn'+str(neuron_per_layer)

if(output_dim==1):
  x0=1.0*np.sqrt(15)
else:
  x0=np.sqrt(15)*np.ones(output_dim)

"""###Rate and sampling rate"""

lambda_coeff=2.0
sampling_lambda=-1.#can be lambda,-lambda,0=uniform or other value
print('sampling_lambda=',sampling_lambda)
test_case +='_sampLambda_'+str(sampling_lambda)

"""###Problem definition"""

T=1
test_case +='_x0'+str(np.round(x0,2))+'_lam'+str(lambda_coeff)+'_T'+str(T)

print('test case=',test_case)

def ode_function(t,x):
  #sin lambda t equation: for vector arguments we expect batch is first and then are the dimensions
 s,c=x[...,0],x[...,1]
 return np.array([lambda_coeff*c,-lambda_coeff*s])

@tf.function
def ode_function(t,x):#sin lambda t equation
 s,c=x[...,0],x[...,1]
 return tf.stack([lambda_coeff*c,-lambda_coeff*s])

@tf.function
def tf_ode_function(t,x):
  return lambda_coeff*x


def ode_function(t,x):
  return lambda_coeff*x

time_batch_size=100
trange_odeint=np.linspace(0,T,time_batch_size,endpoint=True)
odeint_sol=odeint(ode_function,x0,t=trange_odeint,tfirst=True)

debug=True

def sample_truncated(size=(time_batch_size,input_dim),rate=sampling_lambda):
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
  if(np.abs(rate*T)<1.e-6):
    return U_samples*T
  else:
    return -np.log(1-U_samples+U_samples*np.exp(-rate*T))/rate

def quantiles_truncated(size=(time_batch_size,input_dim),rate=sampling_lambda):
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
  if(np.abs(rate*T)<1.e-6):
    return U_samples*T
  else:
    return -np.log(1-U_samples+U_samples*np.exp(-rate*T))/rate

"""#**The model**
"""

# convert all data to tf.Variable
#time_sample_tensor=tf.convert_to_tensor(t_sampled)
time_sample_tensor=tf.Variable(trange_odeint.reshape(time_batch_size,input_dim))
#initial_time_tensor=tf.convert_to_tensor(np.zeros(1))
initial_time_tensor=tf.Variable(np.zeros(1).reshape((1,input_dim)))

"""##**The architecture of the Model**

"""

# activation function for all hidden layers
activation_function = "tanh"

# input layer with 2 neurons
input_layer = tf.keras.layers.Input(shape=(input_dim,))

# hidden layers
current_map = input_layer
for _ in range(nb_hidden_layers):
    current_map = tf.keras.layers.Dense(neuron_per_layer,
                    activation=activation_function,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_seed)
                  )(current_map)

# last layer has a single output without activation function
output_layer = tf.keras.layers.Dense(output_dim, activation=None,
    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_seed))(current_map)

model = tf.keras.Model(input_layer, output_layer)
model.summary()

# u(t, x) just makes working with more intuitive model and the whole code looks
#like its mathematical backend
@tf.function
def u(t):
    # model input shape is (1,) if `u` receives 2 arguments with shape (1,)
    # to be able to feed those 2 args (t, x) to the model, a shape (2,) matrix
    # is build by simply concatenation of (t, x)
#    u = model(t) #
#      u = model(tf.concat([t], axis=1)) # note the axis ; `column`
#option: set exact initial condition:
    u = model(t)-model(initial_time_tensor)+x0

    return u

# equation error function
@tf.function
def equation_loss(t):
  if(True):
    u_val = u(t)
    u_t = tf.gradients(u_val, t)[0]
#    u_t = tf.gradients(u_val, t)
    E = u_t - tf_ode_function(t,u_val)
  if(False):
    with tf.GradientTape() as tape2:
      tape2.watch(t)
      u_val = u(t)
    u_t = tape2.gradient(u_val, t)[0]
    print("u_t shape=",u_t.shape)
    E = u_t - tf_ode_function(t,u_val)

  if(debug):
    equation_loss.count +=1
    print('f count=',equation_loss.count)
    equation_loss.u_val=u_val
    equation_loss.u_t=u_t
  return tf.reduce_mean(tf.square(E))
if(debug):
  equation_loss.count=0

# MSE loss function
# This function computes the mean squarred error between the two inputs
@tf.function
def mse(y, y_target):
    return tf.reduce_mean(tf.square(y-y_target))

"""##**Definition of the loss function**##

##Training##
"""

epochs=500
print("sampling_lambda=",sampling_lambda)
print("sampling_fixed=",sampling_fixed)


loss_list = []

# Here we use the Adam optimizer
opt = tf.keras.optimizers.Adam(learning_rate=1.e-3)

start = time.time()
print('eager execution=',tf.executing_eagerly())

# training loop
for epoch in range(epochs):
    with tf.GradientTape(persistent=True) as tape:
        if sampling_fixed==True:
          time_sample_tensor.assign(quantiles_truncated(size=(time_batch_size,input_dim),rate=sampling_lambda))#from truncated exponential distribution; when rate=0 this is uniform
        else:
          time_sample_tensor.assign(sample_truncated(size=(time_batch_size,input_dim),rate=sampling_lambda))#from truncated exponential distribution; when rate=0 this is uniform
        xt_init = u(initial_time_tensor)
        # physics-informed loss for equation
        eq_loss_term = equation_loss(time_sample_tensor)
        # MSE loss for data points
        initial_value_loss = mse(x0, xt_init)
        loss = init_val_loss_coeff*initial_value_loss + eq_loss_term
    # compute gradients
    g = tape.gradient(loss, model.trainable_weights)
    loss_list.append(loss.numpy())
    #break;
    # log every 10 epochs
    if (not epoch%50) or (epoch == epochs-1):
        print(f"{epoch:4} {loss.numpy():.9f}")
    # apply gradients
    opt.apply_gradients(zip(g, model.trainable_weights))

end = time.time()
print(f"{end - start:.3} (s)")

"""##Convergence"""

print('test case: ',test_case,'last loss=',loss_list[-1])

plt.figure('loss',figsize=(3,3))
plt.semilogy(range(len(loss_list)), loss_list)
plt.title(test_case)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig('losses.pdf')

"""##Numerical results


"""

#compute local equation error
local_error=[]
for tt in trange_odeint:
#  time_sample_tensor.assign(tt)
  local_error.append(np.sqrt(equation_loss(np.array([[tt]])).numpy() ) )

plt.figure('solution',figsize=(16, 4), dpi=60)
plt.suptitle(test_case)
sol = u(trange_odeint).numpy()
plt.subplot(1,4,1)
plt.title("x(t)")
plt.xlabel("t")
plt.ylabel("x")
plt.plot(trange_odeint, sol, 'b-.',label='PINN solution')
#plt.plot(trange_odeint,odeint_sol,"g-",label='odeint exact solution')
plt.legend()
plt.subplot(1,4,2)
plt.title("x(t)")
plt.xlabel("t")
plt.ylabel("x")
plt.plot(trange_odeint, sol, 'b-.',label='PINN solution')
plt.plot(trange_odeint,odeint_sol,"g-",label='odeint exact solution')
plt.legend()
plt.subplot(1,4,3)
plt.plot(trange_odeint,local_error,label='equation error')
plt.legend()
plt.subplot(1,4,4)
plt.plot(trange_odeint,odeint_sol-sol,label='PINN-exact')
plt.legend()
plt.savefig('solution.pdf')

"""###Final error"""

print('test case: ',test_case,'final error=',np.abs(odeint_sol[-1]-sol[-1])[0])

"""##Backup data##"""

backup_file_name=backup_filename_prefix+"_T"+str(T)+"_lam"+str(lambda_coeff)+".npz"
print('backup data on file ',backup_file_name)
np.savez(backup_file_name,{m.name:m.numpy() for m in model.trainable_weights}.update(
  {"loss_list":loss_list,"T":T,"x0":x0,
                "input_dim":input_dim,"output_dim":output_dim,"lambda_coeff":lambda_coeff,"nb_hidden_layers":nb_hidden_layers,
"neuron_per_layer":neuron_per_layer,"random_seed":random_seed}) )