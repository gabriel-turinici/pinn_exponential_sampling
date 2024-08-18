# -*- coding: utf-8 -*-
"""PINN for Lorenz system

Supporting implementations for the paper 
Optimal time sampling in physics-informed neural networks
by
Gabriel Turinici

"""

import numpy as np
import tensorflow as tf
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class PhysicsInformedNeuralNetwork:
    def __init__(self, random_seed=123,sampling_lambda = 10.0,
                 lorenz_param = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0, 'T': 0.5},
                 nb_hidden_layers = 5,neuron_per_layer = 20,test_case = 'Lorenz',
                 input_dim = 1,output_dim = 3,time_batch_size = 256,epochs = 50,
                 u0=np.ones(3)):
        tf.keras.backend.set_floatx("float64")
        np.random.seed(random_seed)

        self.random_seed = random_seed
        self.test_case = test_case
        self.nb_hidden_layers = nb_hidden_layers
        self.neuron_per_layer = neuron_per_layer
        self.lorenz_param = lorenz_param
        self.sampling_lambda = sampling_lambda
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_batch_size = time_batch_size
        self.epochs = epochs
        self.u0 = u0
        self.trange=np.linspace(0, self.lorenz_param['T'], self.time_batch_size)
        self.odeint_sol=odeint(self.ode_function,self.u0,t=self.trange,tfirst=True)
        self.loss_list=None
        self.state_labels=['x(t)','y(t)','z(t)']


        self.time_sample_tensor = tf.Variable(
            self.trange.reshape(self.time_batch_size, self.input_dim)
        )
        self.initial_time_tensor = tf.Variable(np.zeros((1, self.input_dim)))
        self.build_model()

    def build_model(self):
        activation_function = "tanh"

        input_layer = tf.keras.layers.Input(shape=(self.input_dim,))
        current_map = input_layer

        for _ in range(self.nb_hidden_layers):
            current_map = tf.keras.layers.Dense(
                self.neuron_per_layer,
                activation=activation_function,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_seed)
            )(current_map)

        output_layer = tf.keras.layers.Dense(
            self.output_dim,
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_seed)
        )(current_map)

        self.model = tf.keras.Model(input_layer, output_layer)
        self.model.summary()

    @tf.function
    def u(self, t):
        u_val = self.model(t)
        return u_val - self.model(self.initial_time_tensor) + self.u0

    @tf.function
    def equation_loss(self, t):
        u_val = self.u(t)
        u_t = tf.concat([tf.gradients(u_val[..., i], t)[0] for i in range(self.output_dim)], axis=1)
        E = tf.exp(-self.sampling_lambda * t / 2.0) * (u_t - self.tf_ode_function(t, u_val))
        return tf.reduce_mean(tf.square(E))

    @tf.function
    def tf_ode_function(self, t, x):
        xl, yl, zl = x[..., 0], x[..., 1], x[..., 2]
        return tf.transpose(tf.stack([
            self.lorenz_param['sigma'] * (yl - xl),
            xl * (self.lorenz_param['rho'] - zl) - yl,
            xl * yl - self.lorenz_param['beta'] * zl
        ]))

    def ode_function(self,t,x):
      xl,yl,zl=x
      return np.array([self.lorenz_param['sigma']*(yl-xl) ,xl*(self.lorenz_param['rho']-zl)-yl,
                        xl*yl-self.lorenz_param['beta']*zl])


    def train(self,epochs=None):
        if epochs is None:
              epochs = self.epochs  # Use class attribute as default value
        loss_list = []
        opt = tf.keras.optimizers.Adam(learning_rate=1.e-3)

        for epoch in range(epochs):
            with tf.GradientTape(persistent=True) as tape:
                loss = self.equation_loss(self.time_sample_tensor)

            g = tape.gradient(loss, self.model.trainable_weights)
            loss_list.append(loss.numpy())

            if not epoch % 50 or epoch == epochs - 1:
                print(f"{epoch:4} {loss.numpy():.9f}")

            opt.apply_gradients(zip(g, self.model.trainable_weights))
        self.loss_list=np.copy(loss_list)
        return loss_list

    def predict(self, new_input):
        # Make predictions using the trained model
        new_input = np.array(new_input)
        new_input =new_input.reshape(-1,self.input_dim)
        prediction = self.u(new_input)
        return prediction.numpy()

    def plot_loss_convergence(self):
      if(self.loss_list is not None):
        print('test case: ',self.test_case,'last loss=',self.loss_list[-1])
        plt.figure('loss',figsize=(3,3))
        plt.clf()
        plt.semilogy(range(len(self.loss_list)), self.loss_list)
        plt.title(self.test_case)
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.savefig('losses.pdf')

    def plot_train_results(self):
      print(self.test_case)
      local_error = [np.sqrt(self.equation_loss(np.array([[tt]])).numpy()) for tt in self.trange]
      plt.figure('solution',figsize=(10.5, 3.5), dpi=300)
      plt.clf()
      sol = self.predict(self.trange)
      plt.subplot(1,3,1)
      plt.xlabel("t")
      for pi in range(self.output_dim):
          plt.plot(self.trange, sol[:,pi],'-.',label=self.state_labels[pi]+' (PINN)',linewidth=4)
      plt.legend()
      plt.subplot(1,3,2)
      plt.xlabel("t")
      for pi in range(self.output_dim):
          plt.plot(self.trange, sol[:,pi],'-.',label=self.state_labels[pi]+' (PINN)',linewidth=4)
      for pi in range(self.output_dim):
          plt.plot(self.trange, self.odeint_sol[:,pi],label=self.state_labels[pi]+' (exact)',linewidth=4)
      plt.legend()
      plt.subplot(1,3,3)
      plt.xlabel("t")
      plt.plot(self.trange,local_error,label='equation error',linewidth=4)
      plt.legend()
      plt.tight_layout()
      plt.savefig('solution_'+self.test_case+str(self.sampling_lambda)+'.pdf')

# Usage:
pinn10 = PhysicsInformedNeuralNetwork(random_seed=1234,sampling_lambda=10.0)
loss_history10 = pinn10.train(10000)
pinn10.plot_loss_convergence()
pinn10.plot_train_results()

pinn0 = PhysicsInformedNeuralNetwork(random_seed=1234,sampling_lambda=0.0)
loss_history0 = pinn0.train(10000)
pinn0.plot_loss_convergence()
pinn0.plot_train_results()

pinnm10 = PhysicsInformedNeuralNetwork(random_seed=1234,sampling_lambda=-10.0)
loss_historym10 = pinnm10.train(10000)
pinnm10.plot_loss_convergence()
pinnm10.plot_train_results()