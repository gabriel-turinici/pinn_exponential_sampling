# pinn_exponential_sampling
Implementation files for the paper "Optimal time sampling in physics-informed neural networks" arXiv:2404.18780 [cs.LG] https://arxiv.org/abs/2404.18780


## Requirements

The following packages are required: tensorflow numpy scipy matplotlib pickle and their dependencies. The installation 
is depending on your precise environment but usually reads :

```setup
pip install -r tensorflow numpy scipy matplotlib pickle 
```

Note: your environment may require "pip3".

The file names correspond to the considered numerical tests. To change the 'r' parameter of the truncated exponential sampling change the
value of 

- "lambda_sample" (burgers)
- "sampling_lambda" (linear and Lorenz)

