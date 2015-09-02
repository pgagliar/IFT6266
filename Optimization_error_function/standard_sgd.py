__author__ = 'pierregagliardi'
import theano as T

def standard_sgd(cost, params, learning_rate):
  # create a list of gradients for all model parameters
  grads = T.grad(cost, params)
 # Train_model is a function that updates the model parameters by
 # SGD Since this model has many parameters, it would be tedious to
 # manually create an update rule for each model parameter. We thus
 # create the updates list by automatically looping over all
 # (params[i], grads[i]) pairs.
  return [ (param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads) ]