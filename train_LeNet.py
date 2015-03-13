import time
import os
import sys
import theano
import numpy
import matplotlib.pyplot as plt
from theano import tensor as T
from LeNet_conv_poollayer import LeNetConvPoolLayer
from hidden_layer import HiddenLayer
from logistic_regression_class import LogisticRegression
from ift6266h15.code.pylearn2.datasets.variable_image_dataset import DogsVsCats, RandomCrop

def shared_dataset(data_x, data_y, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """

    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX),borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def relu(x):
    return theano.tensor.switch(x<0, 0, x)

if __name__=="__main__":
 #config.compute_test_value = 'raise'
 rng=numpy.random.RandomState(23455)
 dataset = DogsVsCats(transformer=RandomCrop(256, 221),
                         start=0, stop=20000)

 it = dataset.iterator(mode='random_slice', batch_size=2500, num_batches=1)
 for X, y in it:

    train_set_x, train_set_y = shared_dataset(X[0:2000,:,:,:], y[0:2000].reshape(2000) )
    valid_set_x, valid_set_y = shared_dataset(X[2000:2250,:,:,:], y[2000:2250].reshape(250) )
    test_set_x, test_set_y = shared_dataset(X[2250:2500,:,:,:], y[2250:2500].reshape(250))


    nkerns=numpy.array([32,16,16,16])


    # Hyperparameters
    learning_rate = 0.01
    batch_size = 100
    n_epochs=10

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size


    index=T.iscalar('index')
    x = T.tensor4(name='x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels



    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 221 * 221)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (221, 221) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 221, 221))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (221+5-1 , 221+5-1) = (224, 224)
    # maxpooling reduces this further to (112, 112)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 112, 112)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 221, 221),
        filter_shape=(nkerns[0],3, 4, 4),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (112+5-1, 112+5-1) = (116, 116)
    # maxpooling reduces this further to (116/2, 116/2) = (58, 58)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 58, 58)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 112, 112),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (58+5-1, 58+5-1) = (62, 62)
    # maxpooling reduces this further to (62/2, 62/2) = (31, 31)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 31, 31)

    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 58, 58),
        filter_shape=(nkerns[2], nkerns[1], 4, 4),
        poolsize=(2, 2)
    )

     # Construct the second convolutional pooling layer
    # filtering reduces the image size to (31+4-1, 31+4-1) = (34, 34)
    # maxpooling reduces this further to (34/2, 34/2) = (17, 17)
    # 4D output tensor is thus of shape (batch_size, nkerns[2], 17, 17)

    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], 31, 31),
        filter_shape=(nkerns[3], nkerns[2], 4, 4),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[3] * 18 * 18),
    # or (100, 16 * 18* 18) = (100, 5184) with the default values.
    layer4_input = layer3.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer4 = HiddenLayer(
        rng,
        input=layer4_input,
        n_in=nkerns[3] * 17 * 17,
        n_out=16,
        activation=relu
    )

    # construct a fully-connected sigmoidal layer
    layer5 = HiddenLayer(
        rng,
        input=layer4.output,
        n_in=16,
        n_out=16,
        activation=relu
    )

    # classify the values of the fully-connected sigmoidal layer
    layer6 = LogisticRegression(input=layer5.output, n_in=16, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = layer6.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 2000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    train_error=numpy.zeros(n_train_batches)
    train_graph=numpy.zeros(n_epochs)
    validation_graph=numpy.zeros(n_epochs)

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            train_error[minibatch_index]=minibatch_avg_cost

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                validation_graph[epoch-1]=this_validation_loss
                train_graph[epoch-1]=numpy.mean(train_error)


                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.

                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break


    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    plt.plot(range(n_epochs),train_graph,color='r',label='training',marker='o')
    plt.plot(range(n_epochs),validation_graph,color='b',label='validation', marker='o')
    plt.ylabel('Error rates')
    plt.xlabel('Number of epochs')
    legend = plt.legend(loc='upper center')
    plt.grid(True)
    plt.show()
