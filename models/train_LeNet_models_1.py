import time
import os
import sys
import theano
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Optimization_error_function.momentum import momentum_bis
from Optimization_error_function.rmsprop import rmsprop
from theano import tensor as T
from Layers.LeNet_conv_poollayer import LeNetConvPoolLayer
from Layers.hidden_layer import HiddenLayer
from Layers.logistic_regression_class import LogisticRegression
from ift6266h15.code.pylearn2.datasets.variable_image_dataset import DogsVsCats, RandomCrop


def relu(x):
    return theano.tensor.switch(x<0, 0, x)

if __name__=="__main__":
 #config.compute_test_value = 'raise'
    rng=numpy.random.RandomState(23455)

    # Hyperparameters
    learning_rate = 0.01
    n_epochs=200
    nkerns=numpy.array([8,16,16,16,32])
    batch_size=50
    momentum=0.9

    #Symbolic theano variables
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
    layer0_input = x.reshape((batch_size, 3, 100, 100))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (221+5-1 , 221+5-1) = (224, 224)
    # maxpooling reduces this further to (112, 112)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 112, 112)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 100, 100),
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
        image_shape=(batch_size, nkerns[0], 52,52),
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
        image_shape=(batch_size, nkerns[1], 28, 28),
        filter_shape=(nkerns[2], nkerns[1], 4, 4),
        poolsize=(2, 2)
    )

     # Construct the second convolutional pooling layer
    # filtering reduces the image size to (31+5-1, 31+5-1) = (35, 35)
    # maxpooling reduces this further to (34/2, 34/2) = (17, 17)
    # 4D output tensor is thus of shape (batch_size, nkerns[2], 17, 17)

    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], 16, 16),
        filter_shape=(nkerns[3], nkerns[2], 4, 4),
        poolsize=(2, 2)
    )
    
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (16+5-1, 16+5-1) = (20, 20)
    # maxpooling reduces this further to (20/2, 20/2) = (10, 10)
    # 4D output tensor is thus of shape (batch_size, nkerns[2], 10, 10)

    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, nkerns[3], 10, 10),
        filter_shape=(nkerns[4], nkerns[3], 4, 4),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[3] * 18 * 18),
    # or (100, 16 * 18* 18) = (100, 5184) with the default values.
    layer5_input = layer4.output.flatten(2)

    # construct a fully-connected relu layer
    layer5 = HiddenLayer(
        rng,
        input=layer5_input,
        n_in=nkerns[4] * 7 * 7,
        n_out=256,
        activation=relu
    )

    # construct a fully-connected relu layer
    layer6 = HiddenLayer(
        rng,
        input=layer5.output,
        n_in=256,
        n_out=256,
        activation=relu
    )

    # classify the values of the fully-connected sigmoidal layer
    layer7 = LogisticRegression(input=layer6.output, n_in=256, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = layer7.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [x,y],
        layer7.errors(y),
    )

    validate_model = theano.function(
        [x,y],
        layer7.errors(y),
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer7.params+layer6.params + layer5.params+ layer4.params+ layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    # grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.

    # updates = [
    #     (param_i, param_i - learning_rate * grad_i)
    #     for param_i, grad_i in zip(params, grads)
    # ]
    updates = momentum_bis(cost, params, learning_rate)

    train_model = theano.function(
        [x,y],
        cost, #Le calcul de l'erreur est la partie forward propagation
        updates=updates #L'update est la partie backward propagation
     )
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    start_time = time.clock()
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score=0
    start_time_total = time.clock()

    epoch =0
    done_looping = False

    train_set = DogsVsCats(transformer=RandomCrop(256, 100),start=0, stop=20000)
    valid_set=DogsVsCats(transformer=RandomCrop(256, 100),start=20000, stop=22500)
    test_set=DogsVsCats(transformer=RandomCrop(256, 100),start=22500, stop=25000)

     # compute number of minibatches for training, validation and testing
    n_train_batches = train_set.get_num_examples()/batch_size
    n_valid_batches = valid_set.get_num_examples()/ batch_size
    n_test_batches = test_set.get_num_examples()/ batch_size

    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    train_error=numpy.zeros(n_train_batches)
    train_graph=numpy.zeros(n_epochs)
    validation_graph=numpy.zeros(n_epochs)

    while (epoch < n_epochs) and (not done_looping):
        start_time_epoch = time.clock()
        epoch = epoch + 1
        it_train = train_set.iterator(mode='random_slice', batch_size=batch_size, num_batches= n_train_batches)
        it_valid = valid_set.iterator(mode='random_slice', batch_size=batch_size, num_batches= n_valid_batches)
        it_test = test_set.iterator(mode='random_slice', batch_size=batch_size, num_batches= n_test_batches)
        minibatch_index=0

        for X_train, y_train in it_train:
            y_train_reshape_cast=y_train.reshape(batch_size).astype('int32')
            minibatch_avg_cost = train_model(X_train,y_train_reshape_cast )
            train_error[minibatch_index]=minibatch_avg_cost

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(X_valid,y_valid.reshape(batch_size).astype('int32'))
                                     for X_valid, y_valid in it_valid]
                this_validation_loss = numpy.mean(validation_losses)

                validation_graph[epoch-1]=this_validation_loss
                train_graph[epoch-1]=numpy.mean(train_error)
                end_time_epoch = time.clock()

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.

                    )
                )
                print(('     epoch %i, minibatch %i/%i, train error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           train_graph[epoch-1] * 100.)
		)
                print >> sys.stderr,('The epoch %i ran for %.2fm' %
                                       (
                                            epoch,
                                            (end_time_epoch- start_time_epoch) / 60.
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

            if patience <= iter:
                done_looping = True
                break
            minibatch_index=minibatch_index+1

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    path='/u/gagliarp/git/ProjetCatsVsDogs/Figures/'
    numpy.save(path+'training_data',train_graph)
    numpy.save(path+'validation_data',validation_graph)

    plt.plot(range(n_epochs),train_graph,color='r',label='training',marker='o')
    plt.plot(range(n_epochs),validation_graph,color='b',label='validation', marker='o')
    plt.ylabel('Error rates')
    plt.xlabel('Number of epochs')
    legend = plt.legend(loc='upper center')
    plt.grid(True)
    fig = plt.gcf()
    fig.savefig(path+'train_validation_error.png')
