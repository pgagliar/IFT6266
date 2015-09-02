__author__ = 'pierregagliardi'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy

#Plot the training error and the validation error according to the number of epochs
def plot_training_validation_error():
    path='/Users/pierregagliardi/DossierTravail/Programmation/PythonPath/ProjetCatsVsDogs/Figures'

    train_data=numpy.load(path+'/training_data.npy')
    validation_data=numpy.load(path+'/validation_data.npy',)

    n_epochs=train_data.shape[0]

    plt.plot(range(n_epochs),train_data,color='r',label='training',marker='o')
    plt.plot(range(n_epochs),validation_data,color='b',label='validation', marker='o')

    #ylabel error rates
    plt.ylabel('Error rates')

    #xlabel number of epochs
    plt.xlabel('Number of epochs')

    legend = plt.legend(loc='upper center')
    plt.grid(True)
    fig = plt.gcf()
    fig.savefig(path+'/train_validation_error_test.png')
