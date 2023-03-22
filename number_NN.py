import numpy
import matplotlib.pyplot
from main import NeuralNetwork
inputnodes = 784
hidenodes = 200
outputnodes = 10
learningrate = 0.3

n = NeuralNetwork(inputnodes,hidenodes,outputnodes,learningrate)


training_data_file = open('mnist_train_100.csv')
training_data_list = training_data_file.readlines()
training_data_file.close()

test_data_file = open('mnist_test_10.txt')
test_data_list = test_data_file.readlines()
t = test_data_list[0].split(',')

for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:])/255*0.99)+0.01

    targets = numpy.zeros(outputnodes) + 0.01
    targets[int(all_values[0])] = 0.99

    n.train(inputs,targets)

print(n.query((numpy.asfarray(t[1:])/255*0.99)+0.01)) #7