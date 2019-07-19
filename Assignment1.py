
# coding: utf-8

# In[1]:


#Prabhat Thakur Date 04/20/2019
#2019SP_MSDS_458-DL_SEC56 Week3- Assignment1
#25-Input-Node Alphabet Letter Classification (5 Classes)
#Code Addopted from simple_5x5-letter-classification_tutorial-code_v7_2017-10-15.py and F. Challot (2018), Deep Learning with Python (Manning)

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors

from keras import models
from keras import layers
from keras import backend as K

# to make this notebook's output stable across runs
np.random.seed(42)

# For pretty-printing the arrays
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True) 


# In[2]:


def welcome ():
    print (' ' )
    print ('******************************************************************************')
    print (' ' )
    print ('This program learns to distinguish between five capital letters: X, M, H, A, and N')
    print ('It allows users to examine the hidden weights to identify learned features')
    print (' ' )
    print ('******************************************************************************')
    print (' ' )
    return()


# In[3]:


####################################################################################################
####################################################################################################
#
# Function to obtain the neural network size specifications
#
####################################################################################################
####################################################################################################

def obtainNeuralNetworkSizeSpecs ():

# This procedure operates as a function, as it returns a single value (which really is a list of 
#    three values). It is called directly from 'main.'
#        
# This procedure allows the user to specify the size of the input (I), hidden (H), 
#    and output (O) layers.  
# These values will be stored in a list, the arraySizeList. 
# However, even though we're calling this procedure, we will still hard-code the array sizes for now.   

    numInputNodes = 25
    numHiddenNodes = 8
    numOutputNodes = 5   
    print (' ')
    print ('  The number of nodes at each level are:')
    print ('    Input: 5x5 (square array)')
    print ('    Hidden: 8')
    print ('    Output: 5 (Five classes)')
            
# We create a list containing the crucial SIZES for the connection weight arrays                
    arraySizeList = (numInputNodes, numHiddenNodes, numOutputNodes)
    
# We return this list to the calling procedure, 'main'.       
    return (arraySizeList)  


# In[4]:


# Function to return a trainingDataList
# The training data list have 5x5 pixel-grid representation of the letter
#       represented as a 1-D array (0 or 1 for each)
# Also contains string associated with that class, e.g., 'X'
# We are starting with five letters in the training set: X, M, N, H, and A
# Thus there are five choices for training data, which we'll select on random basis

def process_data():

    training_alpha = [(0,[1,0,0,0,1, 0,1,0,1,0, 0,0,1,0,0, 0,1,0,1,0, 1,0,0,0,1],'X'),
                     (1,[1,0,0,0,1, 1,1,0,1,1, 1,0,1,0,1, 1,0,0,0,1, 1,0,0,0,1],'M'),
                     (2,[1,0,0,0,1, 1,1,0,0,1, 1,0,1,0,1, 1,0,0,1,1, 1,0,0,0,1],'N'),
                     (3,[1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1],'H'),
                     (4,[0,0,1,0,0, 0,1,0,1,0, 1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1],'A')]
   
    # List of indices to be one-hot encoded
    indices = []

    # Initial array to load training data
    training_data = np.zeros((len(training_alpha), 25))

    # List of letter labels
    letters = []

    # Loop through letters to process data
    for i in training_alpha:
        indices.extend([i[0]])
        training_data[i[0]] = i[1]
        letters.extend([i[2]])

    # One hot encode letters
    indices_array = np.array(indices)
    labels = np.zeros((5, 5))
    labels[np.arange(5), indices_array] = 1

    return labels, training_data, letters


# In[5]:


def display_letter(case_input_single):
    """Quick display of letter (assuming square)"""

    # Dimension assuming square
    img_size = np.sqrt(case_input_single.shape[0]).astype(int)

    # Reshape assumping square
    img = case_input_single.reshape(img_size, img_size)

    # Select white background with blue letters
    colors = np.array([[1, 1, 1],
                       [0, 0, 1]])
    cmap = matplotlib.colors.ListedColormap(colors)

    # Show image
    plt.imshow(img, cmap=cmap)
    plt.show()


# In[6]:


def display_all(case_input):
    """Plot all letters in a set"""

    for i in range(0, case_input.shape[0]):
        display_letter(case_input[i])


# In[7]:


def print_weights(model):
    
    weights0, biases0 = model.layers[0].get_weights()
    print ('weights - Input-Hidden')
    print (weights0)
    print ('biases - Input-Hidden')
    print (biases0)
    print ('******************************************************************************')    
    weights1, biases1 = model.layers[1].get_weights()
    print ('weights - Hidden-Output')
    print (weights1)
    print ('biases - Hidden-Output')
    print (biases1)  
    print ('******************************************************************************')
    print ('******************************************************************************')
    return [weights0, weights1]


# In[8]:


def display_influence(case_input_single, credit1,output):
    """Quick display of hidden layer activations for a single letter"""
    
    # Dimension assuming square
    img_size = np.sqrt(case_input_single.shape[0]).astype(int)

    # Reshape assumping square
    img = case_input_single.reshape(img_size, img_size)

    # Select white background with gray letters
    colors = np.array([[1, 1, 1],
                       [0.9, 0.9, 0.9]])

    cmap = matplotlib.colors.ListedColormap(colors)

    # Show image
    plt.imshow(img, cmap=cmap)
    plt.show()
    
    # Normalize for uniform colors
    credit = (credit1 - credit1.min()) / (credit1.max() - credit1.min())

    # Loop through nodes
    #f,ax = plt.subplots(1,credit.shape[0])
    
    for i in range(0, credit.shape[0]):

        #print(f"\nActivation = {output[0][0][i]:.4f}")
        print('Node "%s" activation value= ' %str(i+1),output[0][0][i])
        # Plot another layer showing infuence values
        influence = credit[i].reshape(img_size, img_size)

        plt.imshow(influence, vmin=0, vmax=1,
                         cmap=plt.cm.coolwarm, alpha=0.5, interpolation='bilinear',)
        plt.show()


# In[9]:


def activation_outputs(model,data,labels,weight):
    print ('******Input to Hidden Layer Weights After Training **********')
    weightnpa=np.asarray(weight, dtype=np.float32)
    weightnpa = weightnpa.transpose()
    print(type(weightnpa))
    print (weightnpa.shape[:])
    print (weightnpa)
        
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]
    
    for i in range(0, data.shape[0]):
        data_single = data[i:i+1]
        target_single = labels[i]
             
        layer_outs = [func([data_single,1.]) for func in functors]
        print ('Hidden Layer Outputs for "%s":' %target_single, layer_outs[0])
        print ('Output Layer Outputs for "%s":' %target_single, layer_outs[1])

        credit = data_single[0] * weightnpa
        display_influence(data_single[0],credit,layer_outs[0])


# In[10]:


def main():
    """Load data, visualize, train model"""
    welcome()    
        
    # Define the variable arraySizeList, which is a list. It is initially an empty list. 
    # Its purpose is to store the size of the array.

    arraySizeList = list() # empty list

    # Obtain the actual sizes for each layer of the network       
    arraySizeList = obtainNeuralNetworkSizeSpecs ()

    # Unpack the list; ascribe the various elements of the list to the sizes of different network layers    
    inputArrayLength = arraySizeList [0]
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]

    # Load data
    training_labels, training_data, letters = process_data()

    # Visualize all letters
    display_all(training_data)

    #Neural Network with Densely connected (fully connected) Single hidden layer 
    network = models.Sequential()
    network.add(layers.Dense(hiddenArrayLength, activation='relu', input_shape= (inputArrayLength,)))
    network.add(layers.Dense(outputArrayLength, activation='softmax'))
    
    network.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    print ('Weights and Biases Before Traning:')    
    weightsBT = print_weights(network)
 
    history = network.fit(training_data, training_labels, epochs=500, batch_size=5)

    print ('Weights and Biases After Traning:')
    weightsAT =print_weights(network)

    history_dict = history.history
    print ('history_dict.keys()')
    print (history_dict.keys())   

    loss_values = history_dict['loss']
    acc_values = history_dict['acc']
    
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, acc_values, 'b', label='Training acc')
    plt.title('Training loss and Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()
    
    print ('******************************************************************************') 
    print ('Testing for training_data X')
    print (training_data[0])
    print ('training_labels One Hot encoding')
    print (training_labels[0])
    
    test_loss, test_acc = network.evaluate(training_data[0:1], training_labels[0:1])
    print('test_acc:', test_acc)

    predictions = network.predict(training_data[0:1])
    print ('# of Predicted values : ', predictions[0].shape)
    print ('Predicted probabilities of each letter : ', predictions[0])
    print ('Sum of Predicted probabilities : ', np.sum(predictions[0]))
    print ('The class with the highest probability : ', np.argmax(predictions[0]))
    print ('Predicted Letter : ',letters[np.argmax(predictions[0])])
    print ('******************************************************************************')    
    
    activation_outputs(network,training_data,letters,weightsAT[0])


# In[11]:


if __name__ == "__main__": main()

