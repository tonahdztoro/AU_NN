# -*- coding: utf-8 -*-
"""
last update 15/03/22
@author: Tonatiuh Hernández-del-Toro (tonahdztoro@gmail.com)

Code for the paper 
"AU-NN: ANFIS Unit Neural Network"

arXiv: 
    
This file contains the code of the netowrk AU-NN used in the experiments
For now, the AU-NN only uses the triangular function, and the adam optimizer

To use this model:
    1.- Configure de instances set X as a matrix (no_instances, no_features)
    2.- Configure the classes y in a one_hot fashion
    3.- Create the model using "model = AU_NN(Arch)" according to an architecture Arch in te form:
            Arch = {'NoF': [# of features in the feature vector],
            'NoMF': [# of membership functions],
            'NoAU_Layers': [# of neurons in each layer]}
        Example of a network for 5 classes with 3 hidden layers with 6, 12, 18 neurons
        respectively, and 10 membership functions in all units
            Arch = {'NoF': [no_features, 6, 12, 18],
            'NoMF': [10,10,10],
            'NoAU_Layers': [6,12,18,5]}
    4.- Train the network using the fis function fit as "model.fit(X_train,y_train,epochs,learning_rate)"
        with "X_train" the train set, "y_train" the corresponding classes, "epochs" the number of epochs to train,
        and "learning_rate" as the learning rate
"""


# Libraries needed
import tensorflow as tf
import numpy as np



class AU_NN:
    
    """
    The class AU_NN recieves as input the architecture of the network.
    The architecture must be in te form:
    Arch = {'NoF': [# of Features in the feature vector],
    'NoMF': [# of membership functions],
    'NoAU_Layers': [# of classes]}
    """
    
    def __init__(self,Arch):
        """
        Initialize all the parameters of the network to be trained

        Parameters
        ----------
        Arch : architecture in te form:
            Arch = {'NoF': [# of Features in the feature vector],
            'NoMF': [# of membership functions],
            'NoAU_Layers': [# of classes]}

        Returns
        -------
        None.

        """
        
        self.nof = Arch['NoF']
        self.nomf = Arch['NoMF']
        self.NoAU_Layers = Arch['NoAU_Layers']

        AU_NN_Parameters = []
        
        for layer in range(len(self.NoAU_Layers)):
            
            for _ in range(self.NoAU_Layers[layer]):
                
                NoF = self.nof[layer]
                NoMF = self.nomf[layer]
            
                b = np.linspace(-0.5, 0.5, NoMF).reshape(NoMF,1,1)
                a = np.linspace(-0.35, -0.25, NoMF).reshape(NoMF,1,1)
                c = np.linspace(2.5, 0.35, NoMF).reshape(NoMF,1,1)
                
                B = tf.Variable( np.repeat(b,NoF,axis=1), trainable=True, dtype=tf.float64, 
                                            constraint=lambda z: tf.clip_by_value(z, -10, 10))
                A = tf.Variable( np.repeat(a,NoF,axis=1), trainable=True, dtype=tf.float64,
                                constraint=lambda z: tf.clip_by_value(z, -100, -10 ))
                C = tf.Variable( np.repeat(c,NoF,axis=1), trainable=True, dtype=tf.float64, 
                                    constraint=lambda z: tf.clip_by_value(z, 10, 100))
                
                P = tf.Variable(np.random.randn(1,NoF,NoMF)*0.01, trainable=True, dtype=tf.float64)
                r = tf.Variable(np.zeros((1,1,NoMF)), trainable=True, dtype=tf.float64)
                
                AU_parameters = {'A': A, 'B': B, 'C': C, 'P': P, 'r': r}
                AU_NN_Parameters.append(AU_parameters)
        
        self.AU_NN_Parameters = AU_NN_Parameters
        self.delta = tf.constant(1e-20, dtype=tf.float64)



    
    @tf.function
    def triangular_mf(self,X,parameters,no_layer):
        """
        Evaluates the fuzzy triangular function for given parameters
        
        Parameters
        ----------
        X : The set of instances as a matrix (no_instances, no_features)
        parameters : The parameters of a given ANFIS
        no_layer : The number of the layer to apply

        Returns
        -------
        mu : The value of the triangular function of an instance
        """
        NoF = self.nof[no_layer]
        NoIns = X.shape[0]
        X_reshaped = tf.reshape(X,[NoIns,NoF,1])
        a = tf.reshape(parameters[:,0],[NoF,1])
        b = tf.reshape(parameters[:,1],[NoF,1])
        c = tf.reshape(parameters[:,2],[NoF,1])
        minim = tf.math.minimum((X_reshaped-a)/(b-a), (c-X_reshaped)/(c-b))
        mu = tf.math.maximum(minim, 0)
        return mu


    @tf.function
    def membership_values(self,X, AU_parameters,no_layer):
        """
        Evaluates the membership values of a instance according to a fuzzy triangular function of each ANFIS-Neuron
        
        Parameters
        ----------
        X : The set of instances as a matrix (no_instances, no_features)
        AU_parameters : All the parameters of the Networks
        no_layer : The number of the layer to apply

        Returns
        -------
        values : The evaluated membership values of the features

        """
        MFP = tf.concat([AU_parameters['A'],AU_parameters['B'],AU_parameters['C']],axis=2)
        NoIns = X.shape[0]
        NoF = self.nof[no_layer]
        NoMF = self.nomf[no_layer]
        values = tf.zeros([NoIns,NoF,1], dtype=tf.float64)
        for mf in range(NoMF):
            parameters = MFP[mf,:,:]
            mv = tf.reshape(self.triangular_mf(X,parameters,no_layer),[NoIns,NoF,1])
            values = tf.concat([values,mv],axis=2)
        values = values[:,:,1:]
        return values

    
    
    def forward_each_layer(self,X,no_layer):
        """
        Function to apply layer by layer

        Parameters
        ----------
        X : The set of instances as a matrix (no_instances, no_features)

        Returns
        -------
        y_hat : The results of the last layer

        """
        NoIns = X.shape[0]
        y_hat = tf.Variable(np.zeros((1,NoIns)), trainable=True, dtype=tf.float64)
        
        # For first layer
        if no_layer == 0:
            # For every ANFIS-Neuron in the layer
            for au in range(self.NoAU_Layers[no_layer]):
                
                # Apply all the layer of the each ANFIS
                AU_parameters = self.AU_NN_Parameters[au]
                O_1 = self.membership_values(X, AU_parameters,no_layer)
                w = tf.math.reduce_prod(O_1,axis=1)
                w_hat = w/(tf.reshape(tf.math.reduce_sum(w,axis=1),[NoIns,1]) + self.delta)
                f = tf.matmul(X, AU_parameters['P']) + AU_parameters['r']
                O_4 = w_hat*f
                AU_y_hat = tf.math.reduce_sum(O_4, axis=2)
                y_hat = tf.concat([y_hat,AU_y_hat],axis=0)
            y_hat = y_hat[1:,:]
        
        # For the rest of the layers
        else: 
            # For every ANFIS-Neuron in the layer
            for au in range( np.sum(self.NoAU_Layers[:no_layer]), np.sum(self.NoAU_Layers[:no_layer+1]) ):
                
                # Apply all the layer of the each ANFIS
                AU_parameters = self.AU_NN_Parameters[au]
                O_1 = self.membership_values(X, AU_parameters,no_layer)
                w = tf.math.reduce_prod(O_1,axis=1)
                w_hat = w/(tf.reshape(tf.math.reduce_sum(w,axis=1),[NoIns,1]) + self.delta)
                f = tf.matmul(X, AU_parameters['P']) + AU_parameters['r']
                O_4 = w_hat*f
                AU_y_hat = tf.math.reduce_sum(O_4, axis=2)
                y_hat = tf.concat([y_hat,AU_y_hat],axis=0)
            y_hat = y_hat[1:,:]
        
        return y_hat 
    
    def forward(self,X):
        """
        Function to apply all the layers in the network to a set

        Parameters
        ----------
        X : The set of instances as a matrix (no_instances, no_features)

        Returns
        -------
        y_hat : The results of the last layer

        """
        
        # Apply each layer
        for no_layer in range(len(self.NoAU_Layers)): 
            X = tf.transpose(self.forward_each_layer(X,no_layer))
        
        y_hat =  tf.transpose(X)
        return y_hat

    
    def cost_function(self,X,y):
        """
        The loss function

        Parameters
        ----------
        X : The train set X_train
        y : The classes y_train in a one_hot fashion

        Returns
        -------
        J : The mean squared error as loss function

        """
        y_hat = self.forward(X)
        L = (y_hat - y)**2
        J = tf.math.reduce_sum(L)/np.sum(L.shape)
        return J
    
    
    def fit(self,X,y,epochs,learning_rate):
        """
        Function to train the model over a train set X

        Parameters
        ----------
        X : The train set X_train as a matrix (no_instances, no_features)
        y : The classes y_train in a one_hot fashion
        epochs : The number of epochs to train the network
        learning_rate : The learning rate

        Returns
        -------
        None.

        """
        opt = tf.keras.optimizers.Adam(learning_rate)
        loss = lambda: self.cost_function(X,y)
        for epoch in range(epochs):
            opt.minimize(loss, self.AU_NN_Parameters)


    def predict(self,X):
        """
        Predicts the classes of a set of instances

        Parameters
        ----------
        X : The set of instances as a matrix (no_instances, no_features)

        Returns
        -------
        predictions : A matrix with the predicted classes in a one_hot fashion 
            before the softmax (with continuous values)

        """
        predictions = self.forward(X)
        return predictions
    
    
    
    
    
    
    