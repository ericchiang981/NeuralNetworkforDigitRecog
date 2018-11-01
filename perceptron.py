import numpy as np

import backend

class Perceptron(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.get_data_and_monitor = backend.make_get_data_and_monitor_perceptron()

        "*** YOUR CODE HERE ***"
        self.weightvector = np.zeros((dimensions,))


    def get_weights(self):
        """
        Return the current weights of the perceptron.

        Returns: a numpy array with D elements, where D is the value of the
            `dimensions` parameter passed to Perceptron.__init__
        """
        "*** YOUR CODE HERE ***"
        return self.weightvector

    def predict(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        #print(x)
        #print(np.transpose(self.weightvector))
        pred = np.dot(x, self.weightvector)
        #print(pred)
        if pred >= 0:
        	return 1
        else:
        	return -1

    def update(self, x, y):
        """
        Update the weights of the perceptron based on a single example.
            x is a numpy array with D elements, where D is the value of the
                `dimensions`  parameter passed to Perceptron.__init__
            y is either 1 or -1

        Returns:
            True if the perceptron weights have changed, False otherwise
        """
        "*** YOUR CODE HERE ***"
        predictedY = self.predict(x)
        if predictedY != y:
        	if predictedY == -1 and y == 1:
        		self.weightvector = np.add(self.weightvector, x)
        		return True
        	elif predictedY == 1 and y == -1:
        		self.weightvector = np.subtract(self.weightvector, x)
        		return True
        else:
        	return False
		
    def train(self):
        """
        Train the perceptron until convergence.

        To iterate through all of the data points once (a single epoch), you can
        do:
            for x, y in self.get_data_and_monitor(self):
                ...

        get_data_and_monitor yields data points one at a time. It also takes the
        perceptron as an argument so that it can monitor performance and display
        graphics in between yielding data points.
        """
        "*** YOUR CODE HERE ***"
        updated = False
        for x, y in self.get_data_and_monitor(self):
        	if self.update(x, y):
        		updated = True
        	else:
        		pass
        if updated:
        	self.train() 





