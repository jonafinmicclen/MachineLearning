import numpy as np

class LinearModel:

    def test(self,featureMatrix,outcomeVector):

        predictions = self.predict(featureMatrix)
        absoluteError = np.absolute(outcomeVector-predictions)
        self.absoluteError = absoluteError/np.size(outcomeVector)
        return f'Model testing complete current absolute error is {self.absoluteError}.'

    def report_model_status(self):
        return f'{self.convergency_status}\nModel contains {self.number_of_features} features.\nCurrent weights for model are {self.weights}.'
    
    def fit(self, featureMatrix, outputVector, learningRate, accuracyGoal, maxIterations):
        
        #Check that number of features attribute is not yet created
        if hasattr(self, 'number_of_features'):
            raise Exception('Fit method not available for pre-trained models')
        
        #Get number of features from width of featureMatrix
        self.number_of_features = np.shape(featureMatrix)[1]
        
        #Create vector of all weights
        self.weights = np.matrix(np.random.rand(self.number_of_features))
        
        #Adam implementation
        m = np.zeros_like(self.weights)
        v = np.zeros_like(self.weights)
        beta1 = 0.9  # Exponential decay rates for moment estimates
        beta2 = 0.999
        epsilon = 1e-8  # Small constant to prevent division by zero

        iterations = 1
        
        while True:
            
            #Calculate partial derivatives
            try:
                dLdw = -2 * (outputVector - featureMatrix*self.weights.T).T * featureMatrix
            except:
                return f'Error in gradient calculation, model did not converge on iteration {iterations}.'
            
            #Calculate new average gradient
            current_avg_gradient = np.mean(np.absolute(dLdw))
            
            #Test stop conditions
            if iterations > maxIterations:
                self.convergency_status = f'Non converging after {iterations} iterations.'
                return 'Model did not converge'
            
            if current_avg_gradient < accuracyGoal:
                self.convergency_status = f'Converged after {iterations} iterations.'
                return self.convergency_status
            
            # Adam update rules
            m = beta1 * m + (1 - beta1) * dLdw
            v = beta2 * v + (1 - beta2) * (dLdw ** 2)
            m_hat = m / (1 - beta1 ** iterations)
            v_hat = v / (1 - beta2 ** iterations)

            #Update weights
            self.weights -= learningRate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            iterations += 1
            
    #Make prediction using the model can also take matrix for multiple predictions
    def predict(self, featureMatrix):
        return featureMatrix * self.weights.T