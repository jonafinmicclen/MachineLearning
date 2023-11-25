import numpy as np

class LinearModel:
    
    def report_model_status(self):
        return f'{self.convergency_status}\nModel contains {self.number_of_features} features.\nCurrent weights for model are {self.weights}.'
    
    def fit(self, featureMatrix, outputVector):
        
        #Set default learning rate
        learningRate = 0.1
        
        #Check that number of features attribute is not yet created
        if hasattr(self, 'number_of_features'):
            Exception('Fit method not available for pre-trained models')
        
        #Get number of features from width of featureMatrix
        self.number_of_features = np.shape(featureMatrix)[1]
        
        #Create vector of all weights
        self.weights = np.matrix(np.linspace(0.1,0.1,self.number_of_features))
        
        total_of_all_avg_gradients = 0
        iterations = 1
        
        while True:
            
            #Calculate partial derivatives
            dLdw = -2 * (outputVector - featureMatrix*self.weights.T).T * featureMatrix
            
            #Adjust learning rate if diverging
            current_avg_gradient = np.mean(np.absolute(dLdw))
            if current_avg_gradient > total_of_all_avg_gradients/iterations:
                learningRate /= 2
            
            #Update all gradients sum placeholder
            total_of_all_avg_gradients += current_avg_gradient
            
            #Test stop conditions
            if iterations > 10000:
                self.convergency_status = f'Non converging after {iterations} iterations.'
                return 'Model did not converge'
            
            if current_avg_gradient < 0.001:
                self.convergency_status = f'Converged after {iterations} iterations.'
                return self.convergency_status
            
            #Update weights using the partial derivatives
            self.weights -= dLdw * learningRate
            
            iterations += 1
            
    #Make prediction using the model can also take matrix for multiple predictions
    def predict(self, inputVector):
        return inputVector * self.weights.T