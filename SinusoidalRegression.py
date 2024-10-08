import numpy as np

class SinusoidalRegression:
    
    def test(self,featureMatrix,outcomeVector,significanceLevel):
        
        if significanceLevel >= 1:
            raise ValueError(f'significance level {significanceLevel} must be between 0 and 1')
        
        predictions = self.predict(featureMatrix)
        self.test_results = predictions
        
        accuracy_count = 0
        insignificance_count = 0 
        
        for prediction, outcome in zip(predictions, outcomeVector):
            if prediction > 0.5+significanceLevel and outcome == 1 or prediction < 0.5-significanceLevel and outcome == 0:
                accuracy_count += 1
            if 0.5-significanceLevel<prediction<0.5+significanceLevel:
                insignificance_count +=1
                
        self.accuracy = accuracy_count/(len(predictions)-insignificance_count)
        self.frequency_of_non_null_prediction = 1 - insignificance_count/len(predictions)
        return f'Model testing complete,\n{round(self.frequency_of_non_null_prediction*100,2)}% of the inputs are useable to the {round(significanceLevel*200,2)}% significance level\nThe model predicted accurately {self.accuracy*100}% of the time.'
    
    def report_model_status(self):
        return f'{self.convergency_status}\nModel contains {self.number_of_features} features.\nCurrent weights a:{self.a}, b:{self.b}, c:{self.c}.\nAverage gradient dLdw is {self.average_gradients}.'
    
    def fit(self, featureMatrix, outputVector, learningRate, accuracyGoal, maxIterations):
        
        #Check that number of features attribute is not yet created
        if hasattr(self, 'number_of_features'):
            raise Exception('Fit method not available for pre-trained models')
        
        #Get number of features from width of featureMatrix
        self.number_of_features = np.shape(featureMatrix)[1]
        
        n_of_rows = len(outputVector)
        print(f'Number of rows {n_of_rows}.')
        
        #Create vector of all weights with random starting values
        self.a = np.random.rand()
        self.b = np.random.rand()
        self.c = np.random.rand()
        
        print(f'Random starting weights a:{self.a}, b:{self.b}, c:{self.c}')
        
        iterations = 1
        
        while True:
            
            #Calculate partial derivatives
            try:
                dLda = sum([-2 * np.sin(self.b * x) * (y - self.predict(x)) for x, y in zip(featureMatrix, outputVector)])/n_of_rows
                dLdb = sum([-2 * x * self.a * np.cos(self.b*x) * (y - self.predict(x)) for x, y in zip(featureMatrix, outputVector)])/n_of_rows
                dLdc = sum([-2 * (y - self.predict(x)) for x, y in zip(featureMatrix, outputVector)])/n_of_rows
            except:
                return f'Error in gradient calculation, model did not converge on iteration {iterations}.'
            
            #Calculate new average gradient
            current_avg_gradient = np.mean(np.absolute([dLda, dLdb, dLdc]))
            
            #Test stop conditions
            if iterations >= maxIterations:
                self.convergency_status = f'Non converging after {iterations} iterations.'
                self.average_gradients = current_avg_gradient
                return self.convergency_status
            
            if current_avg_gradient < accuracyGoal:
                self.convergency_status = f'Converged after {iterations} iterations.'
                self.average_gradients = current_avg_gradient
                return self.convergency_status
            
            self.a -= dLda * learningRate
            self.a -= dLdb * learningRate
            self.a -= dLdc * learningRate
            
            iterations += 1
    
    def predict(self,input):
        return self.a * np.sin(self.b * input) + self.c
        
