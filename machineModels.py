import random
from scipy.spatial import distance

data = {"food_type":[1, 2, 3, 4, 5], "age":[15, 20, 25, 30, 35 ]}


############################################
#######Machine learning model###############

def euc(a, b):
    """this  calulate the distance of value """
    return distance.euclidean(a, b)


class Model():

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        predictions = []
        # The empty list will be used to store prediction from the closest function 
        
        for row in X_test:
            # loop over the feature from X_test then passed looped row to the closest feature for prediction
            label = self.closest(row)
            predictions.append(label)
        return predictions
        
    def closest(self, row):
        """this function contain computation of predicting X_test """
        best_distance = euc(row, self.X_train[0])
        best_index = 0
        
        for i in range(1, len(self.X_train)):
            dis = euc(row, self.X_train[i])
            if dis < best_distance:
                best_distance = dis 
                best_index = i
            
        return  self.y_train[best_index]
        

        
m = Model()
m.fit(data["age"], data["food_type"])
print(m.predict([[33]]))

                     