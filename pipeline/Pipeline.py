from sklearn.preprocessing import StandardScaler
import numpy as np

class Pipeline():
    #We use weight and height data to predict gender
    def __init__(self):
        
        self.train_X = np.array([[161,57],
                                [181,88],
                                [190,95],
                                [156,54],
                                [162,58],
                                [170,60],
                                [179,75],
                                [178,74]])

        self.train_X = StandardScaler().fit_transform(self.train_X)

        self.train_Y = np.array([[0],
                                [1],
                                [1],
                                [0],
                                [0],
                                [0],
                                [1],
                                [1]])
                            
        self.test_X = np.array([[165,57],
                            [194,92],
                            [185,78],
                            [157,56],
                            [162,60]])

        self.test_X = StandardScaler().fit_transform(self.test_X)

        return self.train_X, self.train_Y, self.test_X