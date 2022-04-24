import numpy as np


def activation_function(x):
    if x < 0:
        return -1
    return 1


class Matrix:

    @staticmethod
    def matrix_vector_multiplication(matrix, vector):
        return matrix.dot(vector)
    
    @staticmethod
    def clear_diagonal(matrix):
        np.fill_diagonal(matrix, 0)
        return matrix
    
    @staticmethod
    def outer_product(pattern):
        return np.outer(pattern, pattern)
    
    @staticmethod
    def add_matrices(matrix1, matrix2):
        return matrix1 + matrix2
    
    
class HopfieldNetwork:
    
    def __init__(self, dimension):
        self.weight_matrix = np.zeros((dimension, dimension))

    def train(self, pattern):
        pattern_bipolar = HopfieldNetwork.transform(pattern)
        
        pattern_weight_matrix = Matrix.outer_product(pattern_bipolar)
        
        pattern_weight_matrix = Matrix.clear_diagonal(pattern_weight_matrix)
        
# porque somar matrizes?
        self.weight_matrix = Matrix.add_matrices(self.weight_matrix, pattern_weight_matrix)
        
        
    def recall(self, pattern):
        pattern_bipolar = HopfieldNetwork.transform(pattern)
      
        result = Matrix.matrix_vector_multiplication(self.weight_matrix, pattern_bipolar)
        result = np.array([activation_function(x) for x in result])
        result = HopfieldNetwork.re_transform(result)
        
        print(result)

    @staticmethod
    def transform(pattern):
        return np.where(pattern == 0, -1, pattern)
    
    @staticmethod
    def re_transform(pattern):
        return np.where(pattern == -1, 0, pattern)


if __name__ == '__main__':
    
    network = HopfieldNetwork(9)
    network.train(np.array([1, 1, 1, 1, 0, 0, 1, 1, 1]))
    network.train(np.array([1, 1, 1, 0, 1, 0, 0, 1, 0]))
    network.recall(np.array([1, 0, 1, 0, 1, 0, 0, 1, 0]))
    
