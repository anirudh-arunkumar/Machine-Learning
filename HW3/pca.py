import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio


class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) ->None:
        """		
		Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
		You may use the numpy.linalg.svd function
		Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
		corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose
		
		Hint: np.linalg.svd by default returns the transpose of V
		      Make sure you remember to first center your data by subtracting the mean of each feature.
		
		Args:
		    X: (N,D) numpy array corresponding to a dataset
		
		Return:
		    None
		
		Set:
		    self.U: (N, min(N,D)) numpy array
		    self.S: (min(N,D), ) numpy array
		    self.V: (min(N,D), D) numpy array
		"""
        # raise NotImplementedError
        adj = X - np.mean(X, axis=0)
        U, S, V = np.linalg.svd(adj)
        U = U[:, :min(X.shape[1], X.shape[0])]
        self.U = U
        self.S = S
        self.V = V

    def transform(self, data: np.ndarray, K: int=2) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that final data (X_new) has K features (columns)
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    K: int value for number of columns to be kept
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
        # raise NotImplementedError
        u = self.U[:, :K]
        s = self.S[:K]
        v = self.V[:K, :].T
        US = u * s
        return US

    def transform_rv(self, data: np.ndarray, retained_variance: float=0.99
        ) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
		in X_new with K features
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    retained_variance: float value for amount of variance to be retained
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
		           to be kept to ensure retained variance value is retained_variance
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
        # raise NotImplementedError
        adj = data - np.mean(data, axis=0)
        cumulative = np.cumsum(self.S**2)
        sm = np.sum(self.S**2)
        var = cumulative / sm
        k = np.argmax(var >= retained_variance) + 1
        v = self.V[:k, :].T
        X_new = np.dot(adj, v)
        return X_new

    def get_V(self) ->np.ndarray:
        """		
		Getter function for value of V
		"""
        # raise NotImplementedError
        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) ->None:
        """		
		You have to plot three different scatterplots (2D and 3D for strongest two features and 2D for two random features) for this function.
		For plotting the 2D scatterplots, use your PCA implementation to reduce the dataset to only 2 (strongest and later random) features.
		You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
		Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using plotly.
		Hint: Refer to https://plotly.com/python/line-and-scatter/ for making scatter plots with plotly.
		Hint: We recommend converting the data into a pandas dataframe before plotting it. Refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html for more details.
		
		Args:
		    xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
		    ytrain: (N,) numpy array, the true labels
		
		Return: None
		"""
        second_pca = PCA()
        third_pca = PCA()
        second_reduce = second_pca.transform(data=second_pca.fit(X), K=2)
        third_reduce = third_pca.transform(data=third_pca.fit(X), K=3)
        
        second_data_body = {
            'Feature 1':second_reduce[:, 0], 
            'Feature 2':second_reduce[:, 1], 
            'label':y
        }
        second_dataframe = pd.DataFrame(second_data_body)
        second_figure = px.scatter(
            second_dataframe, 
            color='label', 
            x='Feature 1', 
            y='Feature 2', 
            title=fig_title
        )
        second_figure.show()

        third_data_body = {
            'Feature 1':third_reduce[:, 0], 
            'Feature 2':third_reduce[:, 1], 
            'Feature 3':third_reduce[:, 2], 
            'label':y
        }
        third_dataframe = pd.DataFrame(third_data_body)
        third_figure = px.scatter_3d(
            third_dataframe, 
            color='label', 
            x='Feature 1', 
            y='Feature 2', 
            z='Feature 3', 
            title=fig_title
        )
        third_figure.show()
        # raise NotImplementedError
        
