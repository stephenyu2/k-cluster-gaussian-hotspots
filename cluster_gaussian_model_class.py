from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import numpy as np
from bootstrap_gmm import bootstrap_gmm

class ClusterGaussianModel:
    def __init__(self, K, n_iterations = 1000):
        self.K = K  # Number of clusters per label
        self.n_iterations = n_iterations
        self.models = []  # To store the Gaussian parameters and labels

    def fit(self, X, y):
        labels = np.unique(y)
        for label in labels:
            # Filter data by current label
            X_label = X[y == label]
            # Apply the bootstrap GMM approach
            best_gmm = bootstrap_gmm(X_label, self.K)
            for i in range(self.K):
                mean = best_gmm.means_[i]
                cov = best_gmm.covariances_[i]
                weight = best_gmm.weights_[i]  # Adjust weight by total number of points
                
                # Store the model parameters and label
                self.models.append({'mean': mean, 'cov': cov, 'label': label, 'weight': weight}) 

    def predict_proba(self, X):
        # This will store the sum of probabilities for each class
        return_proba = []

        for point in X:
            
            # Calculate probabilities for each model and sum them by label
            prob_sums = {label: 0 for label in set(model['label'] for model in self.models)}
            total_prob_sum = 0
            
            for model in self.models:

                pdf = multivariate_normal(mean=model['mean'], cov=model['cov'])
                prob = pdf.pdf(point) * model['weight']
                prob_sums[model['label']] += prob
                prob_sums[model['label']] 
                total_prob_sum += prob
                
            prob_class_1 = prob_sums[1] / total_prob_sum if total_prob_sum > 0 else 0
            return_proba.append(prob_class_1) 
        
        return return_proba

    def predict(self, X):
        prob_class_1 = self.predict_proba(X)
        prob_class_1 = np.array(prob_class_1)
        # Classify based on 0.5 threshold
        return np.where(prob_class_1 > 0.5, 1, 0)
