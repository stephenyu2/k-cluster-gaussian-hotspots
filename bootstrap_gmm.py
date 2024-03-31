import numpy as np
from sklearn.mixture import GaussianMixture

def bootstrap_gmm(X, n_components=3, n_iterations=100, hard=True):
    best_bic = np.inf
    best_gmm = None
    best_labels = None
    no_improvement_count = 0  # Counter for iterations without improvement in BIC

    iteration = 0  # Initialize iteration counter
    while True:
        iteration += 1
        # Randomly select initial points for the means
        init_means = X[np.random.choice(X.shape[0], n_components, replace=False), :]
        
        # Create and fit the GMM
        gmm = GaussianMixture(n_components=n_components, init_params='random', means_init=init_means, random_state=0)
        gmm.fit(X)
        
        # Calculate BIC as the loss
        bic = gmm.bic(X)
        
        # Check for improvement
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_labels = gmm.predict(X)
            no_improvement_count = 0  # Reset counter after improvement
        else:
            no_improvement_count += 1  # Increment counter if no improvement
        
        # Check stopping condition
        if hard:
            if iteration >= n_iterations:  # Stop after fixed number of iterations
                break
        else:
            if no_improvement_count >= n_iterations:  # Stop if no improvement for n_iterations straight
                break

    # Return the best model's parameters and labels
    return best_gmm
