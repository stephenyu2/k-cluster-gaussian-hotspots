import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse

def plot_gaussian_ellipses(ax, model):
    for model_params in model.models:
        mean = model_params['mean']
        cov = model_params['cov']
        v, w = np.linalg.eigh(cov)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = Ellipse(mean, v[0], v[1], 180. + angle, edgecolor='black', lw=2, facecolor='none')
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

def plot_decision_boundaries(X, y, model):
    # Define bounds of the domain
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)

    # Create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)

    # Flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # Horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))

    # Predict the function value for the whole gid
    yhat = model.predict(grid)

    # Reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)

    # Plot the grid of x, y and z values as a surface
    fig, ax = plt.subplots()
    cmap = ListedColormap(['#AAAAFF', '#FFAAAA'])
    plt.contourf(xx, yy, zz, cmap=cmap, alpha=0.5)

    # Plot Gaussian hotspots
    plot_gaussian_ellipses(ax, model)

    # Create scatter plot for samples from each class
    for class_value in np.unique(y):
        # Get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # Create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap=cmap)

    plt.show()

def plot_decision_boundaries_subplot(X, y, model, ax):
    # Define bounds of the domain
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    xx, yy = np.meshgrid(x1grid, x2grid)
    grid = np.hstack((xx.flatten().reshape(-1, 1), yy.flatten().reshape(-1, 1)))
    yhat = model.predict(grid)
    zz = yhat.reshape(xx.shape)
    cmap = ListedColormap(['#AAAAFF', '#FFAAAA'])
    ax.contourf(xx, yy, zz, cmap=cmap, alpha=0.5)
    plot_gaussian_ellipses(ax, model)
    for class_value in np.unique(y):
        row_ix = np.where(y == class_value)
        ax.scatter(X[row_ix, 0], X[row_ix, 1], cmap=cmap)
    ax.set_title(f'K = {model.K}')
