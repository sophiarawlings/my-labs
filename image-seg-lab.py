# image_segmentation.py
"""Volume 1: Image Segmentation.
<Name> Sophia Rawlings
<Class> Math 345 Section 3
<Date> November 5th 2019
"""

import numpy as np
from scipy import sparse
from scipy import linalg as la
from scipy.sparse import linalg
from imageio import imread
from matplotlib import pyplot as plt

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    D = np.zeros_like(A)    #makes a matrix to find the Laplacian matrix
    b = A.sum(axis = 1)
    for i in range(0,len(A)):   #Makes each entry of the D matrix
        D[i][i] = b[i]
    L = D - A
    return L


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    L = laplacian(A)
    eigs = la.eig(L)[0]
    eigs = np.real(eigs)    #finds the real eigenvalues
    lili = []
    connected = 0
    alge_con = 0 
    for i in eigs:  #finds the connectedness of the matrix
        lili.append(i)
        if i < tol:
            i = 0
            connected += 1
    lili.sort()
    alge_con = lili[1]
    return connected, alge_con


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        self.filename = filename
        self.image = imread(filename)
        self.scaled = self.image / 255
        if len(self.image.shape) == 3:  #finds attributes for color matrices
            self.m,self.n,self.z = self.image.shape
            self.brightness = self.scaled.mean(axis = 2)
            self.bright = np.ravel(self.brightness)
        else:                           #finds attributes for gray matrices
            self.m, self.n = self.image.shape
            self.z = 0
            self.bright = np.ravel(self.scaled)

    # Problem 3
    def show_original(self):
        """Display the original image."""
        if self.z == 3: #plots color pictures
            plt.imshow(self.image)
        else:   #plots gray scale pictures
            plt.imshow(self.image, cmap = "gray")

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        mn = self.m*self.n
        A = sparse.lil_matrix((mn,mn))
        D = np.zeros(mn)
        for i in range(0, mn):  #constructs A matrix
            index, weight = get_neighbors(i,r,self.m,self.n)
            b_diff = abs(self.bright[i] - self.bright[index])
            b_diff = b_diff/sigma_B2
            weight = weight/sigma_X2
            A[i,index] = np.exp(-b_diff-weight)
        summs = A.sum(axis =1)
        for j in range(0,mn):   #constructs D matrix
            D[j] = summs[j]
        #A = sparse.csc_matrix(A)
        return A.tocsc(),D

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        L = sparse.csgraph.laplacian(A) #finds Laplacian
        Dsqr = D**(-1/2)
        Dhalf = sparse.diags(Dsqr)
        DLD = Dhalf @ L @ Dhalf
        eigs = sparse.linalg.eigsh(DLD, which = "SM", k=2)
        eigen = eigs[1][:,1].reshape((self.m,self.n))
        mask = eigen > 0 #finds the mask using eigenvalues
        return mask

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A,D = self.adjacency(r,sigma_B,sigma_X)
        mask = self.cut(A,D)
        if self.z == 3: #plots color pictures with masks
            mask = np.dstack((mask,mask,mask))
            img1 = self.scaled *mask
            img2 = self.scaled * ~mask
            plt.subplot(131)
            plt.imshow(self.image)
            plt.subplot(132)
            plt.imshow(img1)
            plt.subplot(133)
            plt.imshow(img2)
        else:           #plots gray scale pictures with masks
            img1 = self.scaled *mask
            img2 = self.scaled * ~mask
            plt.subplot(131)
            plt.imshow(self.image, cmap = 'gray')
            plt.subplot(132)
            plt.imshow(img1, cmap = 'gray')
            plt.subplot(133)
            plt.imshow(img2, cmap = 'gray')

# if __name__ == '__main__':
#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
