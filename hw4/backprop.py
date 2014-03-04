"""
backprop.py

Calculates derivatives of loss function in a neural network, using the back-
propagation algorithm.
"""

import numpy as np

def backprop(tree, h, W, 
