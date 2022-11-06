import numpy as np
import random
import math

# Import maxflow (PyMaxflow library) which can be used to solve min-cut problem
import maxflow

# Set seeds for random generators to get reproducible results
random.seed(0)
np.random.seed(0)

def perform_min_cut(unary_potential_foreground, unary_potential_background, pairwise_potential):
    """
    We provide a simple fuction to perform min cut using PyMaxFlow library. You
    may use this function to implement your algorithm if you wish. Feel free to
    modify this function as desired, or implement your own function to perform
    min cut.  

    args:
        unary_potential_foreground - A single channel NumPy array specifying the
            source (foreground) unary potentials for each pixel in the image
        unary_potential_background - A single channel NumPy array specifying the
            sink (background) unary potentials for each pixel in the image
        pairwise_potential - A single channel NumPy array specifying the pairwise
            potentials. We assume a graph where each pixel in the image is 
            connected to its four neighbors (left, right, top, and bottom). 
            Furthermore, we assume that the pairwise potential for all these 4
            edges are same, and set to the value of pairwise_potential at that 
            pixel location
    """    
    
    # create graph
    maxflow_graph = maxflow.Graph[float]()
    
    # add a node for each pixel in the image
    nodeids = maxflow_graph.add_grid_nodes(unary_potential_foreground.shape[:2])

    # Add edges for pairwise potentials. We use 4 connectivety, i.e. each pixel 
    # is connected to its 4 neighbors (up, down, left, right). Also we assume
    # that pairwise potential for all these 4 edges are same
    # Feel free to change this if you wish
    maxflow_graph.add_grid_edges(nodeids, pairwise_potential)

    # Add unary potentials
    maxflow_graph.add_grid_tedges(nodeids, unary_potential_foreground, unary_potential_background)

    maxflow_graph.maxflow()
    
    # Get the segments of the nodes in the grid.
    mask_bg = maxflow_graph.get_grid_segments(nodeids)
    mask_fg = (1 - mask_bg.astype(np.uint8))* 255

    return mask_fg


class ImageSegmenter:
    def __init__(self):
        pass
    
    def segment_image(self, im_rgb, im_aux, im_box):
        # TODO: Modify this function to implement your algorithm
        im_mask = im_box
        luminance_im = relative_luminance(im_aux[:,:,0], im_aux[:,:,1], im_aux[:,:,2])
        sigma = 0.1
        
        for i in range(0,30):
           
            # Get foreground and background pixels
            foreground = luminance_im[np.nonzero(im_mask)]
            im_mask = np.ndarray.flatten(im_mask)
            background = np.delete(np.ndarray.flatten(luminance_im), np.nonzero(im_mask))

            # calculate expected instensity of foreground and background
            It = np.mean(foreground)
            Is = np.mean(background)
            
            # Perform simple min cut
            # Foreground potential
            unary_potential_foreground = unary_link(luminance_im, It, sigma)
           
            # Background potential 
            unary_potential_background = unary_link(luminance_im, Is, sigma)
            
            # Pairwise potential 
            pairwise_potential = pair_links(luminance_im, sigma)
            last_p = pairwise_potential[:,-1]
            last_p = np.reshape(last_p, (len(pairwise_potential[:,0]),1))
            pairwise_potential = np.hstack([pairwise_potential, last_p])

            # Perfirm min cut to get segmentation mask
            im_mask = perform_min_cut(unary_potential_foreground, unary_potential_background, 
                                    pairwise_potential)
                                    
            # Get rid of any "foreground" pixels outside of box
            im_mask = np.multiply(im_mask, im_box.astype(np.float32)/255)
        
        return im_mask


"""
helper functions
"""

def relative_luminance(r,g,b):
    """
    Using grey scale for luminance
    """
    return (0.2126*(r) + 0.7152*(g) + 0.0722*(b))

def unary_link(I, I_, sigma): return np.exp(-(np.absolute(I - I_)**2)/2*(sigma**2))

def pair_links(im, sigma):
    h_diff = np.diff(im)
    return nlink(h_diff, sigma)

def nlink(Ipq,sigma):
    return np.exp(-np.absolute(Ipq)/(2*sigma**2))

