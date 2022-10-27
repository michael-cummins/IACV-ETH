# Note: You are not allowed to import additional python packages except NumPy
import numpy as np
from scipy.linalg import svd

K = 16

def reconstruct_image(image):
  rgb = np.reshape(image, (96, 96, 3))
  return rgb

def vectorise_image(image):
  return np.reshape(image, (1,27648))

class ImageCompressor:
    # This class is responsible to i) learn the codebook given the training images
    # and ii) compress an input image using the learnt codebook.
    def __init__(self):
        # You can modify the init function to add / remove some fields
        self.mean_image = np.array([])
        self.principal_components = np.array([])
        
    def get_codebook(self):
        # This function should return all information needed for compression
        # as a single numpy array
        
        # TODO: Modify this according to you algorithm
        mean_image_re = np.reshape(self.mean_image, (1,-1))
        # principal_components_re = np.reshape(self.principal_components, (1, -1))
        principal_components_re = self.principal_components.T
        codebook = np.concatenate((mean_image_re, principal_components_re), 0)
        print(codebook.shape)
        return codebook

    def train(self, train_images):
        # Given a list of training images as input, this function should learn the 
        # codebook which will then be used for compression
        # ******************************* TODO: Implement this ***********************************************#
        images = [vectorise_image(image) for image in train_images]
        X_m = np.vstack(images)
        self.mean_image = X_m.mean(0) # (27648,)
        X = X_m - self.mean_image
        X = X.T # image 0 = X[:,0] # ((27648, 100))
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        U = np.array(u)
        U = U[:, :K] # (27648, K)
        self.principal_components = np.float16(U)
        self.mean_image = np.float16(self.mean_image)

    def compress(self, test_image):
        # Given a test image, this function should return the compressed representation of the image
        # ******************************* TODO: Implement this ***********************************************#
        test_image = vectorise_image(test_image)
        im = test_image - self.mean_image
        test_image_compressed = self.principal_components.T @ im.T
        return np.float16(test_image_compressed) # (20,1)


class ImageReconstructor:
    # This class is used on the client side to reconstruct the compressed images.
    def __init__(self, codebook):
        # The codebook learnt by the ImageCompressor is passed as input when
        # initializing the ImageReconstructor
        self.mean_image = np.array(codebook[0]) # (27648,)
        self.mean_image = np.reshape(self.mean_image, (27648,1))
        self.principal_components = np.array(codebook[1:]) # (20, 27648)

    def reconstruct(self, test_image_compressed):
        # Given a compressed test image, this function should reconstruct the original image
        # ******************************* TODO: Implement this ***********************************************#
        test_image_recon = (self.principal_components.T @ test_image_compressed)
        test_image_recon = test_image_recon + self.mean_image
        test_image_recon = test_image_recon.astype(int) # column
        rec = reconstruct_image(test_image_recon)

        # Clean Image - Gives 0 reconstruction error
        tresh = 120
        tresh2 = 150
        tresh3 = 150
        tresh4 = 100
        
        # check each rgb componenet of each pixel
        # get indices of pixels that are closest to respective colour
        white = np.logical_and(rec[:,:,0]>tresh,np.logical_and(rec[:,:,1]>tresh,rec[:,:,2]>tresh)) 
        green = np.logical_and(rec[:,:,0]<tresh2,np.logical_and(rec[:,:,1]>tresh2,rec[:,:,2]<tresh2))
        red = np.logical_and(rec[:,:,0]>tresh3,np.logical_and(rec[:,:,1]<tresh3,rec[:,:,2]<tresh3)) 
        black = np.logical_and(rec[:,:,0]<tresh4,np.logical_and(rec[:,:,1]<tresh4,rec[:,:,2]<tresh4))
        # white = (rec[:,:,0].all() >= tresh).all() and (rec[:,:,1].all() >= tresh).all() and (rec[:,:,2].all() >= tresh).all()

        # set pixels to closest colour
        rec[green,:] = [0,255,0]
        rec[red,:] = [255, 0,0]
        rec[black,:] = [0, 0,0]
        rec[white,:] = [255,255,255]

        return rec