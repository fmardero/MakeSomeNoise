import numpy as np
from numba import jit
from scipy.ndimage import rotate
from skimage.transform import resize

class MakeSomeNoise:
    def __init__(self, image, return_img = False):
        self.image = image
        self.return_img = return_img 
    
    
    def _normalSelection(self, mean = 0, stddev = None, value_range = [0, 1]):
        '''
            Returns a value included in "value_range".
            The value is computed as the absolute value obtained from a normal distribution
            with "mean" and "stddev" parameters.
            
            'mean' : normal distribution mean
            'stddev' : normal distribution standard deviation
            'value_range' : value range of normal distribution output
        '''
        
        if stddev is None:
            stddev = value_range[1] - value_range[0]

        val = np.abs(np.random.normal(mean, stddev))
        if val > value_range[1]:
            return int(value_range[1])
        elif val < value_range[0]:
            return int(value_range[0])
        else:
            return int(val)
                    
            
    def _gammaCorrection(self, image, gamma = 1.0):  
        '''
            Gamma correction of the brightness image.
            
            'image' : image to be modified
            'gamma' : gamma parameter
        '''
        
        inverse_gamma = 1.0 / gamma
        return np.asarray(255 * (image/255.0)**(inverse_gamma)).astype('uint8')
    
    
    def randomRotation(self, max_rot = 180, return_img = False):
        '''
            Performs a rotation of the input image.
            The angle is choosen using a normal distribution.
            
            
            'image' : image than needs to be rotated (must be a [y, x] array)
            'max_rot' : "maximum" rotation angle 
                        (normal distribution returns higher values with a probability of 0.006%)
        '''
        
        try:
            rot = np.random.normal(0, max_rot/2.)
            self.image = rotate(self.image, rot, mode = 'reflect', reshape = False)
            
            if (self.return_img == True) | (return_img == True):
                return self.image
        except:
            raise ValueError('Invalid image.')
    
    
    def randomCropping(self, starting_prop = [None, None], return_img = False):
        '''
            Performs random cropping of the image based on one focus point.
            Focus point can be choosen initializing the 'starting_prop' variable 
            referring to the proportions of the picture.
            Based on focus point the cropping is done using a normal distribution
            with standard deviation equal to half of the image width.
            
            'starting_prop' : centroid of the random cropping expressed as proportion of [y, x] dimensions
        '''
        
        # Porportions are robust against changing image dimension
        # Starting proportion must be set as x axis ad y axis respectively
        
        # Transform from x prop to x value
        if starting_prop[0] in [None, 0]:
            x = self.image.shape[1] // 2
        else: 
            x = int(starting_prop[0] * self.image.shape[1])    

        # Transform from y prop to y value
        if starting_prop[1] in [None, 0]:
            y = self.image.shape[0] // 2
        else: 
            y = int((1 - starting_prop[1]) * self.image.shape[0])

        try:
            y_max = 0
            y_min = 0
            y_stddev = 1.6 * max(y, self.image.shape[0] - y)
            
            x_max = 0
            x_min = 0
            x_stddev = 1.6 * max(x, self.image.shape[1] - x)
            
            # Cropped image needs to be at least half of the original image
            while ((y_max - y_min) < self.image.shape[0]/2) | ((x_max - x_min) < self.image.shape[1]/2):
                y_min = y - self._normalSelection(0, stddev = y_stddev, value_range = [0, y])
                y_max = self._normalSelection(y, stddev = y_stddev, value_range = [y, self.image.shape[0]])
                
                x_min = x - self._normalSelection(0, stddev = x_stddev, value_range = [0, x])
                x_max = self._normalSelection(x, stddev = x_stddev, value_range = [x, self.image.shape[1]])

            self.image = self.image[y_min:y_max, x_min:x_max]
            
            if (self.return_img == True) | (return_img == True):
                return self.image
        except:
            raise ValueError('Invalid Image.')


    def randomBrightness(self, return_img = False):
        '''
            Change image brightness randomly from a normal distribution.
        '''
        
        gamma = np.abs(np.random.normal(1, 1))
        try:
            self.image = self._gammaCorrection(image = self.image, gamma = gamma)
            if (self.return_img == True) | (return_img == True):
                return self.image
        except:
            raise ValueError('Invalid Image.')
            
            
    def randomNoise(self, noise_factor = 0.01, return_img = False):
        '''
            Add random noise to an image.
            
            'noise_factor' : factor applied to the noise matrix
        '''
        
        try:
            self.image = np.asarray(self.image + noise_factor * np.random.normal(0, 255, self.image.shape) , dtype = 'uint8')
            if (self.return_img == True) | (return_img == True):
                return self.image
        except:
            raise ValueError('Invalid Image.')
            
            
    def padImage(self, new_shape = (None, None), random_noise = True, return_img = False):
        '''
            Rescale image to "new_shape". If the image has not the same proportion pad will be added.
            The type of padded can be black or random ("random_noise" = True).
            
            'new_shape' : new image shape
            'random_noise' : True if the padding will be random
        '''
        
        try:
            comparison = [new_shape[i] < self.image.shape[i] for i in [0, 1]]
            if any(comparison):
                print('The image will rescaled because it is too wide.')
                
                scale_y = float(new_shape[0]) / self.image.shape[0]
                scale_x = float(new_shape[1]) / self.image.shape[1]
                rescaling_factor = min(scale_y, scale_x)
                
                if scale_y < scale_x:
                    new_y = new_shape[0] 
                    new_x = int(rescaling_factor * self.image.shape[1])        
                else:
                    new_y = int(rescaling_factor * self.image.shape[0])
                    new_x = new_shape[1] 
                
                self.image = (255. * resize(self.image, (new_y, new_x))).astype('uint8')
            
            pad_y = (new_shape[0] - self.image.shape[0]) // 2
            pad_x = (new_shape[1] - self.image.shape[1]) // 2
            
            npad = ((pad_y, pad_y + (pad_y % new_shape[0] > 0)),
                    (pad_x, pad_x + (pad_x % new_shape[1] > 0)), 
                    (0, 0))
            
            try:
                if random_noise:
                    def _pad_random(vector, pad_width, iaxis, kwargs):
                        a, b = np.random.randint(0, 255, size = 2)
                        if (pad_width[0] > 0) & (pad_width[1] > 0):
                            vector[:pad_width[0]] = np.abs(np.random.normal(0, 255/3, size = pad_width[0]))
                            vector[-pad_width[1]:] = np.abs(np.random.normal(0, 255/3, size = pad_width[1]))
                        return vector
                    
                    self.image = np.pad(self.image, pad_width = npad, mode = _pad_random).astype('uint8')
                    
                else:
                    self.image = np.pad(self.image, pad_width = npad, mode = 'constant', constant_values = 0)
                    
                if (self.return_img == True) | (return_img == True):
                    return self.image
            except:
                raise ValueError('Invalid Image.')
        except:
            raise ValueError('Invalid \'new_shape\' parameter value.')
    
    
    def grayScale(self, return_img = False):
        '''
            Convert image to grayscale
        '''
        
        self.image = np.asarray(np.average(self.image, axis = -1), dtype='uint8')
        if (self.return_img == True) | (return_img == True):
            return self.image
        
        
    @jit(nopython=True)
    def randomLightSource(self, return_img = False):
        '''
            Adds a random light source with gaussian diffusion.
        '''
        
        y_max = self.image.shape[0]
        x_max = self.image.shape[1]
        # coordinates fromat is (y, x)
        source = (np.random.randint(low = 0, high = y_max), 
                  np.random.randint(low = 0, high = x_max))

        # inverse-gamma matrix
        inverse_gamma = np.ones(self.image.shape)

        # gamma limits
        gamma_max = 2.
        gamma_min = 0.01

        # Euclidian distance computing function
        def _dist_calc(point1, point2):
            point1 = np.asarray(point1)
            point2 = np.asarray(point2)
            return np.sqrt(np.sum((point1 - point2) ** 2))

        # Distance beetween the source and the farthest image point
        dist_max = 0
        edges = [(0,0), (y_max, 0), (0, x_max), (y_max, x_max)]
        for edge in edges:
            dist_max = max(dist_max, _dist_calc(edge, source))

        # Compute gaussian function "variance" parameter
        A = - dist_max**2
        B = 2 * np.log(gamma_min / gamma_max)
        var = (A/B)

        for xy in np.ndindex(self.image.shape[:2]):
            # euclidian distance between point and source
            dist = _dist_calc(xy, source)

            # compute gamma using the gaussian function
            gamma_pt = gamma_max * np.exp(-dist ** 2 / (2 * var))
            inverse_gamma[xy] /= gamma_pt

        # Modify image brightness based on the inverse-gamma matrix
        self.image = np.asarray(255 * np.power(self.image/255.0, inverse_gamma)).astype('uint8')
        
        if (self.return_img == True) | (return_img == True):
            return self.image
    
