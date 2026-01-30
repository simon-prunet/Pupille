import numpy as np
import shapely 

class CloudOfPoints:
    def __init__(self, points_array):
        self.points = points_array  # points_array should be a Nx2 numpy array
        self.num_points = points_array.shape[0]

    def apply_transformation(self, matrix):
        """Apply a 2D affine transformation defined by a 2x2 matrix."""
        transformed_points = self.points @ matrix.T
        return CloudOfPoints(transformed_points)

    def apply_vector_field(self, vector_field, inplace=False):
        """Apply a vector field to the points.
        
        vector_field: function that takes Nx2 array and returns Nx2 array of displacements
        """
        displacements = vector_field(self.points)
        new_points = self.points + displacements
        if inplace: 
            self.points = new_points
            return self
        else:   
            return CloudOfPoints(new_points)
    def filter_inside_shape(self, shape):
        """Keep only points inside the given shapely shape."""
        mask = np.array([shape.contains(shapely.geometry.Point(pt)) for pt in self.points])
        filtered_points = self.points[mask]
        return CloudOfPoints(filtered_points)
    
    
    def compute_density_map(self, n_pixels, padding=0.1):
        """Compute a 2D histogram (density map) of the points,
        using a band-limited approach and an non-uniform Fourier Transform.
        
        n_pixels: number of pixels along each axis
        """
        xpmin, ypmin = np.min(self.points, axis=0)
        xpmax, ypmax = np.max(self.points, axis=0)
        center = np.array([(xpmin + xpmax) / 2, (ypmin + ypmax) / 2])
        size = (1 + padding) * max(xpmax - xpmin, ypmax - ypmin)
        scaled_coords = (self.points - center) * 2.*np.pi / size -n_pixels/2 # To place origin at center of first pixel
        dummy = np.ones(self.num_points)
        f = finufft.nufft2d2(scaled_coords[:,0], scaled_coords[:,1], dummy, n_pixels, n_pixels, eps=1e-10, modorder=1, isign=-1)
        density_map = np.fft.ifft2(f)
        return(density_map)





class RandomCloudOfPoints(CloudOfPoints):
    def __init__(self, num_points, bounds):
        """Generate random points within given bounds.
        
        bounds: (xmin, xmax, ymin, ymax)
        """
        xmin, xmax, ymin, ymax = bounds
        points_array = np.random.uniform(low=[xmin, ymin], high=[xmax, ymax], size=(num_points, 2))
        super().__init__(points_array) 


