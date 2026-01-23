import numpy as np
import shapely
import shapely.geometry as geom
from shapely.ops import unary_union
from shapely.geometry import Polygon, box, Point, MultiPolygon
from shapely.affinity import rotate
from shapely.prepared import prep

# args = {'circular_resolution': 360, 'pupil_diameter': 800, 'obscuration_diameter': 112, 'spider_width': 40, 'spider_angle': 50.5,
#        'x_offset': 111}

class PupilGeometry:
    def __init__(self, **kwargs):
        self.pupil_diameter = kwargs.get('pupil_diameter', 800)
        self.obscuration_diameter = kwargs.get('obscuration_diameter', 112)
        self.spider_width = kwargs.get('spider_width', 4) 
        self.spider_angle = kwargs.get('spider_angle', 50.5)
        self.x_offset = kwargs.get('x_offset', 111.0/2)
        self.circular_resolution = kwargs.get('circular_resolution', 360)
        self.horizontal_offset = self.x_offset * np.array([1, -1, -1, 1])
        
    def create_pupil(self):
        outer_circle = Point(0, 0).buffer(self.pupil_diameter / 2, resolution=self.circular_resolution)
        if self.obscuration_diameter > 0:
            inner_circle = Point(0, 0).buffer(self.obscuration_diameter / 2, resolution=self.circular_resolution)
            pupil_shape = outer_circle.difference(inner_circle) 
        else:
            pupil_shape = outer_circle

        if self.spider_width > 0:
            spiders = []
            angles = [self.spider_angle, 180.-self.spider_angle, 180+self.spider_angle, -self.spider_angle]
            for i in range(4):
                angle = angles[i]
                x_offset = self.horizontal_offset[i]

                spider = Polygon([
                    (x_offset, -self.spider_width / 2),
                    (x_offset + self.pupil_diameter / 2 + 10, -self.spider_width / 2),
                    (x_offset + self.pupil_diameter / 2 + 10, self.spider_width / 2),
                    (x_offset, self.spider_width / 2)
                ])
                spider = rotate(spider, angle, origin=(x_offset,0))
                spiders.append(spider)

            spiders_union = unary_union(spiders)
            pupil_shape = pupil_shape.difference(spiders_union)
        return pupil_shape

    def get_area(self):
        pupil_shape = self.create_pupil()
        return pupil_shape.area

    def get_perimeter(self):
        pupil_shape = self.create_pupil()
        return pupil_shape.length
    def plot_pupil(self, outer_scale_factor=1.1):
        import matplotlib.pyplot as plt
        pupil_shape = self.create_pupil()
        plt.figure(figsize=(6,6))

        if isinstance(pupil_shape, MultiPolygon):
            for poly in pupil_shape.geoms:
                x, y = poly.exterior.xy
                plt.fill(x, y, color='gray')
        else:               
            x, y = pupil_shape.exterior.xy
            plt.fill(x, y, color='gray')
        plt.xlim(-self.pupil_diameter/2*outer_scale_factor, self.pupil_diameter/2*outer_scale_factor)
        plt.ylim(-self.pupil_diameter/2*outer_scale_factor, self.pupil_diameter/2*outer_scale_factor)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Pupil Geometry')
        plt.show()

    def create_pixel_weights(self, num_pixels, outer_scale_factor=1.1):
        pupil_shape = self.create_pupil()
        pupil_prepared = prep(pupil_shape)
        radius = (self.pupil_diameter / 2) * outer_scale_factor
        x = np.linspace(-radius, radius, num_pixels+1)
        y = np.linspace(-radius, radius, num_pixels+1)
        pixel_weights = np.zeros((num_pixels, num_pixels))

        for i in range(num_pixels):
            for j in range(num_pixels):
                pixel = box(x[i], y[j], x[i+1], y[j+1])
                if pupil_prepared.intersects(pixel):
                    pixel_weights[i, j] = pupil_shape.intersection(pixel).area / pixel.area
                else:
                    pixel_weights[i, j] = 0.0
        return x, y, pixel_weights
    
