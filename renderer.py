import tensorflow as tf
import numpy as np

class Renderer :
    """
    Render a 3d function : f(k,i,j) -> y > 0, using a ray tracing algorithm
    """
    
    def __init__(self, m, n, screen_dist) :
        """
        Parameters :
        m : the size of the first coordinate i, matrix convention of the screen 
        n : the size of the second coordinate j, matrix convention of the screen 
        """
        self.display_m,self.display_n = m, n
        self.indices = np.indices((self.display_m,self.display_n)).swapaxes(0,2).swapaxes(0,1)
        self.screen_dist = screen_dist
         
    def __call__(self, input_tensor, camera_position) :
        """
        Render a 3d function : f(k,i,j) -> y > 0, using a ray tracing algorithm.
        Render input_tensor from camera_position point of view, projected on a screen at screen_dist from camera_position
        
        Parameters :
        
        input_tensor : the 3d function.
        camera_position : the coordinate of the camera in the k,i,j basis (matrix convention) must be outside of input_tensor and preferably with a small i coordinate. 
        screen_dist : the distance from camera_position of the screen where the rendering is projected.
        """
        order = 1
        input_tensor = tf.pad(input_tensor, [[1,1],[1,1],[1,1]]).numpy()
        input_tensor[1,1:-2,1] = 1
        input_tensor[-2,1:-2,1] = 1
        input_tensor[1,1:-2,-2] = 1
        input_tensor[-2,1:-2,-2] = 1

        input_tensor[1:-2,1,1] = 1
        input_tensor[1:-2,-2,1] = 1
        input_tensor[1:-2,1,-2] = 1
        input_tensor[1:-2,-2,-2] = 1

        input_tensor[1,1,1:-2] = 1
        input_tensor[-2,1,1:-2] = 1
        input_tensor[1,-2,1:-2] = 1
        input_tensor[-2,-2,1:-2] = 1
        
        if abs(camera_position[2]) > abs(camera_position[0]) :
            camera_position = camera_position[::-1]
            input_tensor = np.swapaxes(input_tensor, 0, 2)
            order = -1
        p,m,n = input_tensor.shape
        center = (np.array([m,n,p])-1)/2
        c0 = center - camera_position
        c0 = c0 / np.linalg.norm(c0)
        ci = np.array([0,1,0])
        ci = ci - np.sum(ci*c0)*c0
        ci = ci / np.linalg.norm(ci)
        cj = np.cross(c0,ci)

        pixel_pos = 0
        pixel_pos = camera_position + c0*self.screen_dist
        pixel_pos = pixel_pos + np.reshape(ci,[1,1,3])*(self.indices[:,:,0:1] - (self.display_m-1) / 2) 
        pixel_pos = pixel_pos + np.reshape(cj,[1,1,3])*(self.indices[:,:,1:2] - (self.display_n-1) / 2) 

        #solving : camera_position[0] + (pixel_pos[:,:,0] - camera_position[0])*x = k
        batch_fact = []
        for k in range(p) :
            fact = (k - camera_position[0]) / (pixel_pos[:,:,0] - camera_position[0])
            batch_fact.append(fact)
        batch_indices = []
        #Using : new indicies = camera_position[1:] + (pixel_pos[1:] - camera_position[1:])*x
        max_clip = np.stack([np.ones((self.display_m,self.display_n))*(m-1),np.ones((self.display_m,self.display_n))*(n-1)], axis=-1)
        for x in batch_fact:
            cam_pos_reshaped = np.reshape(camera_position[1:],(1,1,2))
            intersect_indices = cam_pos_reshaped + (pixel_pos[:,:,1:] - cam_pos_reshaped)*np.reshape(x,(self.display_m,self.display_n,1))
            batch_indices.append(tf.clip_by_value(intersect_indices, 0, max_clip).numpy().astype(np.int))

        batch_indices = np.array(batch_indices)
        result = tf.gather_nd(input_tensor, batch_indices, batch_dims=1)
        return result.numpy().sum(axis = 0)[:,::order]

    