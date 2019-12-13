import tensorflow as tf
import numpy as np

class geo2d_33conv :
    """
    Represents the different geometrical operation needed in 2 dimensions.
    This class is fully implemented in tensorflow 2.0 ans is gpu compatible
    Most operations are computed using 2d convolution. As most of the kernel are sparse, we should expect a huge gain in performace if the operations were better inplemented.  
    """
    def __init__(self, m,n) :
        """
        m,n the dimensions of the 2d space.
        """
        self.m = m
        self.n = n
        
        gi = np.array([[0,-1,0],[0,0,0],[0,+1,0]])/2
        gj = np.array([[0,0,0],[-1,0,1],[0,0,0]])/2
        grad_kernel = np.stack([gi,gj], -1)
        grad_kernel = np.expand_dims(grad_kernel, 2)
        self.grad_kernel = tf.constant(grad_kernel, dtype=tf.float32)

        fi = np.array([[0,-1,0],[0,0,0],[0,+1,0]], )/2
        fj = np.array([[0,0,0],[-1,0,1],[0,0,0]])/2
        div_kernel = np.stack([fi,fj], 2)
        div_kernel = np.expand_dims(div_kernel, -1)
        self.div_kernel = tf.constant(div_kernel, dtype=tf.float32)

        lap_vec_kernel = np.zeros((3,3,2,2))
        lap_vec_kernel[:,:,0,0] = np.array([[0,+1,0],[+1,-4.,+1],[0,+1,0]])
        lap_vec_kernel[:,:,1,1] = np.array([[0,+1,0],[+1,-4.,+1],[0,+1,0]])
        self.lap_vec_kernel = tf.constant(lap_vec_kernel, dtype=tf.float32)
        
        self.indices = tf.constant(np.indices((m,n)).swapaxes(0,2).swapaxes(0,1), dtype=tf.float32)
        self.clipping = tf.constant(np.stack([np.ones((m,n), dtype=np.int32)*(m-1),np.ones((m,n), dtype=np.int32)*(n-1)], axis=-1))
        
        
    def grad(self, u) :
        expended = tf.expand_dims(tf.expand_dims(u, 0), -1)
        return tf.nn.convolution(expended, self.grad_kernel, padding="SAME", strides=1)[0]
    
    def lap_vec(self, u) :
        expended = tf.expand_dims(u, 0)
        return tf.nn.convolution(expended, self.lap_vec_kernel, padding="SAME", strides=1)[0,:,:]
    
    def div(self, u) :
        expended = tf.expand_dims(u, 0)
        return tf.nn.convolution(expended, self.div_kernel, padding="SAME", strides=1)[0,:,:,0]    
    
    def lap(self, u) :
        """
        The scalar laplacian, implemented as the composition of our grad and div.
        Doing so allows to completly nullify the loss in the poisson equation.
        
        The other alternativ being to implement the lapacian as convolution with the corresponding 3*3 kernel.
        """
        expended = tf.expand_dims(tf.expand_dims(u, 0), -1)
        res_grad = tf.nn.convolution(expended, self.grad_kernel, padding="SAME", strides=1)
        return tf.nn.convolution(res_grad, self.div_kernel, padding="SAME", strides=1)[0,:,:,0]    
    
    def apply_speed(self, dt, speed, array) :
        """
        Function mostly used for the advection. "move" a vectorial field with respect to a speed field. 
        """
        #Position of point i,j at t-dt, shape : m,n,2 
        m,n = self.m,self.n
        indices = self.indices - dt*speed
        indices_floor = tf.math.floor(indices)
        frac = indices-indices_floor
        frac = [1-frac,frac]
        indices_floor = tf.dtypes.cast(indices_floor, tf.int32)
        indices_ceil = indices_floor + 1
        indices_ceil = tf.clip_by_value(indices_ceil, 0, self.clipping)
        indices_floor = tf.clip_by_value(indices_floor, 0, self.clipping)  
        indices = [indices_floor, indices_ceil] # shape : 2 * m,n,2
        res = tf.zeros((m,n,2))
        for i in range(2) :
            for j in range(2) :
                indices_to_take = tf.stack([indices[i][:,:,0], indices[j][:,:,1]], axis=-1)
                res = res + tf.expand_dims(frac[i][:,:,0] * frac[j][:,:,1], -1) * tf.gather_nd(array,indices_to_take)
        return res

class geo3d_13conv :
    """
    Represents the different geometrical operation needed in 3 dimensions.
    This class is fully implemented in tensorflow 2.0 ans is gpu compatible
    Most operations are computed using 3d convolution, with a 1*3 kernel, This make the kernel not to sparce and imprive the performance compared to a 3*3*3 sparce kernel.
    """
    def __init__(self, m,n,p) :
        """
        m,n,p the dimensions of the 3d space.
        """
        self.m, self.n, self.p = m,n,p
        grad_kernel_base = np.array([-1,0,1])/2
        grad_kernel_x = np.expand_dims(np.expand_dims(grad_kernel_base,-1),-1)
        grad_kernel_y = np.expand_dims(np.expand_dims(grad_kernel_base,-1),0)
        grad_kernel_z = np.expand_dims(np.expand_dims(grad_kernel_base,0),0)

        grad_kernel_x = np.expand_dims(np.expand_dims(grad_kernel_x,-1),-1)
        grad_kernel_y = np.expand_dims(np.expand_dims(grad_kernel_y,-1),-1)
        grad_kernel_z = np.expand_dims(np.expand_dims(grad_kernel_z,-1),-1)

        self.grad_kernel_x = tf.constant(grad_kernel_x, dtype=tf.float32)
        self.grad_kernel_y = tf.constant(grad_kernel_y, dtype=tf.float32)
        self.grad_kernel_z = tf.constant(grad_kernel_z, dtype=tf.float32)

        lap_base = np.array([1,-2,1])
        lap_base = np.stack([np.zeros(3,),lap_base,np.zeros(3,)],0)
        lap_kernel_x = np.stack([np.zeros((3,3)),lap_base,np.zeros((3,3))],0)
        lap_kernel_y = lap_kernel_x.swapaxes(1,2)
        lap_kernel_z = lap_kernel_x.swapaxes(0,2)
        lap_vec_kernel2 = lap_kernel_x + lap_kernel_y + lap_kernel_z
        lap_vec_kernel2 = np.expand_dims(np.expand_dims(lap_vec_kernel2,-1),-1)
        self.lap_vec_kernel = tf.constant(lap_vec_kernel2, dtype=tf.float32)
        
        self.clipping = tf.constant(np.stack([np.ones((m,n,p), dtype=np.int32)*(m-1),np.ones((m,n,p), dtype=np.int32)*(n-1), np.ones((m,n,p), dtype=np.int32)*(p-1)], axis=-1))
        
        self.indices = tf.constant(np.indices((m,n,p)).swapaxes(0,1).swapaxes(1,2).swapaxes(2,3), dtype=tf.float32)
        
    def grad(self, u) :
        expanded = tf.expand_dims(tf.expand_dims(u,0),-1)
        convol_x = tf.nn.conv3d(expanded, self.grad_kernel_x, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
        convol_y = tf.nn.conv3d(expanded, self.grad_kernel_y, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
        convol_z = tf.nn.conv3d(expanded, self.grad_kernel_z, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
        result = tf.stack([convol_x,convol_y,convol_z],-1)
        return result

    def div(self, u):
        expanded_x = tf.expand_dims(tf.expand_dims(u[:,:,:,0],0),-1)
        expanded_y = tf.expand_dims(tf.expand_dims(u[:,:,:,1],0),-1)
        expanded_z = tf.expand_dims(tf.expand_dims(u[:,:,:,2],0),-1)
        result = tf.nn.conv3d(expanded_x, self.grad_kernel_x , padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
        result += tf.nn.conv3d(expanded_y, self.grad_kernel_y, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
        result += tf.nn.conv3d(expanded_z, self.grad_kernel_z, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
        return result

    def lap_vec(self, u) :
        expanded_x = tf.expand_dims(tf.expand_dims(u[:,:,:,0],0),-1)
        expanded_y = tf.expand_dims(tf.expand_dims(u[:,:,:,1],0),-1)
        expanded_z = tf.expand_dims(tf.expand_dims(u[:,:,:,2],0),-1)

        lap_x = tf.nn.conv3d(expanded_x, self.lap_vec_kernel, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
        lap_y = tf.nn.conv3d(expanded_y, self.lap_vec_kernel, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
        lap_z = tf.nn.conv3d(expanded_z, self.lap_vec_kernel, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
        result = tf.stack([lap_x,lap_y,lap_z],-1)
        return result

    def lap(self, u) :
        """
        The scalar laplacian, implemented as the composition of our grad and div.
        Doing so allows to completly nullify the loss in the poisson equation.
        
        The other alternativ being to implement the lapacian as convolution with the corresponding 3*3 kernel.
        """
        return self.div(self.grad(u)) 

    
    def apply_speed(self, dt, speed, array) :
        """
        Function mostly used for the advection. "move" a vectorial field with respect to a speed field. 
        """
        m,n,p = self.m, self.n, self.p
        #Position of point i,j,k at t-dt, shape : m,n,p,3 
        w1 = tf.constant(speed, dtype=tf.float32)
        m,n,p = self.m,self.n,self.p
        indices = self.indices - dt*w1
        indices_floor = tf.math.floor(indices)
        frac = indices-indices_floor
        frac = [1-frac,frac]
        indices_floor = tf.dtypes.cast(indices_floor, tf.int32)
        indices_ceil = indices_floor + 1
        indices_ceil = tf.clip_by_value(indices_ceil, 0, self.clipping)
        indices_floor = tf.clip_by_value(indices_floor, 0, self.clipping)  
        indices = [indices_floor, indices_ceil] # shape : 2 m,n,p,3
        res = tf.zeros((m,n,p,3))
        for i in range(2) :
            for j in range(2) :
                for k in range(2) :
                    indices_to_take = tf.stack([indices[i][:,:,:,0], indices[j][:,:,:,1], indices[k][:,:,:,2]], axis=-1)
                    res = res + tf.expand_dims(frac[i][:,:,:,0] * frac[j][:,:,:,1] * frac[k][:,:,:,2], -1) * tf.gather_nd(array,indices_to_take)   
        return res