import tensorflow as tf
import numpy as np

print("GPUs found :", tf.config.experimental.list_physical_devices("GPU"))
      
class Simulator :
    """
    Simulate a fluid using from idea of Jos Stam in the paper "Stable fluid".
    This class represent the high level opperation needed of fluid simulation,
    Independant of the geometry (2d or 3d)
    """
    
    def __init__(self, shape, dx, nu, force, geometry, border_condition=1.):
        """
        Shape : the shape of the space : m,n or m,n,p
        dx : the elementary distance corresponding to one division
        nu : physical property of the fluid
        border_condition : a mask of where the speed should be zero
        force : the volumic force to apply to the fluid as a function of the speed
        """
        
        self.shape = shape
        self.dx = dx
        self.nu = nu
        self.border_condition = border_condition
        self.force = force
        self.w = tf.constant(np.zeros(shape + (len(shape),), dtype=np.float32))

        self.geo = geometry
        
        self.error_max_diff = tf.constant(1e-5*np.prod(shape), dtype=tf.float32)
        
        #helper tensors for w3 to w4
        self.u = tf.Variable(np.zeros(shape), dtype=tf.float32)        
        self.error_max_lap = tf.constant(1e-4*np.prod(shape), dtype=tf.float32)
        self.r = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.p = tf.Variable(self.r, dtype=tf.float32)
        self.rsold = tf.Variable(0, dtype=tf.float32)
        
    def _compute_w1(self, w0, dt):
        """
        Applying the forces to the fluid.
        """
        return w0 + dt*self.force(w0)
        
    def _compute_w2(self, w1, dt):
        """
        advection.
        """
        return self.geo.apply_speed(dt, w1, w1)
    
    def _conjgrad_diff_step(self, r,x,p,rsold,nu,dt) :
        """
        Step function for the conjugate gradient descent solver of the diffusion.
        """
        Ap = p - dt*nu*self.geo.lap_vec(p)
        tmp = tf.reduce_sum(p*Ap)
        alpha = rsold / tmp
        x.assign(x + alpha * p)
        r.assign(r - alpha * Ap)
        rsnew = tf.reduce_sum(r*r)
        p.assign(r + (rsnew / rsold) * p)
        rsold.assign(rsnew)

    def _conjgrad_diff(self, w2, nu, dt, error_max) :
        """
        Solver for the diffusion with conjugate gradient descent
        """
        x = tf.Variable(w2)
        r = tf.Variable(w2 - x + dt*nu*self.geo.lap_vec(x))
        p = tf.Variable(r)
        rsold = tf.Variable(tf.reduce_sum(r*r))
        for i in range(200) :
            self._conjgrad_diff_step(r,x,p,rsold,nu,dt)
            if  tf.sqrt(rsold) < error_max :
                break
        return x     
    
    def _compute_w3(self, w2, dt):
        """
        Diffusion of the speed
        solving (I - dt * nu * lap ) v = w2
        """
        v = self._conjgrad_diff(w2, self.nu, dt, self.error_max_diff)
        diff = v-dt*self.nu*self.geo.lap_vec(w2)-w2
        loss = tf.reduce_sum(diff*diff)
        return v.numpy()
        

    def _conjgrad_lap_step(self,r,x,p,rsold) :
        """
        Helper function for the conjugate gradient descent solver of the poisson equation
        """
        Ap = self.geo.lap(p)
        alpha = rsold / tf.reduce_sum(p*Ap)
        x.assign(x + alpha * p)
        r.assign(r - alpha * Ap)
        rsnew = tf.reduce_sum(r*r)
        p.assign(r + (rsnew / rsold) * p)
        rsold.assign(rsnew) 

    @tf.function
    def _conjgrad_lap(self,b,x,r,p,rsold) :
        """
        Conjugate gradient descent to solve the poisson equation.
        For better speed we use @tf.function. But all the code inside this 
        function must be compatible with tensorflow 2.0 graph execution.
        """
        r.assign(b-self.geo.lap(x))
        p.assign(r)
        rsold.assign(tf.reduce_sum(r*r))             
        for i in tf.range(200) :
            self._conjgrad_lap_step(r,x,p,rsold)
            if  tf.sqrt(rsold) < self.error_max_lap :
                break
                
    def _compute_w4(self, w3, dt):
        """
        Ensuring the divergence of the speed is zero : conservation of matter
        Need to solve the corresponding poisson equation
        """
        target = self.geo.div(w3)
        self._conjgrad_lap(target, self.u, self.r, self.p, self.rsold)
        diff = self.geo.lap(self.u)-target
        loss = tf.reduce_sum(diff*diff)
        loss2 = tf.reduce_sum((diff*diff)[0]+(diff*diff)[-1])
        grad_u = self.geo.grad(self.u).numpy()
        return  w3 - grad_u

    def time_step(self, dt, debug=False) :
        w1 = self._compute_w1(self.w, dt)
        w2 = self._compute_w2(w1, dt)
        w3 = self._compute_w3(w2, dt)*self.border_condition
        w4 = self._compute_w4(w3, dt)*self.border_condition
        self.w = w4
