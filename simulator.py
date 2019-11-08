import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

gi = np.array([[0,-1,0],[0,0,0],[0,+1,0]])/2
gj = np.array([[0,0,0],[-1,0,1],[0,0,0]])/2
grad_kernel = np.stack([gi,gj], -1)
grad_kernel = np.expand_dims(grad_kernel, 2)
grad_kernel = tf.constant(grad_kernel)

fi = np.array([[0,-1,0],[0,0,0],[0,+1,0]], )/2
fj = np.array([[0,0,0],[-1,0,1],[0,0,0]])/2
div_kernel = np.stack([fi,fj], 2)
div_kernel = np.expand_dims(div_kernel, -1)
div_kernel = tf.constant(div_kernel)

lap_kernel = tf.constant(np.array([[0,+1,0],[+1,-4.,+1],[0,+1,0]]))
lap_kernel = tf.expand_dims(tf.expand_dims(lap_kernel, -1), -1)
lap_kernel = tf.constant(lap_kernel)

lap_vec_kernel = np.zeros((3,3,2,2))
lap_vec_kernel[:,:,0,0] = np.array([[0,+1,0],[+1,-4.,+1],[0,+1,0]])
lap_vec_kernel[:,:,1,1] = np.array([[0,+1,0],[+1,-4.,+1],[0,+1,0]])
lap_vec_kernel = tf.constant(lap_vec_kernel)

def grad(u) :
    expended = tf.expand_dims(tf.expand_dims(u, 0), -1)
    #print(expended.shape)
    return tf.nn.convolution(expended, grad_kernel, padding="SAME", strides=1)[0]
def lap(u) :
    expended = tf.expand_dims(tf.expand_dims(u, 0), -1)
    #print(expended.shape)
    return tf.nn.convolution(expended, lap_kernel, padding="SAME", strides=1)[0,:,:,0]
def lap_vec(u) :
    expended = tf.expand_dims(u, 0)
    #print(expended.shape)
    return tf.nn.convolution(expended, lap_vec_kernel, padding="SAME", strides=1)[0,:,:]
def div(u) :
    expended = tf.expand_dims(u, 0)
    #print(expended.shape)
    return tf.nn.convolution(expended, div_kernel, padding="SAME", strides=1)[0,:,:,0]

def take(a,indices_i, indices_j) :
    m,n = a.shape[:2]
    full_idices = n*indices_i + indices_j
    reshaped = a.reshape(m*n,2)
    res = np.take(reshaped,full_idices, axis=0)
    return res.reshape(m,n,2)

def solve_step_diff(u, dt, dx, nu, v, lr) :
    with tf.GradientTape() as tape :
        left = v - dt*nu*lap_vec(v)
        diff = left-u
        diff2 = diff*diff
        left_border = tf.math.reduce_sum(tf.math.square(v[:,0,1]))
        right_border = tf.math.reduce_sum(tf.math.square(v[:,-1,1]))
        top_border = tf.math.reduce_sum(tf.math.square(v[0,:,0]))
        bottom_border = tf.math.reduce_sum(tf.math.square(v[-1,:,0]))
        loss = tf.math.reduce_sum(diff2) +10*(left_border + right_border + top_border + bottom_border)
    grad_l = tape.gradient(loss,[v])[0]
    return v - lr * grad_l,loss

def solve_diffusion(u, dt, dx, nu) :
    #solving (I - dt * nu * lap ) v = u
    v = tf.Variable(np.random.normal(size=(u.shape)))
    for k in range(100) :      
        next_v, loss = solve_step_diff(u, dt, dx, nu, v,0.08)
        v.assign(next_v)
    print("loss diff",loss.numpy())
    return v.numpy()

#@tf.function
def solve_step_poisson(u, target,w, opt):
    with tf.GradientTape() as tape :
        #res = div(grad(u))
        res = lap(u)
        diff = res-target
        diff2 = diff*diff
        grad_u = w-grad(u)
        left_border = tf.math.reduce_sum(tf.math.square(grad_u[:,0,1]))
        right_border = tf.math.reduce_sum(tf.math.square(grad_u[:,-1,1]))
        top_border = tf.math.reduce_sum(tf.math.square(grad_u[0,:,0]))
        bottom_border = tf.math.reduce_sum(tf.math.square(grad_u[-1,:,0]))
        loss = tf.math.reduce_sum(diff2) #+10*(left_border + right_border + top_border + bottom_border)
    opt.apply_gradients(zip(tape.gradient(loss,[u]), [u]))
    #print(loss.numpy())
    return loss

def solve_poisson(w) :
    #solving : lap u = div w
    target = div(w)
    u = tf.Variable(np.random.normal(size=(w.shape[:2])))
    opt = keras.optimizers.Adam(lr=0.6)
    for k in range(300) :      
        loss = solve_step_poisson(u, target, w,opt) 
    div_w = div(w - grad(u))
    print(np.sum(div_w*div_w))
    print("loss poisson", loss.numpy())
    return u.numpy()


class Simulator :
    def __init__(self, m, n, dx, nu, border_condition, force):
        self.m = m
        self.n = n
        self.dx = dx
        self.nu = nu
        self.border_condition = border_condition
        self.force = force
        self.w = w = np.zeros((m,n,2))
        self.dust = [[np.random.random()*(m-1), np.random.random()*(n-1)] for k in range(400)] 
        
    def display(self, c='b'):
        plt.scatter([d[1] for d in self.dust], [d[0] for d in self.dust], c=c, s=5)
        
    def compute_w1(self, w0, dt):
        return w0 + dt*self.force(self.m,self.n, w0)
        
    def compute_w2(self, w1, dt):
        #Position of point i,j at t-dt, shape : m,n,2 
        m,n = self.m,self.n
        indices = np.indices((m,n)).swapaxes(0,2).swapaxes(0,1) - dt*w1
        indices_floor = np.floor(indices).astype(np.int)
        frac = indices-indices_floor
        frac = [1-frac,frac]
        indices_ceil = indices_floor + 1
        indices_floor[:,:,0] = np.clip(indices_floor[:,:,0], 0, m-1)
        indices_ceil[:,:,0] = np.clip(indices_ceil[:,:,0], 0, m-1)
        indices_floor[:,:,1] = np.clip(indices_floor[:,:,1], 0, n-1)
        indices_ceil[:,:,1] = np.clip(indices_ceil[:,:,1], 0, n-1)
        indices = [indices_floor, indices_ceil] # shape : 2 m,n,2

        w2 = np.zeros((m,n,2))
        for i in range(2) :
            for j in range(2) :
                w2 += np.reshape(frac[i][:,:,0]*frac[j][:,:,1], (m,n,1)) * take(w1, indices[i][:,:,0], indices[j][:,:,1])
        return w2
    
    
    def compute_w3(self, w2, dt):
        return solve_diffusion(w2, dt, self.dx, self.nu)


    def compute_w4(self, w3, dt):
        global test_u
        u = solve_poisson(w3)
        test_u = u

        grad_u = grad(u).numpy()
        w4 = w3 - grad_u
        
        return w4
    
    def update_dust(self, dt) :
        for d in self.dust :
            d[0] = max(min(d[0], self.m-1), 0)
            d[1] = max(min(d[1], self.n-1), 0)
            i = int(d[0])
            j = int(d[1])
            d[0] += dt*self.w[i,j,0]
            d[1] += dt*self.w[i,j,1]
    
    def time_step(self, dt, debug=False) :
        w0 = self.w.copy() 
        self.border_condition(w0)
        w1 = self.compute_w1(w0, dt)

        self.border_condition(w1)
        w2 = self.compute_w2(w1, dt)
        self.border_condition(w2)

        w3 = self.compute_w3(w2, dt)
        self.border_condition(w3)

        w4 = self.compute_w4(w3, dt)
        self.border_condition(w4)
        self.w = w4
        self.update_dust(dt)