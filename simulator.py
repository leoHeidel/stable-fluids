import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("GPUs found :", tf.config.experimental.list_physical_devices("GPU"))

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
    return tf.nn.convolution(expended, grad_kernel, padding="SAME", strides=1)[0]
def lap(u) :
    expended = tf.expand_dims(tf.expand_dims(u, 0), -1)
    return tf.nn.convolution(expended, lap_kernel, padding="SAME", strides=1)[0,:,:,0]
def lap_vec(u) :
    expended = tf.expand_dims(u, 0)
    return tf.nn.convolution(expended, lap_vec_kernel, padding="SAME", strides=1)[0,:,:]
def div(u) :
    expended = tf.expand_dims(u, 0)
    return tf.nn.convolution(expended, div_kernel, padding="SAME", strides=1)[0,:,:,0]


def take(a,indices_i, indices_j) :
    m,n = a.shape[:2]
    full_idices = n*indices_i + indices_j
    reshaped = a.reshape(m*n,2)
    res = np.take(reshaped,full_idices, axis=0)
    return res.reshape(m,n,2)      

def conjgrad_lap_step(r,x,p,rsold) :
    Ap = lap(p)
    alpha = rsold / tf.reduce_sum(p*Ap)
    x.assign(x + alpha * p)
    r.assign(r - alpha * Ap)
    rsnew = tf.reduce_sum(r*r)
    p.assign(r + (rsnew / rsold) * p)
    rsold.assign(rsnew)   

def conjgrad_lap(b,x, error_max) :
    r = tf.Variable(b-lap(x))
    p = tf.Variable(r)
    rsold = tf.Variable(tf.reduce_sum(r*r))
    for i in range(1000) :
        rsnew = conjgrad_lap_step(r,x,p,rsold)
        if  tf.sqrt(rsold) < error_max :
            #print(i)
            break
        
def conjgrad_diff_step(r,x,p,rsold,nu,dt) :
    Ap = p - dt*nu*lap_vec(p)
    tmp = tf.reduce_sum(p*Ap)
    alpha = rsold / tmp
    x.assign(x + alpha * p)
    r.assign(r - alpha * Ap)
    rsnew = tf.reduce_sum(r*r)
    p.assign(r + (rsnew / rsold) * p)
    rsold.assign(rsnew)

def conjgrad_diff(w2, nu, dt, error_max) :
    x = tf.Variable(w2)
    r = tf.Variable(w2 - x + dt*nu*lap_vec(x))
    p = tf.Variable(r)
    rsold = tf.Variable(tf.reduce_sum(r*r))
    for i in range(1000) :
        conjgrad_diff_step(r,x,p,rsold,nu,dt)
        if  tf.sqrt(rsold) < error_max :
            #print(i, rsold.numpy())
            break
        rsold
    return x

def conjgrad_diff_sp(w2, nu, dt, error_max) :
    def lin_func(a) :
        a = np.reshape(a, w2.shape)
        return np.reshape(a - dt*nu*lap_vec(a), (-1,))
    x = np.zeros(w2.shape)
    x = np.reshape(x, (-1,))
    op = LinearOperator((len(x),len(x)),matvec=lin_func)
    return np.reshape(cg(op, np.reshape(w2,(-1,)),x)[0],w2.shape)

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
        self.u = tf.Variable(np.zeros((self.w.shape[:2])))
        self.indices = tf.constant(np.indices((m,n)).swapaxes(0,2).swapaxes(0,1), dtype=tf.float32)
        self.clipping = tf.constant(np.stack([np.ones((m,n), dtype=np.int32)*m,np.ones((m,n), dtype=np.int32)*n], axis=-1))
        
    def display(self, c='b'):
        plt.scatter([d[1] for d in self.dust], [d[0] for d in self.dust], c=c, s=5)
        
    def compute_w1(self, w0, dt):
        return w0 + dt*self.force(self.m,self.n, w0)
        
    def compute_w2_numpy(self, w1, dt):
        #old numpy deprecated version
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
    
    def compute_w2(self, w1, dt):
        #Position of point i,j at t-dt, shape : m,n,2 
        w1 = tf.constant(w1, dtype=tf.float32)
        m,n = self.m,self.n
        indices = self.indices - dt*w1
        indices_floor = tf.math.floor(indices)
        frac = indices-indices_floor
        frac = [1-frac,frac]
        indices_floor = tf.dtypes.cast(indices_floor, tf.int32)
        indices_ceil = indices_floor + 1
        indices_ceil = tf.clip_by_value(indices_ceil, 0, self.clipping)
        indices_floor = tf.clip_by_value(indices_floor, 0, self.clipping)  
        indices = [indices_floor, indices_ceil] # shape : 2 m,n,2
        w2 = tf.zeros((m,n,2))
        for i in range(2) :
            for j in range(2) :
                indices_to_take = tf.stack([indices[i][:,:,0], indices[j][:,:,1]], axis=-1)
                w2 = w2 + tf.expand_dims(frac[i][:,:,0] * frac[j][:,:,1], -1) * tf.gather_nd(w1,indices_to_take)
        return w2.numpy().astype(np.float64)
    
    
    def compute_w3(self, w2, dt):
        #solving (I - dt * nu * lap ) v = w2
        #v = conjgrad_diff_sp(w2, self.nu, dt, 1e-5*self.m*self.n)
        v = conjgrad_diff(w2, self.nu, dt, 1e-6*self.m*self.n)
        diff = v-dt*self.nu*lap_vec(w2)-w2
        loss = tf.reduce_sum(diff*diff)
        #print("diff :", loss.numpy()/(self.m*self.n))
        return v.numpy()
        #return v


    def compute_w4(self, w3, dt):
        target = div(w3)
        #self.u.assign(np.zeros((w3.shape[:2])))
        #self.u.assign(self.u+s)
        conjgrad_lap(target, self.u, 1e-5*self.m*self.n)
        diff = lap(self.u)-target
        loss = tf.reduce_sum(diff*diff)
        #print("poisson :", loss.numpy()/(self.m*self.n))
        grad_u = grad(self.u).numpy()
        return  w3 - grad_u
    
    def update_dust(self, dt) :
        for d in self.dust :
            d[0] = max(min(d[0], self.m-1), 0)
            d[1] = max(min(d[1], self.n-1), 0)
            i = int(d[0])
            j = int(d[1])
            d[0] += dt*self.w[i,j,0]
            d[1] += dt*self.w[i,j,1]
    
    def time_step(self, dt, debug=False) :
        #start = time.time()
        w0 = self.w.copy() 
        w1 = self.compute_w1(w0, dt)
        #w1_time = time.time()

        self.border_condition(w1)
        w2 = self.compute_w2(w1, dt)
        #w2_time = time.time()

        cached_u = self.u.numpy().copy()
        w3 = self.compute_w3(w2, dt)
        #w3_time = time.time()

        w4 = self.compute_w4(w3, dt)
        #w4_time = time.time()

        self.w = w4
        self.update_dust(dt)
        
        #print(w1_time-start, w2_time-w1_time, w3_time-w2_time, w4_time-w3_time) 
        return w3, cached_u
    
def display_w(w) :
    m,n = w.shape[:2]
    for k in range(20) :
        for l in range(20) :
            i = (m-2)*k/20 + 1
            j = (n-2)*l/20 + 1
            plt.plot([j, j+w[int(i), int(j), 1]*3], [i, i+w[int(i), int(j), 0]*3], c="b")
            plt.scatter(j,i, c="b", s=5)