import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("GPUs found :", tf.config.experimental.list_physical_devices("GPU"))

#dim 3 kernels
grad_kernel_base = np.array([-1,0,1])/2
grad_kernel_x = np.expand_dims(np.expand_dims(grad_kernel_base,-1),-1)
grad_kernel_y = np.expand_dims(np.expand_dims(grad_kernel_base,-1),0)
grad_kernel_z = np.expand_dims(np.expand_dims(grad_kernel_base,0),0)
grad_kernel_x = np.expand_dims(np.expand_dims(grad_kernel_x,-1),-1)
grad_kernel_y = np.expand_dims(np.expand_dims(grad_kernel_y,-1),-1)
grad_kernel_z = np.expand_dims(np.expand_dims(grad_kernel_z,-1),-1)
grad_kernel_x = tf.constant(grad_kernel_x)
grad_kernel_y = tf.constant(grad_kernel_y)
grad_kernel_z = tf.constant(grad_kernel_z)

lap_base = np.array([1,-2,1])
lap_base = np.stack([np.zeros(3,),lap_base,np.zeros(3,)],0)
lap_kernel_x = np.stack([np.zeros((3,3)),lap_base,np.zeros((3,3))],0)
lap_kernel_y = lap_kernel_x.swapaxes(1,2)
lap_kernel_z = lap_kernel_x.swapaxes(0,2)
lap_vec_kernel2 = lap_kernel_x + lap_kernel_y + lap_kernel_z
lap_vec_kernel2 = np.expand_dims(np.expand_dims(lap_vec_kernel2,-1),-1)
lap_vec_kernel2 = tf.constant(lap_vec_kernel2)

#dim 3 functions
def grad2(u) :
    expanded = tf.expand_dims(tf.expand_dims(u,0),-1)
    #print(expended_x.shape)
    convol_x = tf.nn.conv3d(expanded, grad_kernel_x , padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
    convol_y = tf.nn.conv3d(expanded, grad_kernel_y, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
    convol_z = tf.nn.conv3d(expanded, grad_kernel_z, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
    #print(convol_x.shape)
    result = tf.stack([convol_x,convol_y,convol_z],-1)
    #print(grad.shape)
    return result

def div2(u):
    expanded_x = tf.expand_dims(tf.expand_dims(u[:,:,:,0],0),-1)
    expanded_y = tf.expand_dims(tf.expand_dims(u[:,:,:,1],0),-1)
    expanded_z = tf.expand_dims(tf.expand_dims(u[:,:,:,2],0),-1)
    #print(expanded_x.shape)
    result = tf.nn.conv3d(expanded_x, grad_kernel_x , padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
    result += tf.nn.conv3d(expanded_y, grad_kernel_y, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
    result += tf.nn.conv3d(expanded_z, grad_kernel_z, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
    return result

def lap_vec2(u) :
    expanded_x = tf.expand_dims(tf.expand_dims(u[:,:,:,0],0),-1)
    expanded_y = tf.expand_dims(tf.expand_dims(u[:,:,:,1],0),-1)
    expanded_z = tf.expand_dims(tf.expand_dims(u[:,:,:,2],0),-1)
    lap_x = tf.nn.conv3d(expanded_x, lap_vec_kernel2, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
    lap_y = tf.nn.conv3d(expanded_y, lap_vec_kernel2, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
    lap_z = tf.nn.conv3d(expanded_z, lap_vec_kernel2, padding="SAME", strides=[1,1,1,1,1])[0,:,:,:,0]
    result = tf.stack([lap_x,lap_y,lap_z],-1)
    return result

def lap2(u) :
    return div2(grad2(u))  

def take2(a,indices_i, indices_j, indices_k) :
    m,n,p = a.shape[:3]
    full_idices = p*(n*indices_i + indices_j) + indices_k
    reshaped = a.reshape(m*n*p,3)
    res = np.take(reshaped,full_idices, axis=0)
    return res.reshape(m,n,p,3)  

def conjgrad_lap_step2(r,x,p,rsold) :
    Ap = lap2(p)
    alpha = rsold / tf.reduce_sum(p*Ap)
    x.assign(x + alpha * p)
    r.assign(r - alpha * Ap)
    rsnew = tf.reduce_sum(r*r)
    p.assign(r + (rsnew / rsold) * p)
    rsold.assign(rsnew)   

def conjgrad_lap2(b,x, error_max) :
    r = tf.Variable(b-lap2(x))
    p = tf.Variable(r)
    rsold = tf.Variable(tf.reduce_sum(r*r))
    for i in range(1000) :
        rsnew = conjgrad_lap_step2(r,x,p,rsold)
        if  tf.sqrt(rsold) < error_max :
            #print(i)
            break

def conjgrad_diff_step2(r,x,p,rsold,nu,dt) :
    Ap = p - dt*nu*lap_vec2(p)
    tmp = tf.reduce_sum(p*Ap)
    alpha = rsold / tmp
    x.assign(x + alpha * p)
    r.assign(r - alpha * Ap)
    rsnew = tf.reduce_sum(r*r)
    p.assign(r + (rsnew / rsold) * p)
    rsold.assign(rsnew)

def conjgrad_diff2(w2, nu, dt, error_max) :
    x = tf.Variable(w2)
    r = tf.Variable(w2 - x + dt*nu*lap_vec2(x))
    p = tf.Variable(r)
    rsold = tf.Variable(tf.reduce_sum(r*r))
    for i in range(1000) :
        conjgrad_diff_step2(r,x,p,rsold,nu,dt)
        if  tf.sqrt(rsold) < error_max :
            #print(i, rsold.numpy())
            break
        rsold
    return x
            
class Simulator2 :
    def __init__(self, m, n, p, dx, nu, border_condition, force):
        self.m = m
        self.n = n
        self.p = p
        self.dx = dx
        self.nu = nu
        self.border_condition = border_condition
        self.force = force
        self.w = w = np.zeros((m,n,p,3))
        self.dust = [[np.random.random()*(m-1), np.random.random()*(n-1), np.random.random()*(p-1)] for k in range(1000)] 
        self.u = tf.Variable(np.zeros((self.w.shape[:3])))
        self.indices = tf.constant(np.indices((m,n,p)).swapaxes(0,1).swapaxes(1,2).swapaxes(2,3), dtype=tf.float32)
        self.clipping = tf.constant(np.stack([np.ones((m,n,p), dtype=np.int32)*m, np.ones((m,n,p), dtype=np.int32)*n, np.ones((m,n,p), dtype=np.int32)*p], axis=-1))
        
    def display(self, c='b'):
        plt.scatter([d[1] for d in self.dust], [d[0] for d in self.dust], c=c, s=5)
        
    def compute_w1(self, w0, dt):
        return w0 + dt*self.force(self.m,self.n,self.p, w0)
    
    def compute_w2(self, w1, dt):
        #Position of point i,j,k at t-dt, shape : m,n,p,3 
        w1 = tf.constant(w1, dtype=tf.float32)
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
        w2 = tf.zeros((m,n,p,3))
        for i in range(2) :
            for j in range(2) :
                for k in range(2) :
                    indices_to_take = tf.stack([indices[i][:,:,:,0], indices[j][:,:,:,1], indices[k][:,:,:,2]], axis=-1)
                    w2 = w2 + tf.expand_dims(frac[i][:,:,:,0] * frac[j][:,:,:,1] * frac[k][:,:,:,2], -1) * tf.gather_nd(w1,indices_to_take)
        return w2.numpy().astype(np.float64)
    
    
    def compute_w3(self, w2, dt):
        #solving (I - dt * nu * lap ) v = w2
        #v = conjgrad_diff_sp(w2, self.nu, dt, 1e-5*self.m*self.n)
        v = conjgrad_diff2(w2, self.nu, dt, 1e-6*self.m*self.n*self.p)
        diff = v-dt*self.nu*lap_vec2(w2)-w2
        loss = tf.reduce_sum(diff*diff)
        #print("diff :", loss.numpy()/(self.m*self.n))
        return v.numpy()
        #return v


    def compute_w4(self, w3, dt):
        target = div2(w3)
        #self.u.assign(np.zeros((w3.shape[:2])))
        #self.u.assign(self.u+s)
        conjgrad_lap2(target, self.u, 1e-5*self.m*self.n*self.p)
        diff = lap2(self.u)-target
        loss = tf.reduce_sum(diff*diff)
        #print("poisson :", loss.numpy()/(self.m*self.n))
        grad_u = grad2(self.u).numpy()
        return  w3 - grad_u
    
    def update_dust(self, dt) :
        for d in self.dust :
            d[0] = max(min(d[0], self.m-1), 0)
            d[1] = max(min(d[1], self.n-1), 0)
            d[2] = max(min(d[2], self.p-1), 0)
            i = int(d[0])
            j = int(d[1])
            k = int(d[2])
            d[0] += dt*self.w[i,j,0]
            d[1] += dt*self.w[i,j,1]
            d[2] += dt*self.w[i,j,2]
    
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
        self.border_condition(w1)

        w4 = self.compute_w4(w3, dt)
        #w4_time = time.time()

        self.w = w4
        #self.update_dust(dt)
        
        #print(w1_time-start, w2_time-w1_time, w3_time-w2_time, w4_time-w3_time) 
        return w3, cached_u