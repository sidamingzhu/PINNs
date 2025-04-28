import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.keras.backend.set_floatx("float64")  

data = scipy.io.loadmat('W:/documents/dev/python/PINNs/PINNs/adaptive_sampling_pinns/Burgers/burgers_shock.mat')
x = np.linspace(-1, 1, 256)
t = np.linspace(0, 1, 100)
X_, T_ = np.meshgrid(x,t) # (100,256)

X=X_.flatten()[:, None] #(25600, 1)
T=T_.flatten()[:, None] #(25600, 1)

xt = np.hstack((X, T))  # (25600, 2)


u_exact = np.real(data['usol']).T  # (100, 256)

idx_init = np.where(xt[:, 1]==0)[0]
xt_init = xt[idx_init] # (256, 2)
x_init=xt_init[:,0:1]
t_init=xt_init[:,1:2]
u_init = -np.sin(np.pi*xt_init[:, 0:1])   # (256, 1)

idx_bc = np.where((xt[:, 0]==1.0)|(xt[:, 0]==-1.0))[0]
xt_bc = xt[idx_bc]  # (200, 2)
x_bc=xt_bc[:,0:1]
t_bc=xt_bc[:,1:2]
u_bc = np.zeros((xt_bc.shape[0], 1))  # (200, 1)

idx_Xf = np.random.choice(xt.shape[0], 2000, replace=False)#随机选2000个点
collocation_points = xt[idx_Xf] # (2000,2)
x_collocation_points=collocation_points[:,0:1]
t_collocation_points=collocation_points[:,1:2]

big_grad_points = np.array([[0.01,0.01]]) 
i=117
while i < 25599:
    temp_array=xt[i:i+20,:]
    big_grad_points=np.append(big_grad_points,temp_array,axis=0)
    i=i+256

x_big_grad_points=big_grad_points[:,0:1]
t_big_grad_points=big_grad_points[:,1:2]

X,T,x_bc,t_bc,u_bc,x_init,t_init,u_init,x_collocation_points,t_collocation_points,x_big_grad_points,t_big_grad_points=map(lambda x: tf.convert_to_tensor(x,dtype=tf.float64),[X,T,x_bc,t_bc,u_bc,x_init,t_init,u_init,x_collocation_points,t_collocation_points,x_big_grad_points,t_big_grad_points])

inputs = keras.Input(shape=(2,))
hidden_layers=[]
for _ in range(5):
    hidden_layers.append(layers.Dense(20, activation='tanh'))
x = hidden_layers[0](inputs)
for layer in hidden_layers[1:]:
    x = layer(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="pinns")  
model.summary()

@tf.function
def mse(x,y):
    return tf.reduce_mean(tf.square(x-y))

@tf.function
def bi_loss_fun(x,t,u):
    u_pred = model(tf.concat([x, t], axis=1))
    return mse(u_pred,u)

@tf.function
def pde_loss_fun(x,t):
    x_collocation, t_collocation = x,t
    with tf.GradientTape(persistent=True) as outer_tape:
        outer_tape.watch([x_collocation, t_collocation])
        with tf.GradientTape(persistent=True) as inner_tape:
            inner_tape.watch([x_collocation, t_collocation])
            u_pde = model(tf.concat([x_collocation, t_collocation], axis=1))
        
        # 计算一阶导数
        u_x = inner_tape.gradient(u_pde, x_collocation)
        u_t = inner_tape.gradient(u_pde, t_collocation)
        
        # 计算二阶导数
    u_xx = outer_tape.gradient(u_x, x_collocation)
    
    # PDE残差
    F = u_pde * u_x + u_t - 0.005 * u_xx  # Burger's equation: u_t + u*u_x = ν*u_xx
    loss_pde = mse(F,0)
    del inner_tape
    del outer_tape
    return loss_pde

opt = tf.keras.optimizers.Adam(learning_rate=2e-4)

@tf.function
def train_model(x_bc,t_bc,u_bc,x_init,t_init,u_init,x_collocation,t_collocation,x_big_grad_points,t_big_grad_points):
    with tf.GradientTape() as tape:
        total_loss=bi_loss_fun(x_bc,t_bc,u_bc)+bi_loss_fun(x_init,t_init,u_init)+pde_loss_fun(x_collocation,t_collocation)+pde_loss_fun(x_big_grad_points,t_big_grad_points)
    grads = tape.gradient(total_loss, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    return total_loss

start_time=time.time()
for epoch in range(2500):
    total_loss=train_model(x_bc,t_bc,u_bc,x_init,t_init,u_init,x_collocation_points,t_collocation_points,x_big_grad_points,t_big_grad_points)
    if epoch%10 ==0:
        print("step: ",epoch," loss: ",total_loss.numpy())

total_time=time.time()-start_time
print("total time:",total_time)
# model.save('pinn')

u_pred = model(tf.concat([X, T], axis=1))


fig = plt.figure(figsize=[15,5])
axes = fig.subplots(1,3, sharex=False, sharey=False)

img1 = axes[0].scatter(xt[:,0:1],xt[:,1:2], c=u_exact.flatten()[:, None], cmap='jet', vmax=1, vmin=-1, s=5)
axes[0].set_title('Reference solution', fontsize=15)
axes[0].set_xlabel('x', fontsize=15)
axes[0].set_ylabel('t', fontsize=15)
plt.colorbar(img1, ax=axes[0])

img2 = axes[1].scatter(xt[:,0:1],xt[:,1:2], c=u_pred.numpy(), cmap='jet', vmax=1, vmin=-1, s=5)
axes[1].set_title('PINNs prediction', fontsize=15)
axes[1].set_xlabel('x', fontsize=15)
axes[1].set_ylabel('t', fontsize=15)
plt.colorbar(img2, ax=axes[1])

img3 = axes[2].scatter(xt[:,0:1],xt[:,1:2], c=u_exact.flatten()[:, None]-u_pred.numpy(), cmap='seismic', vmax=0.01, vmin=-0.01,s=5)
axes[2].set_title('Error', fontsize=15)
axes[2].set_xlabel('x', fontsize=15)
axes[2].set_ylabel('t', fontsize=15)
plt.colorbar(img3, ax=axes[2])

plt.show()