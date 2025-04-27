import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.set_floatx("float64")  

data = scipy.io.loadmat('W:/documents/dev/python/PINNs/PINNs/vallina_pinns/burgers_shock.mat')
x = np.linspace(-1, 1, 256)
t = np.linspace(0, 1, 100)
X, T = np.meshgrid(x,t)
xt = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # (25600, 2)
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

xt,u_exact,xt_init,u_init,xt_bc,u_bc,collocation_points=map(lambda x: tf.convert_to_tensor(x,dtype=tf.float64),[xt,u_exact,xt_init,u_init,xt_bc,u_bc,collocation_points])

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
        x_pde, t_pde = x,t
        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch([x_pde, t_pde])
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch([x_pde, t_pde])
                u_pde = model(tf.concat([x_pde, t_pde], axis=1))
            
            # 计算一阶导数
            u_x = inner_tape.gradient(u_pde, x_pde)
            u_t = inner_tape.gradient(u_pde, t_pde)
            
            # 计算二阶导数
        u_xx = outer_tape.gradient(u_x, x_pde)
        
        # PDE残差
        F = u_pde * u_x + u_t - 0.005 * u_xx  # Burger's equation: u_t + u*u_x = ν*u_xx
        loss_pde = mse(F,0)
        del inner_tape
        del outer_tape
        return loss_pde

opt = tf.keras.optimizers.Adam(learning_rate=2e-4)

for i in range(10000):
    with tf.GradientTape() as tape:
        init_loss=bi_loss_fun(x_init,t_init,u_init)
        bc_loss=bi_loss_fun(x_bc,t_bc,u_bc)
        pde_loss=pde_loss_fun(x_collocation_points,t_collocation_points)
        total_loss=init_loss+bc_loss+pde_loss
    grads = tape.gradient(total_loss, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    if(i%10==0):
        print("step: ",i," loss: ",total_loss)

model.save('pinn')


