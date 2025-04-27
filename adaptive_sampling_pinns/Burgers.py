import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx("float64")  

# data
x=np.linspace(-1,1,64)
y=np.linspace(0,1,32)
X,Y=np.meshgrid(x,y)  
V_star=np.vstack((X.flatten(),Y.flatten())).T

# define ic
idx_init=np.where(V_star[:,1]==0)[0]
V_init=V_star[idx_init]
u_init=-np.sin(np.pi*V_init[:,0:1])

# define bc
idx_bc=np.where((V_star[:,0]==1.0)|(V_star[:,0]==-1.0))[0]
V_bc=V_star[idx_bc]
u_bc=np.zeros((V_bc.shape[0],1))

V_data=np.vstack((V_init,V_bc))
U_data=np.vstack((u_init,u_bc))


np.random.seed(1234)
np.random.shuffle(V_data)
np.random.seed(1234)
np.random.shuffle(U_data)

x_bi_train=V_data[0:64,0:1]
y_bi_train=V_data[0:64,1:2]
u_bi_train=U_data[0:64]

x_bi_test=V_data[64:128,0:1]
y_bi_test=V_data[64:128,1:2]
u_bi_test=U_data[64:128]

idx_c=np.where((V_star[:,1]!=0)&(V_star[:,0]!=1.0)&(V_star[:,0]!=-1.0))[0]
V_c=V_star[idx_c]
np.random.seed(1234)
np.random.shuffle(V_c)
x_c_train=V_c[0:1300,0:1]
y_c_train=V_c[0:1300,1:2]


C_data=np.vstack((x_c_train,y_c_train))
C_data_init=np.random.choice(C_data.shape[0],64,replace=True)
x_c_train=C_data_init[:,0:1]
y_c_train=C_data_init[:,1:2]

x_c_test=V_c[1300:1956,0:1]
y_c_test=V_c[1300:1956,1:2]

x_bi_train,y_bi_train,u_bi_train,x_c_train,y_c_train,x_bi_test,y_bi_test,u_bi_test,x_c_test,y_c_test=map(lambda x: tf.convert_to_tensor(x,dtype=tf.float64),[x_bi_train,y_bi_train,u_bi_train,x_c_train,y_c_train,x_bi_test,y_bi_test,u_bi_test,x_c_test,y_c_test])

# bi_dataset=tf.data.Dataset.from_tensor_slices((x_bi_train,y_bi_train,u_bi_train)).shuffle(buffer_size=256).batch(32)

# pde_dataset=tf.data.Dataset.from_tensor_slices((x_c_train,y_c_train)).shuffle(len(x_c_train)).batch(64)
# test_pde_dataset=tf.data.Dataset.from_tensor_slices((x_c_test,y_c_test)).shuffle(len(x_c_test)).batch(64)
# model
inputs = keras.Input(shape=(2,))
hidden_layers=[]
for _ in range(5):
    hidden_layers.append(layers.Dense(8, activation='tanh'))
x = hidden_layers[0](inputs)
for layer in hidden_layers[1:]:
    x = layer(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="pinns")  
model.summary()
# tf.keras.backend.clear_session()

@tf.function
def mse(u,u_pred):
    return tf.reduce_mean(tf.square(u-u_pred))

# @tf.function
# def bi_loss_fun(bi_batch):
#     x_bi, y_bi, u_bi = bi_batch
#     u_pred = model(tf.concat([x_bi, y_bi], axis=1))
#     return mse(u_pred,u_bi)

@tf.function
def bi_loss_fun(x,y,u):
    u_pred = model(tf.concat([x, y], axis=1))
    return mse(u_pred,u)

@tf.function
def pde_loss_fun(x_pde_i, y_pde_i):
        x_pde, y_pde = x_pde_i, y_pde_i
        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch([x_pde, y_pde])
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch([x_pde, y_pde])
                u_pde = model(tf.concat([x_pde, y_pde], axis=1))
            
            # 计算一阶导数
            u_x = inner_tape.gradient(u_pde, x_pde)
            u_y = inner_tape.gradient(u_pde, y_pde)
            
            # 计算二阶导数
        u_xx = outer_tape.gradient(u_x, x_pde)
        
        # PDE残差
        F = u_pde * u_x + u_y - 0.005 * u_xx  # Burger's equation: u_t + u*u_x = ν*u_xx
        loss_pde = mse(F,0)
        del inner_tape
        del outer_tape
        return loss_pde
        

# 训练循环
opt = tf.keras.optimizers.Adam(learning_rate=2e-4)
epochs = 60000

# pde_data=iter(pde_dataset.repeat())
# test_pde_data=iter(test_pde_dataset.repeat())

total_v = []
test_v=[]
x_v = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        bi_loss=bi_loss_fun(x_bi_train,y_bi_train,u_bi_train)
        pde_loss=pde_loss_fun(x_c_train,y_c_train)
        total_loss=bi_loss+pde_loss

    grads = tape.gradient(total_loss, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    if epoch % 100 == 0:
        test_loss_bi=bi_loss_fun(x_bi_test,y_bi_test,u_bi_test)
        test_loss_pde=pde_loss_fun(x_c_test,y_c_test)
        test_total_loss=test_loss_bi+test_loss_pde
        total_loss=total_loss.numpy()
        test_total_loss=test_total_loss.numpy()
        print("step: ",epoch," total loss: ",total_loss," error on the test dataset: ",test_total_loss)
        total_v.append(total_loss)
        test_v.append(test_total_loss)
        x_v.append(epoch)
plt.figure(figsize=(5, 2.7))  
plt.plot(x_v[:len(total_v)], total_v, label='Total Loss')
plt.plot(x_v[:len(test_v)], test_v, label='Test Loss')
plt.xlabel('Epoch', fontsize=10) 
plt.ylabel('Loss', fontsize=10)  

plt.legend()
plt.title("Simple Plot")
plt.show()