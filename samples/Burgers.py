import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx("float64")

# data
x=np.linspace(-1,1,200)
y=np.linspace(0,1,100)
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

x_bi_train=V_data[0:300,0:1]
y_bi_train=V_data[0:300,1:2]
u_bi_train=U_data[0:300]

x_bi_test=V_data[300:400,0:1]
y_bi_test=V_data[300:400,1:2]
u_bi_test=U_data[300:400]

idx_c=np.where((V_star[:,1]!=0)&(V_star[:,0]!=1.0)&(V_star[:,0]!=-1.0))[0]
V_c_train=V_star[idx_c]
np.random.seed(1234)
np.random.shuffle(V_c_train)
x_c_train=V_c_train[0:300,0:1]
y_c_train=V_c_train[0:300,1:2]

x_bi_train,y_bi_train,u_bi_train,x_c_train,y_c_train,x_bi_test,y_bi_test,u_bi_test=map(lambda x: tf.convert_to_tensor(x,dtype=tf.float64),[x_bi_train,y_bi_train,u_bi_train,x_c_train,y_c_train,x_bi_test,y_bi_test,u_bi_test])


# model
inputs = keras.Input(shape=(2,))
hidden_denses=[]
for i in range(5):
    hidden_denses.append(layers.Dense(8,activation='tanh'))
x=hidden_denses[0](inputs)
for i in range(1,5):
    x=hidden_denses[i](x)

outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="pinns")
model.summary()
# tf.keras.backend.clear_session()

@tf.function
def u(x,y):
    return model(tf.concat([x,y],axis=1))

@tf.function
def f(x,y):
    u0=u(x,y)
    u_x = tf.gradients(u0,y)[0]
    u_y = tf.gradients(u0, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    F=u_y+u0*u_x-0.005*u_xx
    retour = tf.reduce_mean(tf.square(F)) 
    return retour

@tf.function
def mse(u,u_):
    return tf.reduce_mean(tf.square(u-u_))

loss = 0
epochs = 10000
opt = tf.keras.optimizers.Adam(learning_rate=2e-4)
epoch = 0
loss_values = np.array([])
loss_on_pde_values = np.array([])
loss_1_values = np.array([])

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        U_=u(x_bi_train,y_bi_train)
        loss_1=mse(u_bi_train,U_)
        loss_on_pde=f(x_c_train,y_c_train)
        loss=loss_1+loss_on_pde
    grads=tape.gradient(loss,model.trainable_weights)
    opt.apply_gradients(zip(grads,model.trainable_weights))
    if epoch % 100==0 or epoch == epochs-1:
        print(epoch,"  loss: ",loss.numpy())
        loss_values=np.append(loss_values,loss.numpy())
        loss_on_pde_values=np.append(loss_on_pde_values,loss_on_pde)
        loss_1_values=np.append(loss_1_values,loss_1)

