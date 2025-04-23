import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx("float64")

# data
x=np.linspace(-1,1,256)
y=np.linspace(0,1,128)
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

x_bi_test=V_data[300:360,0:1]
y_bi_test=V_data[300:360,1:2]
u_bi_test=U_data[300:360]

idx_c=np.where((V_star[:,1]!=0)&(V_star[:,0]!=1.0)&(V_star[:,0]!=-1.0))[0]
V_c_train=V_star[idx_c]
np.random.seed(1234)
np.random.shuffle(V_c_train)
x_c_train=V_c_train[:,0:1]
y_c_train=V_c_train[:,1:2]

x_bi_train,y_bi_train,u_bi_train,x_c_train,y_c_train,x_bi_test,y_bi_test,u_bi_test=map(lambda x: tf.convert_to_tensor(x,dtype=tf.float64),[x_bi_train,y_bi_train,u_bi_train,x_c_train,y_c_train,x_bi_test,y_bi_test,u_bi_test])
batch_size=32
bi_dataset=tf.data.Dataset.from_tensor_slices((x_bi_train,y_bi_train,u_bi_train)).shuffle(buffer_size=300).batch(batch_size)

pde_dataset=tf.data.Dataset.from_tensor_slices((x_c_train,y_c_train)).shuffle(len(x_c_train)).batch(batch_size)
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
def bi_loss_fun(bi_batch):
    x_bi, y_bi, u_bi = bi_batch
    u_pred = model(tf.concat([x_bi, y_bi], axis=1))
    return tf.reduce_mean(tf.square(u_bi - u_pred))

@tf.function
def pde_loss_fun(pde_batch):
        x_pde, y_pde = pde_batch
        with tf.GradientTape(persistent=True) as pde_tape:
            pde_tape.watch([x_pde, y_pde])
            with tf.GradientTape(persistent=True) as second_tape:
                second_tape.watch([x_pde, y_pde])
                u_pde = model(tf.concat([x_pde, y_pde], axis=1))
            
            # 计算一阶导数
            u_x = second_tape.gradient(u_pde, x_pde)
            u_y = second_tape.gradient(u_pde, y_pde)
            
            # 计算二阶导数
            u_xx = pde_tape.gradient(u_x, x_pde)
        
        # PDE残差
        F = u_y + u_pde * u_x - 0.005 * u_xx
        loss_pde = tf.reduce_mean(tf.square(F))
        del second_tape
        del pde_tape
        return loss_pde
        

# 训练循环
opt = tf.keras.optimizers.Adam(learning_rate=2e-4)
epochs = 10000
loss_history = []


for epoch in range(epochs):
    pde_loss=0
    bi_loss=0
    with tf.GradientTape() as tape:
        for step ,pde_batch in enumerate(pde_dataset):
            pde_loss+=pde_loss_fun(pde_batch)
        for step,bi_batch in enumerate(bi_dataset):
            bi_loss+=bi_loss_fun(bi_batch)
        total_loss=pde_loss+bi_loss
    grads = tape.gradient(total_loss, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    if step % 100 == 0:
        print(f"Step {step}: Total Loss: {total_loss}, "
              f"Boundary Loss: {bi_loss}, "
              f"PDE Loss: {pde_loss}")
        loss_history.append(total_loss.numpy())