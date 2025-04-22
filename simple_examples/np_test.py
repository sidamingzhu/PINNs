import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

tf.keras.backend.set_floatx("float64")

x=np.linspace(-1,1,256)
t=np.linspace(0,1,100)
X,T=np.meshgrid(x,t)
V_star=np.vstack((X.flatten(),T.flatten())).T

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

idx_c=np.where((V_star[:,1]!=0)&(V_star[:,0]!=1.0)&(V_star[:,0]!=-1.0))[0]
V_c_train=V_star[idx_c]
x_c_train=V_c_train[:,0:1]
y_c_train=V_c_train[:,1:2]

x_bi_train,y_bi_train,u_bi_train,x_c_train,y_c_train=map(lambda x: tf.convert_to_tensor(x,dtype=tf.float64),[x_bi_train,y_bi_train,u_bi_train,x_c_train,y_c_train])

# x_test=V_data[300:450,0:1]
# y_test=V_data[300:450,1:2]
# u_test=U_data[300:450]




# def f_r(X_f,model_nn):
#     x_temp=X_f[:,0:1]
#     t_temp=X_f[:,1:2]
#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch(x_temp)
#         tape.watch(t_temp)
#         X_temp=tf.concat([x_temp,t_temp],axis=1)
#         u=model_nn(X_temp)
#         u_x=tape.gradient(u,x_temp)
#         u_xx=tape.gradient(u_x,x_temp)
#         u_t=tape.gradient(u,t_temp)
#     f=u_t+u*u_x-0.025*u_xx
#     return f    


# with tf.GradientTape() as g:
#             g.watch(x)
#             with tf.GradientTape() as gg:
#                 gg.watch(x)
#                 u = self.model(x)
#             du_dtx = gg.batch_jacobian(u, x)
#             du_dt = du_dtx[..., 0]
#             du_dx = du_dtx[..., 1]
#         d2u_dx2 = g.batch_jacobian(du_dx, x)[..., 1]
#         return u, du_dt, du_dx, d2u_dx2

class FNN(tf.keras.Model):
    def __init__(self, layer_sizes):
        super(FNN, self).__init__()
        self.denses = []
        for units in layer_sizes[1:-1]:
            self.denses.append(tf.keras.layers.Dense(units, activation="relu"))
        self.denses.append(tf.keras.layers.Dense(layer_sizes[-1]))

    def call(self, x, y):
        V=tf.concat([x,y],axis=1)
        for f in self.denses:
            V = f(V)
        return V
    
layer_sizes = [2] + [128] * 4 + [1]

nn = FNN(layer_sizes)

@tf.function
def uderx(x, y):
    u = nn(x, y)
    u_x = tf.gradients(u, x)[0]
    return u_x

@tf.function
def udery(x, y):
    u = nn(x, y)
    u_y = tf.gradients(u,y)[0]
    return u_y

@tf.function
def f(x, y):
    u0 = nn(x, y)
    u_x = tf.gradients(u0, x)[0]
    u_y = tf.gradients(u0, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
    F = u_y+u0*u_x-u_xx
    retour = tf.reduce_mean(tf.square(F)) 
    return retour

@tf.function
def mse(u, u_):
    return tf.reduce_mean(tf.square(u-u_))

loss = 0
epochs = 60000
opt = tf.keras.optimizers.Adam(learning_rate=2e-4)
epoch = 0
loss_values = np.array([])
L_values = np.array([])
l_values = np.array([])

#
start = time.time()
#
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        T_ = nn(x_bi_train, y_bi_train)
        # Tderx_= uderx(x_bi_train, y_bi_train)
        # Tdery_ = udery(x_bi_train, y_bi_train)
        
        #loss on PDE
        L = 1*f(x_c_train, y_c_train)

        l = 0
        #l = 1*msex1(t_d, T_) +  1*msey1(t_d, T_) + 1*msey2(t_d, T_) + 0*msex2(t_d, T_)
        
        # Select the loss on data derivatives wrt x or/and wrt y....
        #l = l + 1*msex1(t_dx, Tderx_) + 1*msex2(t_dx, Tderx_) + 1*msey1(t_dx, Tderx_) + 1*msey2(t_dx, Tderx_)
        #l = l + 1*msex1(t_dy, Tdery_) + 1*msex2(t_dy, Tdery_) + 1*msey1(t_dy, Tdery_) + 1*msey2(t_dy, Tdery_)

        l = mse(u_bi_train, T_) 
        loss = L + l
        
    g = tape.gradient(loss, nn.trainable_weights)
    opt.apply_gradients(zip(g, nn.trainable_weights))
    #loss_values = np.append(loss_values, loss)
    #L_values = np.append(L_values, L)
    #l_values = np.append(l_values, l)
    
    
    if epoch % 100 == 0 or epoch == epochs-1:
        print(f"{epoch:5}, {loss.numpy():.9f}")
        loss_values = np.append(loss_values, loss)
        L_values = np.append(L_values, L)
        l_values = np.append(l_values, l)






# @tf.function
# def train(x, y_):
#     with tf.GradientTape() as tape:
#         y = nn(x)
#         # Loss
#         loss = tf.reduce_mean((y - y_) ** 2)
#     gradients = tape.gradient(loss, nn.trainable_variables)
#     opt.apply_gradients(zip(gradients, nn.trainable_variables))
#     return loss


# # Train
# for i in range(nsteps):
#     err_train = train(V_train, U_train).numpy()
#     if i % 1000 == 0 or i == nsteps - 1:
#         pred_y = nn(V_test).numpy()
#         err_test = np.mean((pred_y - U_test) ** 2)
#         print(i, err_train, err_test)

# Plot
# plt.plot(test_x, test_y, "o")
# plt.plot(test_x, pred_y, "v")
# plt.show()