import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

V_train=V_data[:-10000]
U_train=U_data[:-10000]
V_test=V_data[-10000:]
U_test=U_data[-10000:]

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

    def call(self, inputs):
        x,t=inputs
        V=tf.concat([x,t],axis=1)
        for f in self.denses:
            V = f(V)
        return V
    
layer_sizes = [2] + [128] * 4 + [1]
lr = 0.001
nsteps = 10000

# Build NN
nn = FNN(layer_sizes)

# Optimizer
opt = tf.keras.optimizers.Adam(learning_rate=lr)


@tf.function
def train(x, y_):
    with tf.GradientTape() as tape:
        y = nn(x)
        # Loss
        loss = tf.reduce_mean((y - y_) ** 2)
    gradients = tape.gradient(loss, nn.trainable_variables)
    opt.apply_gradients(zip(gradients, nn.trainable_variables))
    return loss


# Train
for i in range(nsteps):
    err_train = train(V_train, U_train).numpy()
    if i % 1000 == 0 or i == nsteps - 1:
        pred_y = nn(V_test).numpy()
        err_test = np.mean((pred_y - U_test) ** 2)
        print(i, err_train, err_test)

# Plot
# plt.plot(test_x, test_y, "o")
# plt.plot(test_x, pred_y, "v")
# plt.show()