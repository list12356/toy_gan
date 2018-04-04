import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)
mb_size = 64
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
prior_std = 1.0
eta=2e-4
alpha=0.01
lr=0.0002
noise_std = np.sqrt(2 * alpha * eta)
dataset_size = 55000
gen_observed = 1000
mcmc_num = 4
condition = 7
base_learning_rate = 0.005
lr_decay_rate = 3.0

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]
# theta_D = [var for var in XXX if 'd_' in var.name]

def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

def disc_prior():
    prior_loss = 0.0
    for var in theta_D:
        nn = tf.divide(var, prior_std)
        prior_loss += tf.reduce_mean(tf.multiply(nn, nn))
        
    prior_loss /= dataset_size

    return prior_loss

def disc_noise(): # for SGHMC
    noise_loss = 0.0
    for var in theta_D:
        noise_ = tf.contrib.distributions.Normal(loc=0., scale=noise_std*tf.ones(var.get_shape()))
        noise_loss += tf.reduce_sum(var * noise_.sample())
    noise_loss /= dataset_size
    return noise_loss

""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = []
G_W2 = []
G_b1 = []
G_b2 = []
theta_G = []

for m in range(mcmc_num):
    G_W1.append(tf.Variable(xavier_init([Z_dim + y_dim, h_dim])))
    G_b1.append( tf.Variable(tf.zeros(shape=[h_dim])))

    G_W2.append(tf.Variable(xavier_init([h_dim, X_dim])))
    G_b2.append(tf.Variable(tf.zeros(shape=[X_dim])))

    theta_G.append([G_W1[m], G_W2[m], G_b1[m], G_b2[m]])


def generator(z, y):
    G_prob = []
    for m in range(mcmc_num):
        inputs = tf.concat(axis=1, values=[z, y])
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1[m]) + G_b1[m])
        G_log_prob = tf.matmul(G_h1, G_W2[m]) + G_b2[m]
        G_prob_ = tf.nn.sigmoid(G_log_prob)
        G_prob.append(G_prob_)
    return G_prob

def gen_prior(m):
    prior_loss = 0.0
    for var in theta_G[m]:
        nn = tf.divide(var, prior_std)
        prior_loss += tf.reduce_mean(tf.multiply(nn, nn))
        
    prior_loss /= gen_observed

    return prior_loss

def gen_noise(m): # for SGHMC
    noise_loss = 0.0
    for var in theta_G[m]:
        noise_ = tf.contrib.distributions.Normal(loc=0., scale=noise_std*tf.ones(var.get_shape()))
        noise_loss += tf.reduce_sum(var * noise_.sample())
    noise_loss /= gen_observed
    return noise_loss

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4 * mcmc_num, 4))
    outer_grid = gridspec.GridSpec(1, mcmc_num, wspace=0.1, hspace=0.1)
    for m in range(mcmc_num):
        gs = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec = outer_grid[m], wspace=0.05, hspace=0.05)
        # gs = gridspec.GridSpec(4, 4)
        # gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples[m]):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


G_sample = generator(Z, y)
D_real, D_logit_real = discriminator(X, y)
#
D_fake = []
D_logit_fake = []
for m in range(mcmc_num):
    D_fake_, D_logit_fake_ = discriminator(G_sample[m], y)
    D_fake.append(D_fake_)
    D_logit_fake.append(D_logit_fake_)

D_logit_fake_all = tf.concat(D_logit_fake, 0)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake_all, labels=tf.zeros_like(D_logit_fake_all)))
D_prior = disc_prior()
D_noise = disc_noise()
D_loss = D_loss_real + D_loss_fake + D_prior + D_noise

#import pdb; pdb.set_trace()
G_loss = []
for m in range(mcmc_num):
    G_loss_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake[m], labels=tf.ones_like(D_logit_fake[m])))
    G_loss_ += gen_noise(m) + gen_prior(m)
    G_loss.append(G_loss_)

learning_rate = tf.placeholder(tf.float32, shape=[])

D_solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5).minimize(D_loss, var_list=theta_D)
D_solver_adam = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = []
G_solver_adam = []
for m in range(mcmc_num):
    G_solver.append(tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5).minimize(G_loss[m], var_list=theta_G[m]))
    G_solver_adam.append(tf.train.AdamOptimizer().minimize(G_loss[m], var_list=theta_G[m]))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


out_dir = 'out_bayes_adam_' + str(condition) 
if not os.path.exists(out_dir + '/'):
    os.makedirs(out_dir + '/')

i = 0

for it in range(1000000):
    if it % 1000 == 0:
        n_sample = 16

        samples = []
        for m in range(mcmc_num): 
            Z_sample = sample_Z(n_sample, Z_dim)
            y_sample = np.zeros(shape=[n_sample, y_dim])
            y_sample[:, condition] = 5

            samples.append(sess.run(G_sample[m], feed_dict={Z: Z_sample, y:y_sample}))

        fig = plot(samples)
        plt.savefig(out_dir + '/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, y_mb = mnist.train.next_batch(mb_size)
    Z_sample = sample_Z(mb_size, Z_dim)

    lr = base_learning_rate * np.exp(-lr_decay_rate *min(1.0, (it*mb_size)/float(dataset_size)))
    
    if it > 0:
        _, D_loss_curr = sess.run([D_solver_adam, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb, learning_rate: lr})
        for m in range(mcmc_num):
            _, G_loss_curr = sess.run([G_solver_adam[m], G_loss[m]], feed_dict={Z: Z_sample, y:y_mb, learning_rate: lr})

    else:
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb, learning_rate: lr})
        for m in range(mcmc_num):
            _, G_loss_curr = sess.run([G_solver[m], G_loss[m]], feed_dict={Z: Z_sample, y:y_mb, learning_rate: lr})

    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
