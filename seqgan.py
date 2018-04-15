import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from scipy import stats

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

data_dim = 784
Z_dim = 100
mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, data_dim])

D_W1 = tf.Variable(xavier_init([data_dim, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, data_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[data_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    # G_num = tf.argmax(G_log_prob, 1)
    # G_one_hot = tf.one_hot(G_num, data_dim)
    G_one_hot = tf.to_int32(G_prob > tf.random_normal([1, data_dim]))
    
    return G_prob, G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


# G_sample = generator(Z)
G_reward = tf.placeholder(tf.float32, shape=[None, 1])
G_prob, G_sample = generator(Z)
# G_loss = -tf.reduce_mean(G_reward * tf.reduce_sum(tf.log(tf.clip_by_value(G_prob, 1e-20, 1.0)), 1))
G_loss = -tf.reduce_mean(G_reward * tf.reduce_sum(G_prob, 1))
fake_X = tf.placeholder(tf.float32, shape=[None, data_dim])

D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(fake_X)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))
# Alternative losses:
# -------------------
# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
# D_loss = D_loss_real + D_loss_fake
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 128

# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

out_dir = 'out_seq'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

i = 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
        # plt.figure(it)
        # plt.plot(np.arange(data_dim), np.sum(samples, 0))
        # plt.show(block=False)
        fig = plot(samples)
        plt.savefig(out_dir + '/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        X_mb, _ = mnist.train.next_batch(16)
        X_mb = (X_mb > 0.5).astype(int)
        fig2 = plot(X_mb)
        plt.savefig(out_dir + '/{}_real.png'.format(str(i).zfill(3)), bbox_inches='tight')
        plt.close(fig2)
        i += 1

    # numbers  = np.random.normal(loc = 30, scale = 5, size=(mb_size, 1) ).astype(int)
    # X_mb = np.zeros((mb_size, data_dim))
    # X_mb[np.arange(mb_size), numbers] = 1
    X_mb, _ = mnist.train.next_batch(mb_size)
    X_mb = (X_mb > 0.5).astype(int)
    # import pdb; pdb.set_trace()

    # reward = np.divide(reward, np.sum(reward))
    for m in range(1):
        z = sample_Z(mb_size, Z_dim)
        sample = sess.run(G_sample, feed_dict={Z: z})
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={Z: z, X: X_mb, fake_X: sample})
    z = sample_Z(mb_size, Z_dim)
    sample = sess.run(G_sample, feed_dict={Z: z})
    reward = sess.run(D_fake, feed_dict={fake_X: sample, Z: z, X: X_mb})
    # reward = reward / (1- reward)
    # import pdb; pdb.set_trace()
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: z, X: sample, G_reward: reward})
    # for i in range(3):
    # import pdb; pdb.set_trace()

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()