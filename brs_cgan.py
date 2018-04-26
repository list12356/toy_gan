import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from scipy import stats

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dir', default="out_brs_cgan")
parser.add_argument('--alpha', type=float, default=10.0)
parser.add_argument('--l', type=float, default=1.)
parser.add_argument('--sigma', type=int, default=1)
parser.add_argument('--mode', default="binary")
args = parser.parse_args()

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)
out_dir = args.dir
save_step = 100
data_dim = 784
Z_dim = 100
C_dim = mnist.train.labels.shape[1]
search_num = 64
alpha = args.alpha
v = 0.02
_lambda = args.l
mode = args.mode
restore = False
D_lr = 1e-4
mb_size = 128

X = tf.placeholder(tf.float32, shape=[None, data_dim])
C = tf.placeholder(tf.float32, shape=[None, C_dim])

D_W1 = tf.Variable(xavier_init([data_dim + C_dim, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


class Generator:
    def __init__(self):
        self.Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
        self.C = C

        self.G_W1 = tf.Variable(xavier_init([Z_dim + C_dim, 128]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[128]))

        self.G_W2 = tf.Variable(xavier_init([128, data_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[data_dim]))

        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]
        self.size = [[Z_dim + C_dim, 128], [128,data_dim], [128], [data_dim]]
        self.build()
    
    def build(self):
        inputs = tf.concat(axis=1, values=[self.Z, tf.to_float(self.C)])
        G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        self.G_prob = tf.nn.sigmoid(G_log_prob)
        if mode == "smooth":
            self.G_sample = self.G_prob
        elif mode == "binary":
            self.G_sample = tf.to_int32(self.G_prob > 0.5)
        elif mode == "multilevel":
            self.G_sample = tf.to_int32(self.G_prob > 1/ 10.0)
            for i in range(2, 10):
                self.G_sample = self.G_sample + tf.to_int32(self.G_prob > i/ 10.0)
            self.G_sample = tf.to_float(self.G_sample) / tf.constant(10.0)
        else:
            print("Incompatiable mode!")
            exit()

    def update(self):
        return
        


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def discriminator(x, c):
    inputs = tf.concat(axis=1, values=[tf.to_float(x), tf.to_float(c)])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
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
G = Generator()
G_prob = G.G_prob
G_sample = G.G_sample

S = Generator()
S_prob = S.G_prob
S_sample = S.G_sample

# placeholder for discriminator
D_real, D_logit_real = discriminator(X, C)
D_fake, D_logit_fake = discriminator(G_sample, C)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = tf.reduce_mean(tf.log(D_fake)) * tf.constant(_lambda)

S_fake, S_logit_fake = discriminator(S_sample, C)

S_loss = tf.reduce_mean(tf.log(S_fake)) * tf.constant(_lambda)
# Alternative losses:
# -------------------
# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
# D_loss = D_loss_real + D_loss_fake
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

# D_solver = tf.train.GradientDescentOptimizer(learning_rate=D_lr).minimize(D_loss, var_list=theta_D)
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

update_Sp = []
update_Sn = []
update_G = []
update_Gph = []
delta_ph = []
for t in range(len(G.theta_G)):
    delta_ph.append(tf.placeholder(tf.float32, shape=G.size[t]))    
    update_Gph.append(tf.placeholder(tf.float32, shape=G.size[t]))
    update_Sp.append(tf.assign(S.theta_G[t], G.theta_G[t] + delta_ph[t]))
    update_Sn.append(tf.assign(S.theta_G[t], G.theta_G[t] - delta_ph[t]))
    update_G.append(tf.assign(G.theta_G[t], G.theta_G[t] + update_Gph[t]))

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for num in range(C_dim):
    if not os.path.exists(out_dir + '/' + str(num)):
        os.makedirs(out_dir + '/' + str(num))

img_num = 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if restore == True:
    weights = np.load('save/generator_200.npz')
    sess.run(tf.assign(G.G_W1, weights['gw1']))
    sess.run(tf.assign(G.G_W2, weights['gw2']))
    sess.run(tf.assign(G.G_b1, weights['gb1']))
    sess.run(tf.assign(G.G_b2, weights['gb2']))
    sess.run(tf.assign(D_W1, weights['dw1']))
    sess.run(tf.assign(D_W2, weights['dw2']))
    sess.run(tf.assign(D_b1, weights['db1']))
    sess.run(tf.assign(D_b2, weights['db2']))
    
out_file = open(out_dir+"/values.txt", "a+")

for it in range(1000000):
    if it % save_step == 0:
        # import pdb; pdb.set_trace()
        for num in range(10):
            n_sample = 16
            Z_sample = sample_Z(n_sample, Z_dim)
            C_sample = np.zeros(shape=[n_sample, C_dim])
            C_sample[:, num] = 1
            samples = sess.run(G_sample, feed_dict={G.Z: sample_Z(n_sample, Z_dim), C: C_sample})

            fig = plot(samples)
            # plt.savefig(out_dir + '/{}'.format(str(img_num).zfill(5)) + '_' + str(num), bbox_inches='tight')
            plt.savefig(out_dir + '/' + str(num) + '/{}.png'.format(str(img_num).zfill(5)), bbox_inches='tight')
            plt.close(fig)
            saver.save(sess, out_dir + '/{}_model.ckpt'.format(str(img_num).zfill(5)))
        img_num += 1


    if mode == "smooth":
        X_mb, C_mb = mnist.train.next_batch(mb_size)
        # X_mb = (X_mb > 0.5).astype(int)
    elif mode == "binary":
        X_mb, C_mb = mnist.train.next_batch(mb_size)
        X_mb = (X_mb > 0.5).astype(int)
    elif mode == "multilevel":
        X_mb_s, C_mb = mnist.train.next_batch(mb_size)
        X_mb = np.zeros((mb_size, 784))
        for j in range(1, 10):
            X_mb = X_mb + (X_mb_s > j / 10.0).astype(float)
        X_mb = X_mb / 10.0
    else:
        print("Incompatiable mode!")
        exit()

    update = []
    reward_list = []
    reward_list_2 = []
    delta_list = []
    for m in range(search_num):
        sample = sample_Z(mb_size, Z_dim)
        delta = []
        for t in range(len(G.theta_G)):
            delta.append(np.random.normal(loc=0.0, scale=v,
                                        size=G.size[t]))
        for t in range(len(G.theta_G)):
            sess.run(update_Sp[t], feed_dict={delta_ph[t]: delta[t]})
        reward = sess.run(S_loss, feed_dict={S.Z: sample, C: C_mb})
        reward_list_2.append(reward)
        for t in range(len(G.theta_G)):
            sess.run(update_Sn[t], feed_dict={delta_ph[t]: delta[t]})
        tmp = sess.run(S_loss, feed_dict={S.Z: sample, C: C_mb})
        reward_list_2.append(tmp)
        reward = reward - tmp

        reward_list.append(reward)
        delta_list.append(delta)

    if args.sigma == 1:
        sigma_R = np.std(reward_list_2)
    else:
        sigma_R = 1.
    if sigma_R == 0:
        sigma_R = 1.

    for m in range(search_num):
        if m == 0:
            for t in range(len(G.theta_G)):
                update.append(reward_list[m] * delta_list[m][t])
        else:
            for t in range(len(G.theta_G)):
                update[t] += reward_list[m] * delta_list[m][t]
    for t in range(len(G.theta_G)):
        sess.run(update_G[t], feed_dict={update_Gph[t]: update[t] * alpha / (search_num * sigma_R)})

    G_loss_curr = sess.run(G_loss, feed_dict={G.Z: sample, C:C_mb})
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={G.Z: sample, X: X_mb, C: C_mb})

    if it % 10 == 0:
        print('Iter: {}'.format(it))
        for t in range(1):
            print(update[0] * alpha / search_num)
            print(sess.run(G.theta_G[0]))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print('Sigma_R: {:.4}'.format(sigma_R))
        out_file.write('Iter: {}'.format(it)+":\n"
            + 'D loss: {:.4}'.format(D_loss_curr) + "\n"
            + 'G_loss: {:.4}'.format(G_loss_curr) + "\n"
            + 'Sigma_R: {:.4}'.format(sigma_R) + "\n")
        print()
