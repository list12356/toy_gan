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
parser.add_argument('--dir', default="out_brs")
args = parser.parse_args()

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

out_dir = args.dir
save_step = 10
data_dim = 10 
Z_dim = 10
hidden_dim = 20
search_num = 20 
top_num = 100 # only top performance is used
alpha = 1
v = 0.02
mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)
restore = False 
D_lr = 1e-4
truth = np.array([12.,31.,27.,-2.,-34.,38.,-14.,-42.,20.,7])

X = tf.placeholder(tf.float32, shape=[None, data_dim])

D_W1 = tf.Variable(xavier_init([data_dim, 20]))
D_b1 = tf.Variable(tf.zeros(shape=[20]))

D_W2 = tf.Variable(xavier_init([20, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

class Generator:
    def __init__(self):
        self.G_W1 = tf.Variable(xavier_init([Z_dim, 20]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[20]))

        self.G_W2 = tf.Variable(xavier_init([20, data_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[data_dim]))

        self.Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]
        self.size = [[Z_dim, 20], [20,data_dim], [20], [data_dim]]
        self.build()
    
    def build(self):
        G_h1 = tf.nn.relu(tf.matmul(self.Z, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        self.G_sample = G_log_prob

    def update(self):
        return
        


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(tf.to_float(x), D_W1) + D_b1)
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
G_sample = G.G_sample

# served as a searcher
S = Generator()
S_sample = S.G_sample

D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(D_real, 1e-16, 1.0)) + tf.log(tf.clip_by_value(1. - D_fake, 1e-3, 1.0)))
G_loss = tf.reduce_mean(tf.log(tf.clip_by_value(D_fake, 1e-16, 1.0))) * 1000

S_fake, S_logit_fake = discriminator(S_sample)

S_loss = tf.reduce_mean(tf.log(tf.clip_by_value(S_fake, 1e-16, 1.0))) * 1000
# Alternative losses:
# -------------------
# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
# D_loss = D_loss_real + D_loss_fake
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

# D_solver = tf.train.GradientDescentOptimizer(learning_rate=D_lr).minimize(D_loss, var_list=theta_D)
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 128

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

sigma_R = 0


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

i = 0
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
        samples = sess.run(G_sample, feed_dict={G.Z: sample_Z(16, Z_dim)})
        # plt.figure(it)
        # plt.plot(np.arange(data_dim), np.sum(samples, 0))
        # plt.show(block=False)
        # fig = plot(samples)
        # plt.savefig(out_dir + '/{}.png'.format(str(i).zfill(5)), bbox_inches='tight')
        # plt.close(fig)
        # X_mb, _ = mnist.train.next_batch(16)
        # X_mb = (X_mb > 0.5).astype(int)
        # fig2 = plot(X_mb)
        # plt.savefig(out_dir + '/{}_real.png'.format(str(i).zfill(5)), bbox_inches='tight')
        # plt.close(fig2)
        # saver.save(sess, out_dir + '/{}_model.ckpt'.format(str(i).zfill(5)))
        out_file.write(str(i)+":\n"+str(samples)+ "\n"+str(np.mean(samples, axis=0))+"\n")
        print(np.mean(samples, axis=0))
        i += 1

    X_mb = np.random.normal(loc=[truth]*mb_size)
    sample = sample_Z(mb_size, Z_dim)

    update = []
    reward_list = []
    delta_list = []
    for m in range(search_num):
        delta = []
        for t in range(len(G.theta_G)):
            delta.append(np.random.normal(loc=0.0, scale=v,
                                        size=G.size[t]))
        for t in range(len(G.theta_G)):
            sess.run(update_Sp[t], feed_dict={delta_ph[t]: delta[t]})
        reward = sess.run(S_loss, feed_dict={S.Z: sample})
        for t in range(len(G.theta_G)):
            sess.run(update_Sn[t], feed_dict={delta_ph[t]: delta[t]})
        reward = reward - sess.run(S_loss, feed_dict={S.Z: sample})
        # if m == 0: 
        #     for t in range(len(G.theta_G)):
        #         update.append(reward * delta[t])
        # else:
        #     for t in range(len(G.theta_G)):
        #         update[t] += reward * delta[t]
        reward_list.append(reward)
        delta_list.append(delta)

    sigma_R = np.std(reward_list)
    if sigma_R == 0:
        sigma_R = 1
    #TODO: rank and make the selection
    sort_ind = np.argsort(reward_list)
    sort_ind = sort_ind[:top_num]
    for m in range(len(sort_ind)):
        ind = sort_ind[m]
        if m == 0: 
            for t in range(len(G.theta_G)):
                update.append(reward_list[ind] * delta_list[ind][t])
        else:
            for t in range(len(G.theta_G)):
                update[t] += reward_list[ind] * delta_list[ind][t]

    for t in range(len(G.theta_G)):
        sess.run(update_G[t], feed_dict={update_Gph[t]: update[t] * alpha / (top_num*sigma_R)})
    if np.isnan(sess.run(G.theta_G[0])).any():
        import pdb;pdb.set_trace()

    G_loss_curr = sess.run(G_loss, feed_dict={G.Z: sample})
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={G.Z: sample, X: X_mb})

    if it % 10 == 0:
        print('Iter: {}'.format(it))
        # for t in range(1):
        #     print(update[0] * alpha / (top_num*sigma_R))
        #     print(sess.run(G.theta_G[0]))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
