import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import time
from scipy import stats

class Config:
    def __init__(self):
        self.input_dim = 1
        self.gen_hidden = [20, 10, 5]
        self.gen_out = 1
        self.dis_hidden = [128, 1]
        self.sample_num = 128
        self.batch_size = 128
        self.n_epochs = 100000
        self.lr = 1e-4
        self.k = 1


class GANModel:
    def __init__(self, config):
        self.config = config
        self.build()
    
    def fit(self, sess, inputs, true_dis):
        losses = {}
        losses["gen"] = []
        losses["dis"] = []
        start_time = time.time()
        # # slice the batches
        # if len(inputs) != self.config.sample_num:
        #     print "Sample num inconsistent"
        #     return
        # batch_size = self.config.batch_size
        # input_batch_list = []
        # true_dis_batch_list = []
        # for batch_num in range(self.config.sample_num / batch_size):
        #     input_batch_list.append(np.reshape(inputs[batch_num * batch_size: (batch_num + 1)* batch_size], (batch_size, 1)))
        #     true_dis_batch_list.append(np.reshape(true_dis[batch_num * batch_size: (batch_num + 1)* batch_size], (batch_size, 1)))  
        # # run
        # n_minibatches = len(input_batch_list)
        for epoch in range(self.config.n_epochs):
            feed = {
                self.input_placeholder: np.random.uniform(low=-1, high=1, size=(64, 1)),
                self.true_distribution_placeholder: np.random.normal(loc = -5, scale = 2, size=(64, 1))
                }
            # run the generator
            if epoch % self.config.k == 0:
                total_loss_gen = 0
                total_loss_dis = 0
                # for batch_num in range(n_minibatches):
                    # feed = {self.input_placeholder:input_batch_list[batch_num], self.true_distribution_placeholder: true_dis_batch_list[batch_num]}
                _, loss_dis, loss_gen = sess.run([self.train_op_gen, self.loss_dis, self.loss_gen], feed_dict=feed)
                # total_loss_gen += loss_gen
                # total_loss_dis += loss_dis
                # losses["gen"].append(total_loss_gen / n_minibatches)
                # losses["dis"].append(total_loss_dis / n_minibatches)
            # run the discriminitor
            total_loss_gen = 0
            total_loss_dis = 0
            # for batch_num in range(n_minibatches):
                # feed = {self.input_placeholder:input_batch_list[batch_num], self.true_distribution_placeholder: true_dis_batch_list[batch_num]}
            _, loss_dis, loss_gen = sess.run([self.train_op_dis, self.loss_dis, self.loss_gen], feed_dict=feed)
            losses["gen"].append(loss_gen)
            losses["dis"].append(loss_dis)
            # total_loss_gen += loss_gen
            # total_loss_dis += loss_dis
            # losses["gen"].append(total_loss_gen / n_minibatches)
            # losses["dis"].append(total_loss_dis / n_minibatches)
            # print and draw figure
            if epoch % 1000 == 0:
                duration = time.time() - start_time
                # import pdb; pdb.set_trace()
                print ('Epoch {:}: gen_loss = {:.4f} dis_loss = {:.4f} ({:.3f} sec)'.format(epoch, losses["gen"][-1], losses["dis"][-1], duration))
                start_time = time.time()
                plt.figure(epoch)
                self.run_display(sess, inputs, true_dis)
        return losses

    # def run_epoch(self, sess, inputs, true_dis, train_op):
    #     """Runs an epoch of training.

    #     Args:
    #         sess: tf.Session() object
    #         inputs: np.ndarray of shape (batch_num, batch_size, input_dim)
    #         labels: np.ndarray of shape (batch_num, batch_size, n_classes)
    #         train_op: trainning operator
    #     Returns:
    #         average_loss: scalar. Average minibatch loss of model on epoch.
    #     """
       
    #     for batch_num in range(n_minibatches):
    #         feed = {self.input_placeholder:inputs[batch_num], self.true_distribution_placeholder: true_dis[batch_num]}
    #         _, loss = sess.run([train_op, self.loss], feed_dict=feed)
    #         total_loss += loss
    #     return total_loss / n_minibatches
    
    def run_display(self, sess, inputs, true_dis):
        feed = {
            self.input_placeholder: np.reshape(inputs, (self.config.sample_num, self.config.input_dim)),
            self.true_distribution_placeholder: np.reshape(true_dis, (self.config.sample_num, self.config.input_dim))
            }
        y_hat, D = sess.run([self.gen_pred, tf.nn.sigmoid(self.dis_fake)], feed_dict=feed)
        y_hat = np.reshape(y_hat, (self.config.sample_num))
        try:
            kernel = stats.gaussian_kde(y_hat, bw_method=1)
            plt.plot(np.arange(-10,0,0.1), kernel(np.arange(-10,0,0.1)))
            plt.plot(y_hat, D)
            plt.show(block=False)
        except:
            import pdb; pdb.set_trace()

    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(shape = [None, self.config.input_dim], dtype = tf.float32)
        self.true_distribution_placeholder = tf.placeholder(shape = [None, self.config.input_dim], dtype = tf.float32)

    def add_generator_op(self):
        """
        input data:
            batch_size, input_dim
        return:
            batch_size, output_dim
        """
        gen_layer1 = {}
        gen_layer2 = {}
        gen_layer3 = {}
        gen_layer4 = {}
        gen_layer1["W"] = tf.Variable(initial_value = tf.random_normal(shape = [self.config.input_dim, self.config.gen_hidden[0]], mean = 0.0, stddev = 1.0), name = 'generator_weight' )
        gen_layer1["b"] = tf.Variable(initial_value = tf.random_normal(shape = [self.config.gen_hidden[0]] , mean = 0.0, stddev = 1.0), name = 'generator_bias' ) 
        gen_layer1["output"] = tf.nn.sigmoid(tf.add(tf.matmul(self.input_placeholder, gen_layer1["W"]), gen_layer1["b"]))
        gen_layer2["W"] = tf.Variable(initial_value = tf.random_normal(shape = [self.config.gen_hidden[0], self.config.gen_hidden[1]], mean = 0.0, stddev = 1.0), name = 'generator_weight' )
        gen_layer2["b"] = tf.Variable(initial_value = tf.random_normal(shape = [self.config.gen_hidden[1]] , mean = 0.0, stddev = 1.0), name = 'generator_bias' ) 
        gen_layer2["output"] = tf.nn.sigmoid(tf.add(tf.matmul(gen_layer1["output"], gen_layer2["W"]), gen_layer2["b"]))
        gen_layer3["W"] = tf.Variable(initial_value = tf.random_normal(shape = [self.config.gen_hidden[1], self.config.gen_hidden[2]], mean = 0.0, stddev = 1.0), name = 'generator_weight' )
        gen_layer3["b"] = tf.Variable(initial_value = tf.random_normal(shape = [self.config.gen_hidden[2]] , mean = 0.0, stddev = 1.0), name = 'generator_bias' ) 
        gen_layer3["output"] = tf.nn.sigmoid(tf.add(tf.matmul(gen_layer2["output"], gen_layer3["W"]), gen_layer3["b"]))
        gen_layer4["W"] = tf.Variable(initial_value = tf.random_normal(shape = [self.config.gen_hidden[2], self.config.gen_out], mean = 0.0, stddev = 1.0), name = 'generator_weight' )
        gen_layer4["b"] = tf.Variable(initial_value = tf.random_normal(shape = [self.config.gen_out] , mean = 0.0, stddev = 1.0), name = 'generator_bias' ) 
        gen_layer4["output"] = tf.add(tf.matmul(gen_layer3["output"], gen_layer4["W"]), gen_layer4["b"])
        return gen_layer4["output"]
        
    def add_discriminator_network(self):
        """
        input data:
            batch_size, input_dim
        return:
            batch_size, output_dim
        """
        self.dis_layer1 = {}
        self.dis_layer2 = {}
        self.dis_layer3 = {}
        self.dis_layer1["W"] = tf.Variable(initial_value = tf.random_normal(shape = [self.config.input_dim, self.config.dis_hidden[0]], mean = 0.0, stddev = 1.0), name = 'discriminator_weight' )
        self.dis_layer1["b"] = tf.Variable(initial_value = tf.random_normal(shape = [self.config.dis_hidden[0]] , mean = 0.0, stddev = 1.0), name = 'discriminator_bias' ) 
        self.dis_layer2["W"] = tf.Variable(initial_value = tf.random_normal(shape = [self.config.dis_hidden[0], self.config.dis_hidden[1]], mean = 0.0, stddev = 1.0), name = 'discriminator_weight' )
        self.dis_layer2["b"] = tf.Variable(initial_value = tf.random_normal(shape = [self.config.dis_hidden[1]] , mean = 0.0, stddev = 1.0), name = 'discriminator_bias' ) 
        self.dis_layer3["W"] = tf.Variable(initial_value = tf.random_normal(shape = [self.config.dis_hidden[1], 1], mean = 0.0, stddev = 1.0), name = 'discriminator_weight' )
        self.dis_layer3["b"] = tf.Variable(initial_value = tf.random_normal(shape = [1] , mean = 0.0, stddev = 1.0), name = 'discriminator_bias' )

    def add_discriminator_op(self, input):
        output1 = tf.nn.relu(tf.add(tf.matmul(input, self.dis_layer1["W"]), self.dis_layer1["b"]))
        output2 = tf.nn.sigmoid(tf.add(tf.matmul(output1, self.dis_layer2["W"]), self.dis_layer2["b"]))
        # output2 = tf.add(tf.matmul(output1, self.dis_layer2["W"]), self.dis_layer2["b"])
        output3 = tf.nn.sigmoid(tf.add(tf.matmul(output2, self.dis_layer3["W"]), self.dis_layer3["b"]))
        return output2

    def add_loss_op(self):
        a = tf.reduce_mean(tf.log(tf.ones_like(self.dis_fake) - self.dis_fake))
        b = tf.reduce_mean(tf.log(self.dis_fake))
        c = tf.reduce_mean(tf.log(self.dis_true))
        self.loss_dis = -c - a
        self.loss_gen = -b
        # D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_true, labels=tf.ones_like(self.dis_true)))
        # D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_fake, labels=tf.zeros_like(self.dis_fake)))
        # self.loss_dis = D_loss_real + D_loss_fake
        # self.loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_fake, labels=tf.ones_like(self.dis_fake)))

    def add_train_op(self):
        vars = tf.trainable_variables()
        dis_var = [v for v in vars if v.name.startswith("discriminator")]
        gen_var = [v for v in vars if v.name.startswith("generator")]
        self.train_op_dis = tf.train.AdamOptimizer().minimize(self.loss_dis, var_list = dis_var)
        self.train_op_gen = tf.train.AdamOptimizer().minimize(self.loss_gen, var_list = gen_var)

    def build(self):
        self.add_placeholder()
        with tf.device("/device:GPU:0"):
            self.gen_pred = self.add_generator_op()
            self.add_discriminator_network()
            self.dis_fake = self.add_discriminator_op(self.gen_pred)
            self.dis_true = self.add_discriminator_op(self.true_distribution_placeholder)
            self.add_loss_op()
            self.add_train_op()
        # add training ops 

def test_gan():
    config = Config()
    #generate the training data
    Z = np.random.uniform(low=-1, high=1, size=config.sample_num)
    Y = np.random.normal(loc = -5, scale = 2, size=config.sample_num)
    kernel = stats.gaussian_kde(Y, bw_method=1)
    plt.figure(1)
    plt.plot(np.arange(-10,0,0.1), kernel(np.arange(-10,0,0.1)))
    plt.show(block=False)
    kernel = stats.gaussian_kde(Z, bw_method=1)
    plt.figure(2)
    plt.plot(np.arange(-10,0,0.1), kernel(np.arange(-10,0,0.1)))
    plt.show(block=False)
    # import pdb; pdb.set_trace()
    tf.device('/device:GPU:0')
    with tf.Graph().as_default():
        # Build the model and add the variable initializer Op
        model = GANModel(config)
        init = tf.global_variables_initializer()
        # Create a session for running Ops in the Graph
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            # Run the Op to initialize the variables.
            sess.run(init)
            # Fit the model
            losses = model.fit(sess, Z, Y)

    plt.figure(1234)
    plt.plot(losses["gen"])
    plt.plot(losses["dis"])
    plt.show(block=False)
    import pdb; pdb.set_trace()
if __name__ == "__main__":
    test_gan()