import warnings
warnings.filterwarnings('ignore')
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
from scDREAMER.utils import dense,lrelu,zinb_model,eval_cluster_on_test,load_gene_mtx
import pandas as pd


# +

def build_model(self):

    """
    Build the complete tensorflow network of the model
    """

    self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.X_dim], name='Input')
    self.x_input_ = tf.placeholder(dtype=tf.float32, shape=[None, self.X_dim], name='Input')
    self.x_target = tf.placeholder(dtype=tf.float32, shape=[None, self.X_dim], name='Target')
    self.batch_input = tf.placeholder(dtype = tf.float32, shape=[None, self.N_batch], name='batch_input') # 6, 3
    self.batch_input_ = tf.placeholder(dtype = tf.float32, shape=[None, self.N_batch], name='batch_input')
    
    self.keep_prob = tf.placeholder(dtype=tf.float32, name = 'keep_prob')
    self.real_distribution = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim], name='Real_distribution')
    self.kl_scale = tf.placeholder(tf.float32, (), name='kl_scale')
    
    self.kl_scale = 0.001 # 0.01, 0, 0.001
    self.dropout_rate = 0.1
    self.training_phase = True 
    self.n_layers = self.num_layers 
    self.n_latent = self.z_dim
    
    self.enc_input = tf.concat([self.x_input, self.batch_input],1)
    self.enc_input_ = tf.concat([self.x_input_, self.batch_input_],1)
    
    print('encoder input shape ',self.enc_input)

    self.encoder_output, self.z_post_m, self.z_post_v, self.l_post_m, self.l_post_v = self.encoder(self.enc_input) # self.x_input
    self.encoder_output_, self.z_post_m_, self.z_post_v_, self.l_post_m_, self.l_post_v_ = self.encoder(self.enc_input_, reuse = True) 

    self.expression = self.x_input               
    self.proj = tf.placeholder(dtype=tf.float32, shape=[None, self.X_dim], name='projection')
  
    log_library_size = np.log(np.sum(self.data_train, axis=1)+1) 
    mean, variance = np.mean(log_library_size), np.var(log_library_size)

    library_size_mean = mean
    library_size_variance = variance
    self.library_size_mean = tf.to_float(tf.constant(library_size_mean))
    self.library_size_variance = tf.to_float(tf.constant(library_size_variance))    
    
    self.z = self.sample_gaussian(self.z_post_m, self.z_post_v) 
    self.z_ = self.sample_gaussian(self.z_post_m_, self.z_post_v_) 

    self.library = self.sample_gaussian(self.l_post_m, self.l_post_v)
    

    print('decoder input shape ',tf.concat([self.z, self.batch_input],1))

    self.decoder_output = self.decoder(tf.concat([self.z, self.batch_input],1))             
    self.n_input = self.expression.get_shape().as_list()[1]
  
    self.x_post_scale = tf.nn.softmax(dense(self.decoder_output, self.g_h_dim[0], self.n_input, name='dec_x_post_scale')) 
    self.x_post_r = tf.Variable(tf.random_normal([self.n_input]), name="dec_x_post_r")           
    self.x_post_rate = tf.exp(self.library) * self.x_post_scale
    self.x_post_dropout = dense(self.decoder_output, self.g_h_dim[0], self.n_input, name='dec_x_post_dropout') 
        
    local_dispersion = tf.exp(self.x_post_r)            
    local_l_mean = self.library_size_mean
    local_l_variance = self.library_size_variance

    self.decoder_output2 = tf.nn.sigmoid(dense(self.decoder_output, self.g_h_dim[0], self.X_dim, 'dec_output2'))  
        
    # Discriminator D2
    self.dis2_real_logit = self.discriminator2(self.x_target, self.X_dim) # True data
    self.dis2_fake_logit = self.discriminator2(self.decoder_output2, self.X_dim, reuse=True) # from decoder network
    
    # Batch Classifier
    self.disb_real_logit = self.batchClassifier(self.z, self.z_dim) # True data
    
    # Reconstruction loss 
    capL = 1e-4 #1e-8
    capU = 1e4 #1e8


    recon_loss = self.zinb_model(self.expression, self.x_post_rate, local_dispersion, self.x_post_dropout)

    
    self.kl_gauss_l = 0.5 * tf.reduce_sum(- tf.log(tf.math.minimum(tf.math.maximum(self.l_post_v, capL), capU))  \
                                      + self.l_post_v/local_l_variance \
                                      + tf.square(self.l_post_m - local_l_mean)/local_l_variance  \
                                      + tf.log(tf.math.minimum(tf.math.maximum(local_l_variance, capL), capU)) - 1, 1)

    self.kl_gauss_z = 0.5 * tf.reduce_sum(- tf.log(tf.math.minimum(tf.math.maximum(self.z_post_v, capL), capU)) + self.z_post_v + tf.square(self.z_post_m) - 1, 1)

    print ('KL gaussian z', self.kl_gauss_z)
    print ('KL gaussian l', self.kl_gauss_l)

    self.ELBO_gauss = tf.reduce_mean(recon_loss - self.kl_scale*self.kl_gauss_l - self.kl_scale*self.kl_gauss_z) - self.kl_scale*tf.reduce_sum(tf.pow(self.z - self.z_, 2)) 

    self.autoencoder_loss = - self.ELBO_gauss - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.disb_real_logit, labels = self.batch_input)) - tf.log(tf.math.minimum(tf.math.maximum(tf.reduce_sum(tf.sqrt(tf.abs(self.dis2_real_logit/tf.reduce_sum(self.dis2_real_logit)* self.dis2_fake_logit/tf.reduce_sum(self.dis2_fake_logit)))), capL), capU))                    
          
    self.dis2_loss = tf.log(tf.math.minimum(tf.math.maximum(tf.reduce_sum(tf.sqrt(tf.abs(self.dis2_real_logit/tf.reduce_sum(self.dis2_real_logit)
                                        * self.dis2_fake_logit/tf.reduce_sum(self.dis2_fake_logit)))), capL), capU)) # epsilon added to avoid Nan
    
           
    # 27Apr: AJ : minimize binary cross entropy     
    self.disb_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.disb_real_logit, labels = self.batch_input)) #self.batch_input

    t_vars = tf.trainable_variables()
    # Discriminator D2
    self.dis2_vars = [var for var in t_vars if 'dis2_' in var.name] #or 'enc_' in var.name or 'dec_' in var.name]

    self.disb_vars = [var for var in t_vars if 'disb_' in var.name ]

    self.saver = tf.train.Saver()


# -

def train_cluster(self):

    """
    Train the model and saves the integrated lowerdimentional embeedings for every 50 epoch
    """

    print('scDREAMER on DataSet {} ... '.format(self.dataset_name))


    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr_ae, beta1=self.beta1)\
    .minimize(self.autoencoder_loss)


    discriminator2_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr_dis, beta1=self.beta1)\
    .minimize(self.dis2_loss, var_list=self.dis2_vars)
        
    batchClassifier_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr_bc,beta1 = self.beta1)\
    .minimize(self.disb_loss, var_list = self.disb_vars)
    
    self.sess.run(tf.global_variables_initializer())
    
    num_batch_iter = self.total_size // self.batch_size
    indices = np.arange(self.data_train.shape[0])
    
    for ep in range(self.epoch):
    
        a_loss_epoch = []
        d2_loss_epoch = []
        db_loss_epoch = [] 
        
        self._is_train = True

        if (self.shuffle_type == 1):
            index = 0
            np.random.shuffle(indices)
        
        for it in range(num_batch_iter):

            # Selecting mini batch            
            if (self.shuffle_type == 1):
                batch_indices = indices[index : index + self.batch_size]

                batch_x = self.data_train[batch_indices, :]
                X_ = self.batch_train[batch_indices, :]         
                index += self.batch_size
                
            elif (self.shuffle_type == 2):
                batch_x, X_ = self.next_batch(self.data_train, self.batch_train, self.train_size)

            batch_x_ = batch_x.copy()
            rand_inx = random.sample(range(self.X_dim), int(0.1*self.X_dim))

            batch_x_[:,rand_inx] = 0
            
            batch_z_real_dist = self.sample_Z(self.batch_size, self.z_dim)

            _, a_loss_curr = self.sess.run([autoencoder_optimizer, self.autoencoder_loss],
                                            feed_dict = {self.x_input: batch_x, self.x_target: batch_x, 
                                                       self.x_input_: batch_x_, self.batch_input_: X_,
                                                      self.batch_input: X_,
                                                      self.keep_prob: self.keep_param}) 
  
            _, d2_loss_curr = self.sess.run([discriminator2_optimizer, self.dis2_loss],
                        feed_dict={self.x_input: batch_x,
                        self.x_target: batch_x,
                        self.batch_input: X_,
                        self.real_distribution: batch_z_real_dist,
                        self.x_input_: batch_x_, self.batch_input_: X_,
                        self.keep_prob: self.keep_param}) 
            
            _, db_loss_curr = self.sess.run([batchClassifier_optimizer, self.disb_loss],
                feed_dict={self.x_input: batch_x,
                            self.batch_input: X_,
                self.real_distribution: batch_z_real_dist,
                self.batch_input: X_, # batch_b
                self.x_input_: batch_x_, self.batch_input_: X_,
                self.keep_prob: self.keep_param})
            
            a_loss_epoch.append(a_loss_curr) # total loss getting appended 
            d2_loss_epoch.append(d2_loss_curr)
            db_loss_epoch.append(db_loss_curr)
            
            if (np.isnan(a_loss_curr) or np.isnan(db_loss_curr) or np.isnan(d2_loss_curr)):
                print("Epoch : [%d] ,  a_loss = %.4f" #, d_loss: %.4f ,  db_loss: %.4f" 
                 % (ep, np.mean(a_loss_epoch)))
                break

        self._is_train = False # enables false after 1st iterations only...to make training process fast
        
        if (np.isnan(a_loss_curr) or np.isnan(db_loss_curr) or np.isnan(d2_loss_curr)):
                print("Epoch : [%d] ,  a_loss = %.4f" #, d_loss: %.4f ,  db_loss: %.4f" 
                 % (ep, np.mean(a_loss_epoch)))#, np.mean(d2_loss_epoch), np.mean(db_loss_epoch)))
                break
                
        if (ep %10 == 0):
            print("Epoch : [%d] ,  a_loss = %.4f" #, d_loss: %.4f ,  db_loss: %.4f" 
             % (ep, np.mean(a_loss_epoch)))#, np.mean(d2_loss_epoch), np.mean(db_loss_epoch)))
        
    self.eval_cluster_on_test(self.epoch)

# +


def encoder(self, x, reuse = False):
    """
    Encoder network of VAE
    
    :param x: tensor to pass encoder shape 
    :type x: tensor
    :param reuse: reuse the weight of network(used at the time of building the network), defaults to False
    :type reuse: bool, optional
    """


    with tf.variable_scope('Encoder') as scope:
        if reuse:
            scope.reuse_variables()

        if self.is_bn:
            h = tf.layers.batch_normalization(

                lrelu(dense(x, self.X_dim + self.N_batch, self.g_h_dim[0], name='enc_h0_lin'), alpha=self.leak),
                training=self._is_train, name='enc_bn0')
                
            for i in range(1, self.num_layers):
                h = tf.layers.batch_normalization(

                    lrelu(dense(h, self.g_h_dim[i - 1], self.g_h_dim[i], name='enc_h' + str(i) + '_lin'),
                          alpha=self.leak),
                    training=self._is_train, name='enc_bn' + str(i))                    

            z_post_m = dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim, name='enc_z_post_m' + str(self.num_layers) + '_lin')                
            z_post_v = tf.exp(dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim, name='enc_z_post_v' + str(self.num_layers) + '_lin'))              
            
            h = tf.nn.relu(dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim, name='enc_h' + str(self.num_layers) + '_lin'))

            l_post_m = dense(h, self.z_dim, 1, name='enc_l_post_m' + str(self.num_layers) + '_lin')                            
            l_post_v = tf.exp(dense(h, self.z_dim, 1, name='enc_l_post_v' + str(self.num_layers) + '_lin')) 
            

        else:

            h = tf.nn.dropout(lrelu(dense(x, self.X_dim + self.N_batch, self.g_h_dim[0], name='enc_h0_lin'), alpha=self.leak),
                              keep_prob=self.keep_prob)                
            
            for i in range(1, self.num_layers):
                
                h = tf.nn.dropout(lrelu(dense(h, self.g_h_dim[i - 1], self.g_h_dim[i], name='enc_h' + str(i) + '_lin'),
                          alpha=self.leak), keep_prob=self.keep_prob)                    

            z_post_m = dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim, name='enc_z_post_m' + str(self.num_layers) + '_lin')                
            z_post_v = tf.exp(dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim, name='enc_z_post_v' + str(self.num_layers) + '_lin'))
            
            
            h = tf.nn.relu(dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim, name='enc_h' + str(self.num_layers) + '_lin'))
                      

            l_post_m = dense(h, self.z_dim, 1, name='enc_l_post_m' + str(self.num_layers) + '_lin')                             
            l_post_v = tf.exp(dense(h, self.z_dim, 1, name='enc_l_post_v' + str(self.num_layers) + '_lin'))                                          
                        
        return h, z_post_m, z_post_v, l_post_m, l_post_v


# -

def discriminator2(self, x, x_dim, reuse=False):    
    """
    Discriminator that is used to match the input x with reconstructed x.
    
    :param x: tensor of shape [batch_size, x_dim]
    :param reuse: True -> Reuse the discriminator variables,
                  False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    
    with tf.variable_scope('Discriminator2') as scope:
        if reuse:
            scope.reuse_variables()

        if self.is_bn:

            h = tf.layers.batch_normalization(
                lrelu(dense(x, x_dim, self.d_h_dim[self.num_layers - 1], name='dis2_h' + str(self.num_layers-1) + '_lin'),      
                      alpha=self.leak),
                training=self._is_train, name='dis2_bn' + str(self.num_layers-1))
            for i in range(self.num_layers - 2, -1, -1):
                h = tf.layers.batch_normalization(
                    lrelu(dense(h, self.d_h_dim[i + 1], self.d_h_dim[i], name='dis2_h' + str(i) + '_lin'),
                          alpha=self.leak),
                    training=self._is_train, name='dis2_bn' + str(i))

        else:

            h = tf.nn.dropout(
                lrelu(dense(x, x_dim, self.d_h_dim[self.num_layers - 1], name='dis2_h' + str(self.num_layers-1) + '_lin'),
                      alpha=self.leak),
                keep_prob=self.keep_prob)
            for i in range(self.num_layers - 2, -1, -1):
                h = tf.nn.dropout(
                    lrelu(dense(h, self.d_h_dim[i + 1], self.d_h_dim[i], name='dis2_h' + str(i) + '_lin'),
                          alpha=self.leak), keep_prob=self.keep_prob)

        output = dense(h, self.d_h_dim[0], 1, name='dis2_output')
        return output


def decoder(self, z, reuse=False):
    
    """
    Decoder part of the autoencoder.

    :param z: input to the decoder
    :type z: tensor
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating, defaults to False
    :type reuse: bool, optional
    :return: tensor which should ideally be the input given to the encoder.
    :rtype: tensor
    """

    with tf.variable_scope('Decoder') as scope:
        if reuse:
            scope.reuse_variables()

        if self.is_bn:

            h = tf.layers.batch_normalization(
              
                lrelu(dense(z , self.z_dim + self.N_batch, self.g_h_dim[self.num_layers-1], name='dec_h' + str(self.num_layers-1) + '_lin'),
                      alpha=self.leak),                   
                training=self._is_train, name='dec_bn' + str(self.num_layers-1))
            for i in range(self.num_layers-2, -1,-1):
                h = tf.layers.batch_normalization(

                    lrelu(dense(h, self.g_h_dim[i + 1], self.g_h_dim[i], name='dec_h' + str(i) + '_lin'),
                          alpha=self.leak),                        
                    training=self._is_train, name='dec_bn' + str(i))
        else:
            h = tf.nn.dropout(lrelu(dense(z, self.z_dim + self.N_batch, self.g_h_dim[self.num_layers-1], name='dec_h' + str(self.num_layers-1) + '_lin'),
                                    alpha=self.leak),                                  
                              keep_prob=self.keep_prob)
            for i in range(self.num_layers-2, -1, -1):
                h = tf.nn.dropout(
                    lrelu(dense(h, self.g_h_dim[i + 1], self.g_h_dim[i], name='dec_h' + str(i) + '_lin'),
                          alpha=self.leak), keep_prob=self.keep_prob)
            
        return h

def batchClassifier(self, z, z_dim, reuse = False):    
    """
    batchClassifier takes the latent space representation and try to differentiate between different batches
    :param z: tensor of shape [batch_size, z_dim]
    :param batch: tensor of shape [batch_size] -> batchinfo of the train data
    :param reuse: True -> Reuse the discriminator variables,False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    
    with tf.variable_scope('discriminatorB') as scope:
        if reuse:
            scope.reuse_variables()

        if self.is_bn:

            h = tf.layers.batch_normalization(
                lrelu(dense(z, z_dim, self.d_h_dim[self.num_layers - 1], name='disb_h' + 
                            str(self.num_layers-1) + '_lin'), alpha = self.leak),
                            training=self._is_train, name='disb_bn' + str(self.num_layers-1))
            
            for i in range(self.num_layers - 2, -1, -1):
                h = tf.layers.batch_normalization(
                    lrelu(dense(h, self.d_h_dim[i + 1], self.d_h_dim[i], name='disb_h' + str(i) + '_lin'),
                          alpha=self.leak),
                    training=self._is_train, name='disb_bn' + str(i))

        else:

            h = tf.nn.dropout(
                lrelu(dense(z, z_dim, self.d_h_dim[self.num_layers - 1], name='disb_h' + str(self.num_layers-1) + '_lin'),
                      alpha=self.leak),
                keep_prob=self.keep_prob)
            
            for i in range(self.num_layers - 2, -1, -1):
                h = tf.nn.dropout(
                    lrelu(dense(h, self.d_h_dim[i + 1], self.d_h_dim[i], name='disb_h' + str(i) + '_lin'),
                          alpha=self.leak), keep_prob=self.keep_prob)

        output = dense(h, self.d_h_dim[0], self.N_batch, name='disb_output')
        return output

class scDREAMER(object):
    """
    scDREAMER class parameter setting


        :param batch: obs key containing the batch information
        :type batch: string
        :param cell_type: obs key containing the cell type information
        :type cell_type: string
        :param name: path to the AnnData(adata) object.
            adata.X contains counts info. 
            adata.obs contains batch and celltype info
        :type name: AnnData
        :param epoch: number of epoches to train the model, defaults to 300
        :type epoch: int, optional
        :param lr_ae: learning rate, defaults to 0.0007
        :type lr_ae: float, optional
        :param lr_bc: learning rate, defaults to 0.0007
        :type lr_bc: float, optional
        :param lr_dis: learning rate, defaults to 0.0007
        :type lr_dis: float, optional
        :param beta1: beta1, defaults to 0.9
        :type beta1: float, optional
        :param batch_size: batch size, defaults to 128
        :type batch_size: int, optional
        :param X_dim: Top heighly variable genes, defaults to 2000
        :type X_dim: int, optional
        :param z_dim: Embeddings dimention, defaults to 10
        :type z_dim: int, optional
        :param dataset_name: dataset name, defaults to "Pancreas"
        :type dataset_name: str, optional
        :param num_layers: number of hidden layers, defaults to 1
        :type num_layers: int, optional
        :param g_h_dim: neurons in encoder hidden layers (uses only num_layers), defaults to [512, 256, 0, 0]
        :type g_h_dim: list, optional
        :param d_h_dim: neurons in decoder hidden layers (uses only num_layers), defaults to [512, 256, 0, 0]
        :type d_h_dim: list, optional
        :param gen_activation: activation function, defaults to "sig"
        :type gen_activation: str, optional
        :param leak: leak, defaults to 0.2
        :type leak: float, optional
        :param keep_param: keep, defaults to 0.9
        :type keep_param: float, optional
        :param trans: translation, defaults to "sparse"
        :type trans: str, optional
        :param sampler: z sampler, defaults to "normal"
        :type sampler: str, optional
        :param shuffle_type: type of shuffling, defaults to 1
        :type shuffle_type: int, optional
    """

    def __init__(self, sess, batch, cell_type, name, epoch = 300, lr_ae = 0.0002, lr_dis = 0.0007,lr_bc=0.0007, 
                 beta1=0.9, batch_size=128, X_dim=2000, z_dim=10, dataset_name='Pancreas',
                 num_layers = 1, g_h_dim = [512, 256, 0, 0], d_h_dim = [512, 256, 0, 0], 
                 gen_activation = 'sig', leak = 0.2, keep_param = 0.9, trans = 'sparse',
                 is_bn = False, g_iter = 2, lam=1.0, sampler = 'normal', shuffle_type = 1):    
        
        self.sess = sess
        self.epoch = epoch
        self.lr_bc = lr_bc
        self.lr_ae = lr_ae
        self.lr_dis = lr_dis
        self.beta1 = beta1
        self.batch_size = batch_size
        self.X_dim = X_dim
        self.z_dim = z_dim
        self.dataset_name = dataset_name
        self.num_layers = num_layers
        self.g_h_dim = g_h_dim  # Fully connected layers for Generator
        self.d_h_dim = d_h_dim  # Fully connected layers for Discriminator
        self.gen_activation = gen_activation
        self.leak = leak
        self.keep_param = keep_param
        self.trans = trans
        self.is_bn = is_bn
        self.g_iter = g_iter
        self.lam = lam
        self.sampler = sampler
        self.eps = 0.001
        self._is_train = False
        self.n_hidden = 128 
        self.batch = batch
        self.cell_type = cell_type
        self.name = name
        self.shuffle_type = shuffle_type
        
        if self.trans == 'sparse':
            
            self.data_train, self.data_test, self.scale, self.labels_train, self.labels_test, \
            self.batch_train, self.batch_test, self.batch_info = load_gene_mtx(self.dataset_name, \
            transform=False, count=False, actv=self.gen_activation, batch = self.batch, \
            cell_type = self.cell_type, name = self.name)
            
            self.N_batch = self.batch_train.shape[1]
            
        else:
            self.data_train, self.data_test, self.labels_train, self.labels_val, self.labels_test  = load_gene_mtx(self.dataset_name, transform=True, batch = self.batch, cell_type = self.cell_type, name = self.name)
            self.scale = 1.0
           
        if self.gen_activation == 'tanh':
            self.data = 2* self.data - 1
            self.data_train = 2 * self.data_train - 1
            self.data_val = 2 * self.data_val - 1

        self.train_size = self.data_train.shape[0]
        self.test_size = self.data_test.shape[0]
        self.total_size = self.test_size
    
        self.build_model()

    build_model = build_model
    train_cluster = train_cluster
    encoder = encoder
    decoder = decoder
    batchClassifier = batchClassifier
    discriminator2 = discriminator2
    eval_cluster_on_test = eval_cluster_on_test
    zinb_model = zinb_model
    
    
    @property
    def model_dir(self):
        s = "DRA_{}_{}_b_{}_g{}_d{}_{}_{}_lr_{}_b1_{}_leak_{}_keep_{}_z_{}_{}_bn_{}_lam_{}_giter_{}_epoch_{}".format(
            datetime.datetime.now(), self.dataset_name, 
            self.batch_size, self.g_h_dim, self.d_h_dim, self.gen_activation, self.trans, self.lr, 
            self.beta1, self.leak, self.keep_param, self.z_dim, self.sampler, self.is_bn,
            self.lam, self.g_iter, self.epoch) 
        s = s.replace('[', '_')
        s = s.replace(']', '_')
        s = s.replace(' ', '')
        return s

    def sample_Z(self, m, n, sampler='uniform'):
        if self.sampler == 'uniform':
            return np.random.uniform(-1., 1., size=[m, n])
        elif self.sampler == 'normal':
            return np.random.randn(m, n)

    def next_batch(self, data, batch_info, max_size):

        indx = np.random.randint(max_size - self.batch_size)
        return data[indx:(indx + self.batch_size), :], batch_info[indx:(indx + self.batch_size), :]
        

    def next_batch_(self, data, max_size):
        #data = data.sample(frac = 1)
        indx = np.random.randint(max_size - self.batch_size)
        return data[indx:(indx + self.batch_size), :]

    def sample_gaussian(self, mean, variance, scope=None):

        with tf.variable_scope(scope, 'sample_gaussian'):
            sample = tf.random_normal(tf.shape(mean), mean, tf.sqrt(variance))
            sample.set_shape(mean.get_shape())
            return sample
