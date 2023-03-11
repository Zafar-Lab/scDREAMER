import tensorflow as tf
import numpy as np
from utils_new import dense,lrelu,zinb_model,eval_cluster_on_test,load_gene_mtx
import pandas as pd

# Class Functions:
def build_model(self):

    self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.X_dim], name='Input')
    #self.x_input_ = tf.placeholder(dtype=tf.float32, shape=[None, self.X_dim], name='Input')
    self.x_target = tf.placeholder(dtype=tf.float32, shape=[None, self.X_dim], name='Target')
    self.batch_input = tf.placeholder(dtype = tf.float32, shape=[None, self.N_batch], name='batch_input') # 6, 3
    #self.batch_input_ = tf.placeholder(dtype = tf.float32, shape=[None, self.N_batch], name='batch_input')
    self.labels = tf.placeholder(dtype = tf.float32, shape=[None, self.N_celltype], name='celltype_input')
    
    self.keep_prob = tf.placeholder(dtype=tf.float32, name = 'keep_prob')
    self.real_distribution = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim], name='Real_distribution')
    self.kl_scale = tf.placeholder(tf.float32, (), name='kl_scale')
    
    self.kl_scale = 0.001 # 0.01, 0, 0.001
    self.dropout_rate = 0.1
    self.training_phase = True
    self.n_layers = self.num_layers
    self.n_latent = self.z_dim
    
    # AJ
    self.enc_input = tf.concat([self.x_input, self.batch_input],1)
    #self.enc_input_ = tf.concat([self.x_input_, self.batch_input_],1)
    print('encoder input shape ',self.enc_input)

    # AJ: Encoder output...
    self.encoder_output, self.z_post_m, self.z_post_v, self.l_post_m, self.l_post_v = self.encoder(self.enc_input) # self.x_input
    #self.encoder_output_, self.z_post_m_, self.z_post_v_, self.l_post_m_, self.l_post_v_ = self.encoder(self.enc_input_, reuse = True) 

    self.expression = self.x_input               
    self.proj = tf.placeholder(dtype=tf.float32, shape=[None, self.X_dim], name='projection')
  
    log_library_size = np.log(np.sum(self.data_train, axis=1)) 
    mean, variance = np.mean(log_library_size), np.var(log_library_size)

    library_size_mean = mean
    library_size_variance = variance
    self.library_size_mean = tf.to_float(tf.constant(library_size_mean))
    self.library_size_variance = tf.to_float(tf.constant(library_size_variance))    
    
    self.z = self.sample_gaussian(self.z_post_m, self.z_post_v) 
    #self.z_ = self.sample_gaussian(self.z_post_m_, self.z_post_v_) 

    self.library = self.sample_gaussian(self.l_post_m, self.l_post_v)
    
    # AJ
    #self.decoder_output = self.decoder(self.z)  
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
    
    # Discriminator D_batch : discriminate between different batches info
    # pass the encoded data; self.x_target, self.X_dim
    
    self.disb_real_logit = self.batchClassifier(self.z, self.z_dim) # True data
    
    self.classifier_logit = dense(self.z, self.z_dim, self.N_celltype, name = 'classifier')
    #self.classifier_logit = self.classifier(self.z)
    self.encoderU_output, self.u_post_m, self.u_post_v = self.encoderU(tf.concat([self.z, self.labels], 1)) 
    self.u = self.sample_gaussian(self.u_post_m, self.u_post_v)
    
    print ("No of cell types", self.N_celltype)
    print("classifier logit is: ", self.classifier_logit)
    print("u is: ", self.u)
    print("z is: ", self.z)
    
    self.zconst_m, self.zconst_v = self.decoderU(tf.concat([self.u, self.labels], 1)) 
 
    
    ### Loss functions below: 
    # Reconstruction loss 
    capL = 1e-4 #1e-8
    capU = 1e4 #1e8

    
    #recon_loss = 0
    recon_loss = self.zinb_model(self.expression, self.x_post_rate, local_dispersion, self.x_post_dropout)
    
    # LLZ
    #recons_lossZ = (tf.constant(np.pi) * self.zconst_v) * np.exp(-0.5*((z_numpy - self.zconst_m)/self.zconst_v)**2)
    #expr = -0.5 * tf.square((self.z - self.zconst_m)/self.zconst_v) - tf.log(self.zconst_v)
    #recons_lossZ = tf.reduce_sum(expr, axis = -1)
    
    self.kl_gauss_l = 0.5 * tf.reduce_sum(- tf.log(tf.math.minimum(tf.math.maximum(self.l_post_v, capL), capU))  \
                                      + self.l_post_v/local_l_variance \
                                      + tf.square(self.l_post_m - local_l_mean)/local_l_variance  \
                                      + tf.log(tf.math.minimum(tf.math.maximum(local_l_variance, capL), capU)) - 1, 1)

    self.kl_gauss_z = 0.5 * tf.reduce_sum(- tf.log(tf.math.minimum(tf.math.maximum(self.z_post_v, capL), capU))  \
                                      + self.z_post_v/self.zconst_v \
                                      + tf.square(self.z_post_m - self.zconst_m)/self.zconst_v  \
                                      + tf.log(tf.math.minimum(tf.math.maximum(self.zconst_v, capL), capU)) - 1, 1)
       
    self.kl_gauss_u = 0.5 * tf.reduce_sum(- tf.log(tf.math.minimum(tf.math.maximum(self.u_post_v, capL), capU)) + self.u_post_v + tf.square(self.u_post_m) - 1, 1)
    
    #self.kl_gauss_z = 0.5 * tf.reduce_sum(- tf.log(tf.math.minimum(tf.math.maximum(self.z_post_v, capL), capU)) + self.z_post_v + tf.square(self.z_post_m) - 1, 1)

    print ('KL gaussian z', self.kl_gauss_z)
    print ('KL gaussian l', self.kl_gauss_l)

    # Evidence lower bound - ELBO : KLscale to prevent posterior collapse...
    #self.ELBO_gauss = tf.reduce_mean(recon_loss - self.kl_gauss_l - self.kl_scale * self.kl_gauss_z) - tf.reduce_sum(tf.pow(self.z - self.z_, 2)) 
    
    #recons_lossZ
    self.ELBO_gauss = tf.reduce_mean(recon_loss - self.kl_scale*self.kl_gauss_l - self.kl_scale*self.kl_gauss_z - self.kl_scale*self.kl_gauss_u) #- self.kl_scale*tf.reduce_sum(tf.pow(self.z - self.z_, 2)) 

    
    # - is added to ELBO because we maximize the ELBO expression & maximize classifier cross entropy loss -> -loss minimize cross entropy
    
    self.classifier_loss = 10*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.classifier_logit, labels = self.labels))
        
    self.autoencoder_loss = - self.ELBO_gauss - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.disb_real_logit, labels = self.batch_input))- tf.log(tf.math.minimum(tf.math.maximum(tf.reduce_sum(tf.sqrt(tf.abs(self.dis2_real_logit/tf.reduce_sum(self.dis2_real_logit)* self.dis2_fake_logit/tf.reduce_sum(self.dis2_fake_logit)))), capL), capU)) + self.classifier_loss           
                   
    # Discriminator D2: minimize distance min max objective function - BD distance between X(generated sample) and X input 
    
    self.dis2_loss = tf.log(tf.math.minimum(tf.math.maximum(tf.reduce_sum(tf.sqrt(tf.abs(self.dis2_real_logit/tf.reduce_sum(self.dis2_real_logit)
                                        * self.dis2_fake_logit/tf.reduce_sum(self.dis2_fake_logit)))), capL), capU)) # epsilon added to avoid Nan
    
         
    # 27Apr: AJ : minimize binary cross entropy     
    self.disb_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.disb_real_logit, labels = self.batch_input)) #self.batch_input
    

                                          
    t_vars = tf.trainable_variables()
    #self.gen_vars = [var for var in t_vars if 'enc_' in var.name or 'dec_' in var.name] #AS:2109

    #Discriminator D2
    self.dis2_vars = [var for var in t_vars if 'dis2_' in var.name] #or 'enc_' in var.name or 'dec_' in var.name]
    
    # Discriminator DB: AJ
    self.disb_vars = [var for var in t_vars if 'disb_' in var.name ]

    self.saver = tf.train.Saver()

def train_cluster(self):

    print('Cluster DRA on DataSet {} ... '.format(self.dataset_name))

    # learning_rate=self.lr,beta1=self.beta1 #0.0002 # later 0.00005 for nan issue in immune human
    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001,beta1 = self.beta1).minimize(self.autoencoder_loss) #self.disb_vars, var_list = self.gen_vars

    # Discriminator D2
    
    discriminator2_optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001,
                                                      beta1=self.beta1).minimize(self.dis2_loss,
                                                                                  var_list=self.dis2_vars)
    
    # Discriminator batch: Classifier....
    
    batchClassifier_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr,
                                              beta1 = self.beta1).minimize(self.disb_loss, var_list = self.disb_vars)
    
    #classifier_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr, beta1 = self.beta1).minimize(self.classifier_loss)
    
    
    self.sess.run(tf.global_variables_initializer())
    a_loss_epoch = []
    d2_loss_epoch = [] # Discriminator D2
    db_loss_epoch = [] # Discriminator batch

    control = 3 # Generator is updated twice for each Discriminator D1 update

    num_batch_iter = self.total_size // self.batch_size
    
    for ep in range(self.epoch):
    #for it in range(num_batch_iter):
        d_loss_curr = g_loss_curr = a_loss_curr = np.inf
        self._is_train = True

        # AJ: oncatenating x_train and batch info for batches preparation
        X = np.concatenate((self.data_train, self.batch_train), axis = 1)


        for it in range(num_batch_iter):

            batch_x, X_, labels_ = self.next_batch(self.data_train, self.batch_train, self.labels_enc, self.train_size)

            #df = pd.DataFrame(batch_x)
            #col = df.columns[:100] # random col generate
            #for c in col: #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            #    df[c] = 0
            
            #batch_x_ = df.to_numpy()

            # Random Sampling for the batch size...
            batch_z_real_dist = self.sample_Z(self.batch_size, self.z_dim)

            _, a_loss_curr = self.sess.run([autoencoder_optimizer, self.autoencoder_loss],
                        feed_dict = {self.x_input: batch_x, self.x_target: batch_x, 
                        #self.x_input_: batch_x_, self.batch_input_: X_,
                        self.batch_input: X_, # batch_b
                        self.labels: labels_,
                        self.keep_prob: self.keep_param}) 
  
                  
            # AJ: Count here, D2 is taking true data only.
            _, d2_loss_curr = self.sess.run([discriminator2_optimizer, self.dis2_loss],
                        feed_dict = {self.x_input: batch_x,
                        self.x_target: batch_x,
                        self.batch_input: X_,
                        self.real_distribution: batch_z_real_dist,
                        #self.x_input_: batch_x_, self.batch_input_: X_,
                        self.keep_prob: self.keep_param}) 
            
            _, db_loss_curr = self.sess.run([batchClassifier_optimizer, self.disb_loss],
                        feed_dict = {self.x_input: batch_x,
                                    self.batch_input: X_,
                        self.real_distribution: batch_z_real_dist,
                        self.batch_input: X_, # batch_b
                        #self.x_input_: batch_x_, self.batch_input_: X_,
                        self.keep_prob: self.keep_param})

            """
            _, classifier_curr = self.sess.run([classifier_optimizer, self.classifier_loss],
                                            
                        feed_dict = {
                        self.labels: labels_,       
                        self.keep_prob: self.keep_param})
            """

        print("Epoch : [%d] ,  a_loss = %.4f, d_loss: %.4f , db_loss: %.4f" 
              % (ep, a_loss_curr, d2_loss_curr, db_loss_curr))

              
        self._is_train = False # enables false after 1st iterations only...to make training process fast

        if (np.isnan(a_loss_curr) or np.isnan(d2_loss_curr) or np.isnan(db_loss_curr)): 
            a_loss_curr = 0
            d2_loss_curr = 0
            db_loss_curr = 0
            break

        #self.x_target: batch_x,
        a_loss_epoch.append(a_loss_curr) # total loss getting appended 
        d2_loss_epoch.append(d2_loss_curr)
        db_loss_epoch.append(db_loss_curr)
        
        if (ep == 200 or (ep == 250 or ep == 300)):
             self.eval_cluster_on_test(ep, self.name + "new")

    self.eval_cluster_on_test(ep, self.name + "new")

def encoder(self, x, reuse = False):
    """
    Encode part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :return: tensor which is the hidden latent variable of the autoencoder.
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


def discriminator2(self, z, z_dim, reuse=False):    
    """
    Discriminator that is used to match the posterior distribution with a given prior distribution.
    :param z: tensor of shape [batch_size, z_dim]
    :param reuse: True -> Reuse the discriminator variables,
                  False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    with tf.variable_scope('Discriminator2') as scope:
        if reuse:
            scope.reuse_variables()

        if self.is_bn:

            h = tf.layers.batch_normalization(
                lrelu(dense(z, z_dim, self.d_h_dim[self.num_layers - 1], name='dis2_h' + str(self.num_layers-1) + '_lin'),      
                      alpha=self.leak),
                training=self._is_train, name='dis2_bn' + str(self.num_layers-1))
            for i in range(self.num_layers - 2, -1, -1):
                h = tf.layers.batch_normalization(
                    lrelu(dense(h, self.d_h_dim[i + 1], self.d_h_dim[i], name='dis2_h' + str(i) + '_lin'),
                          alpha=self.leak),
                    training=self._is_train, name='dis2_bn' + str(i))

        else:

            h = tf.nn.dropout(
                lrelu(dense(z, z_dim, self.d_h_dim[self.num_layers - 1], name='dis2_h' + str(self.num_layers-1) + '_lin'),
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
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
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
    Discriminator takes the real data and try ti differentiate between different batches
    :param x: tensor of shape [batch_size, x_dim]
    :param batch: tensor of shape [batch_size] -> batchinfo of the train data
    :param reuse: True -> Reuse the discriminator variables,False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    '''
    x_ = pd.DataFrame(x.numpy())
    x_ = pd.concat([x_, batch], axis = 1)
    x = torch.tensor(x.values)
    x_dim = torch.tensor(721)
    '''
    
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

        # AJ 6 in pace of 1 , 3 for pancreas 
        output = dense(h, self.d_h_dim[0], self.N_batch, name='disb_output')
        return output


def classifier(self, z, reuse = True):
    
    # 10 -> 128 -> 256-> 16
    with tf.variable_scope('classifier_Celltype') as scope:
        if reuse:
            scope.reuse_variables()
            
    h = tf.nn.dropout(lrelu(dense(z, self.z_dim, 128, name = 'class_h'), alpha = self.leak),
                keep_prob = self.keep_prob)
            

    h = tf.nn.dropout(lrelu(dense(h, 128 ,256, name = "class_h"),alpha = self.leak), keep_prob = self.keep_prob)

    out = dense(h, 256, self.N_celltype, name = 'classifier_output')
                      
    return out      
            
def encoderU(self, z, reuse = False):

    with tf.variable_scope('EncoderU') as scope:
        if reuse:
            scope.reuse_variables()


    h = tf.nn.dropout(lrelu(dense(z, self.z_dim + self.N_celltype, self.g_h_dim[0], name='encu_h0_lin'), alpha=self.leak),
                      keep_prob=self.keep_prob)                

    for i in range(1, self.num_layers):
        print ("entering in for loop", i, " ",  self.g_h_dim[i - 1]," -- >", self.g_h_dim[i])
        h = tf.nn.dropout(lrelu(dense(h, self.g_h_dim[i - 1], self.g_h_dim[i], name='encu_h' + str(i) + '_lin'),
                  alpha=self.leak), keep_prob=self.keep_prob)                    

    u_post_m = dense(h, self.g_h_dim[self.num_layers - 1], self.u_dim, name='encu_u_post_m' +
                     str(self.num_layers) + '_lin')                
    u_post_v = tf.exp(dense(h, self.g_h_dim[self.num_layers - 1], self.u_dim, name='encu_u_post_v' + 
                            str(self.num_layers) + '_lin'))


    h = tf.nn.relu(dense(h, self.g_h_dim[self.num_layers - 1], self.u_dim, name='encu_h' + 
                         str(self.num_layers) + '_lin'))
                                        
                        
    return h, u_post_m, u_post_v


def decoderU(self, u, reuse = False):
    

    with tf.variable_scope('DecoderU') as scope:
        if reuse:
            scope.reuse_variables()
        
    h = tf.nn.dropout(lrelu(dense(u , self.u_dim + self.N_celltype, 8, name = 'decu_lin'),alpha = self.leak), keep_prob = self.keep_prob)
    #h = tf.nn.dropout(lrelu(dense(h, 8,10, name = 'decu_lin2'),alpha = self.leak), keep_prob = self.keep_prob)
           
    zconst_m = dense(h, 8, 10, name = 'decu_z_post_m')                
    zconst_v = tf.exp(dense(h, 8, 10, name = 'decu_z_post_v')) #self.z_dim
        
    return zconst_m, zconst_v

    
class scDREAMER(object):
    
    def __init__(self, sess, batch, cell_type, name, epoch = 200, lr=0.0007, beta1=0.9, batch_size=128, X_dim=2000, 
                 z_dim=10, dataset_name='Pancreas',
                 checkpoint_dir='checkpoint', sample_dir='samples', result_dir = 'result', num_layers = 2, 
                 g_h_dim = [512, 256, 0, 0],d_h_dim = [512, 256, 0, 0], gen_activation = 'sig', leak = 0.2, 
                 keep_param = 0.9, trans = 'sparse',is_bn = False,
                 g_iter = 2, lam=1.0, sampler = 'normal'):

        self.sess = sess
        self.epoch = epoch
        self.lr = lr
        self.beta1 = beta1
        self.batch_size = batch_size
        self.X_dim = X_dim
        self.z_dim = z_dim
        self.u_dim = 6
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.result_dir = result_dir
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
        self.cell_type = cell_type
        self.name = name
        self.batch = batch
        
        if self.trans == 'sparse':
            self.data_train, self.data_test, self.scale, self.labels_train, self.labels_test, self.batch_train, self.batch_test, self.batch_info, self.labels_enc = load_gene_mtx(self.dataset_name, transform=False, count=False, actv=self.gen_activation, batch = self.batch, cell_type = self.cell_type, name = self.name)
            self.N_batch = self.batch_train.shape[1]
            self.N_celltype = self.labels_enc.shape[1]
            
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

        print("Shape self.data_train:", self.data_train.shape) 
        print("Shape self.data_test:", self.data_test.shape) 
    
        self.build_model()

    build_model = build_model
    train_cluster = train_cluster
    encoder = encoder
    decoder = decoder
    encoderU = encoderU
    decoderU = decoderU
    #classifier = classifier
    
    batchClassifier = batchClassifier
    discriminator2 = discriminator2
    
    eval_cluster_on_test = eval_cluster_on_test
    # eval_cluster_on_test_ = eval_cluster_on_test_
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

    def next_batch(self, data, batch_info, labels, max_size):

        indx = np.random.randint(max_size - self.batch_size)
        return data[indx:(indx + self.batch_size), :], batch_info[indx:(indx + self.batch_size), :], labels[indx:(indx + self.batch_size), :]


    def sample_gaussian(self, mean, variance, scope=None):

        with tf.variable_scope(scope, 'sample_gaussian'):
            sample = tf.random_normal(tf.shape(mean), mean, tf.sqrt(variance))
            sample.set_shape(mean.get_shape())
            return sample



        
