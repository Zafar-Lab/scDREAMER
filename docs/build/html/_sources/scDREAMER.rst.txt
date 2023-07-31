scDREAMER
=========

.. function:: (sess, batch, cell_type, name, epoch = 300, lr=0.0007, beta1=0.9, batch_size=128, X_dim=2000, z_dim=10, dataset_name='Pancreas',checkpoint_dir='checkpoint', sample_dir='samples', result_dir = 'result', num_layers = 1, g_h_dim = [512, 256, 0, 0], d_h_dim = [512, 256, 0, 0], gen_activation = 'sig', leak = 0.2, keep_param = 0.9, trans = 'sparse',is_bn = False, g_iter = 2, lam=1.0, sampler = 'normal'):

   This function takes two numbers as input and returns their sum.

   :param batch: obs key for batch information
   :type batch: string

   :param cell_type: obs key for cell type information. Only used for plotting
   :type cell_type: int

   :param name: name
   :type name:  

   :return: The sum of the two numbers.
   :rtype: int
