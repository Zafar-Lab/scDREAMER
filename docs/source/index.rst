.. scDREAMER documentation master file, created by
   sphinx-quickstart on Mon Jul 31 10:38:29 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========

Welcome to scDREAMER's documentation!
-------------------------------------

**scDREAMER** is a single-cell data integration framework that employs a novel adversarial variational autoencoder for learning lower-dimensional cellular embeddings and a batch classifier neural network for the removal of batch effects. See our preprint `bioRxiv`_ for more details. 

.. image:: ../../architecture.png


What Computational tasks can scDREAMER be used for?
---------------------------------------------------

*scDREAMER* suite can be used for:

#. scDREAMER for an unsupervised integration of multiple batches
#. scDREAMER-SUP for a supervised integration across multiple batches
#. scDREAMER-SUP can also be when cell type annotations are missing in the datasets i.e., 10%, 20%, 50%
#. Atlas level and cross-species integration
#. Large datasets with ~1 million cells


Cite this article
-----------------
Ajita Shree\*, Musale Krushna Pavan\*, Hamim Zafar. scDREAMER: atlas-level integration of single-cell datasets using deep generative model paired with adversarial classifier. bioRxiv 2022.07.12.499846; doi: https://doi.org/10.1101/2022.07.12.499846  
\* equally contributed

.. _bioRxiv: https://www.biorxiv.org/content/10.1101/2022.07.12.499846v1

.. toctree::
   :maxdepth: 2

   self
   installation
   scDREAMER_Sup_runs
   scDREAMER
   scDREAMER_SUP
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
