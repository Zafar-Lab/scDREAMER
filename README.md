# scDREAMER
## Overview
**scDREAMER** is a single-cell data integration framework that employs a novel adversarial variational autoencoder for learning lower-dimensional cellular embeddings and a batch classifier neural network for the removal of batch effects. See our preprint below for more details. 

<img src='architecture.png'>

## Cite this article
Ajita Shree, Musale Krushna Pavan, Hamim Zafar. scDREAMER: atlas-level integration of single-cell datasets using deep generative model paired with adversarial classifier. bioRxiv 2022.07.12.499846; doi: https://doi.org/10.1101/2022.07.12.499846 

## Installation

scDREAMER can be installed and imported as follows <br />
```bash
git clone https://github.com/Zafar-Lab/scDREAMER.git
cd scDREAMER/scDREAMER

import model
```

Package versions for scDREAMER: <br />
```bash
numpy == 1.21.5 
pandas == 1.3.5 
scanpy == 1.9.1 
scikit_learn == 1.1.1 
scipy == 1.7.3 
tables == 3.7.0 
tensorflow == 1.15.0 
python == 3.7 
```

## Tutorials
Check out the following Colab notebook to get a flavor of a typical workflow for data integration using scDREAMER. <br /> <br />
Tutorial Link: (https://colab.research.google.com/drive/1UQ3pLd9UgSgsR8TqRFVWuYBy-doZpy2K?usp=sharing) 

