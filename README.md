# scDREAMER
## Overview
**scDREAMER** is a single-cell data integration framework that employs a novel adversarial variational autoencoder for learning lower-dimensional cellular embeddings and a batch classifier neural network for the removal of batch effects. See our preprint below for more details. 

<img src='architecture.png'>

## Cite this article
Ajita Shree, Musale Krushna Pavan, Hamim Zafar. scDREAMER: atlas-level integration of single-cell datasets using deep generative model paired with adversarial classifier. bioRxiv 2022.07.12.499846; doi: https://doi.org/10.1101/2022.07.12.499846 

## Installation

numpy == 1.21.5 <br />
pandas == 1.3.5 <br />
scanpy == 1.9.1 <br />
scikit_learn == 1.1.1 <br />
scipy == 1.7.3 <br />
tables == 3.7.0 <br />
tensorflow == 1.15.0 <br />
python == 3.7 <br />

## Tutorials
Check out the following Colab notebook to get a flavor of a typical workflow for data integration using scDREAMER. 
https://drive.google.com/file/d/1kLGBv9osnMMG6zbg9_45UfvGeobji8t-/view?usp=sharing
