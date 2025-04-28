# sber-movqgan-project
This repository contains a unversity project related to direct latent space generation of the SotA image encoder from Sber: MoVQGAN.
Here i use simple generative architectures: DDPM, WGAN with Self Attention and RealNVP. 


The repository contains piplines both for training and inference with an ability to load my chekpoint weights and fine-tune models if needed. 
All needed requirements located in requirements.txt


'pipeline' folder contains importing original MoVQGAN architetcture including special algorithms to shift datasets with its' encoder into latent space and to save .zip files containing it. Also there is a possibility to load the raw aligned Celeba dataset before shifting into latent space. 


The 'Models' folder containts each model i use with every dependent module and class.


For more information about MoVQGAN: https://github.com/ai-forever/MoVQGAN (original repository)


Weights for used models:

DDPM: https://drive.google.com/uc?export=download&id=1bFkk9Wd5Y-ndsbInitEoDVZRAE-4gKVc

GAN: https://drive.google.com/uc?export=download&id=12L6JPGABnWTmtrLk0-EoFTJ-jPbNchfw

RealNVP: https://drive.google.com/uc?export=download&id=1ZMV4CSGATFh3aydk-N26caHav9IfTkjc
