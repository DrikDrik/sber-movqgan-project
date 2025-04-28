# sber-movqgan-project
This repository contains a university project related to direct latent space generation of the state-of-the-art image encoder developed by Sber: MoVQGAN.
Here I use simple generative architectures: DDPM, WGAN with aelf-attention and RealNVP. 


The repository contains pipelines both for training and inference allowing users to load my pre-trained checkpoint weights and fine-tune models optionally. 
All necessary requirements are listed in the 'requirements.txt' file


The 'pipeline' folder contains the code for importing the original MoVQGAN architecture including special algorithms for encoding datasets into the latent space and saving the resulting latent space data as .zip files. Also there is an option to load the raw aligned Celeba dataset before encoding into the latent space. 


The 'Models' folder contains each model I use along with all dependent modules and classes.


For more information about MoVQGAN, visit the original repository:  https://github.com/ai-forever/MoVQGAN


DDPM: https://drive.google.com/uc?export=download&id=1bFkk9Wd5Y-ndsbInitEoDVZRAE-4gKVc

GAN: https://drive.google.com/uc?export=download&id=12L6JPGABnWTmtrLk0-EoFTJ-jPbNchfw

RealNVP: https://drive.google.com/uc?export=download&id=1ZMV4CSGATFh3aydk-N26caHav9IfTkjc
