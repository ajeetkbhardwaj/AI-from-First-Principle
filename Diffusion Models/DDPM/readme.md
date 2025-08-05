

# DDPM

1. Forward Step
We take an resource/image say $x_0$ then we sequentially add gaussian noise to it after large number of steps resource/image become a noisy sample from gaussian distribution

![alt txt](resource/image.png)

2. Backward Step 
We take an noisy sample from the distribution and try to sequentially again reduce the noisy from sample to construct the original resource/image, in this process of reduction of noisy we learn how images are created. Thus we want model to predict the mean and variance matrix for our data

3. Loss function optimization 

maximize the log likelihood i.e minimize the KL-divergence between groud truth and noisy distribution conditioned on $x_0$.

![alt text](resource/image-1.png)

The minimizing the kl-divergence is the minimizing noise predicted and sampled noise difference.

![alt text](resource/image-2.png)

4. Generating Images

![alt text](resource/image-3.png)

5. Implementation
Computation for forward and reverse process,
Noise Scheduler do above things firstely as

![alt text](resource/image-4.png)


Note : In orginal paper of ddpm author used linear noise scheduler where $\beta is 0.0001$ i.e approx 1000 time steps.

![alt text](resource/image-5.png)

Note : We start the sampling first from the noisy distribution
![alt text](resource/image-6.png)

![alt text](resource/image-7.png)

![alt text](resource/image-8.png)

![alt text](resource/image-9.png)

Note : Implementing the Diffusion Model
1. shape of input and output same and some mechanism to fusion the time step information
2. Information at what time step should always be available to us whether we are at training or sampling ok. help us to predict the noisy at any time step so, at any time step how much noisy the resource/image contains.


For Diffusion Block : 
activation, normaization etc we can use the UNet Architecture used by HuggingFace in diffusors pipline.

![alt text](resource/image-10.png)

How time step information is encoded ?

![alt text](resource/image-11.png)

UNet : Encoder and Decoder Architecture
1. Encoder : is a series of downsampling blocks and each block reduces the size of input and increases the number of channels
2. Mid Block : they work at same spatial resolution
3. Decoder : as series of upsampling block, it increages the input size and reduces the number of channels to match the size.
4. Residual skip Connections : 

![alt text](resource/image-12.png)

Downblock  = REsnet + Self Attension
![alt text](resource/image-13.png)
![alt text](resource/image-14.png)
![alt text](resource/image-15.png)
![alt text](resource/image-16.png)
![alt text](resource/image-17.png)

Implementation
![alt text](resource/image-18.png)
![alt text](resource/image-19.png)
![alt text](resource/image-20.png)
![alt text](resource/image-21.png)
![alt text](resource/image-22.png)

Mid Block
![alt text](resource/image-23.png)

implementation 

![alt text](resource/image-24.png)
![alt text](resource/image-25.png)
![alt text](resource/image-26.png)

![alt text](resource/image-27.png)

Upblock
![alt text](resource/image-28.png)

![alt text](resource/image-29.png)
![alt text](resource/image-30.png)

![alt text](resource/image-31.png)