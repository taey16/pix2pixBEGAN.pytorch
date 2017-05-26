# pix2pix + BEGAN
- [Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)
- [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717)

# Install
- install [pytorch](https://github.com/pytorch/pytorch) and [pytorch.vision](https://github.com/pytorch/vision)

# Dataset
- [Download images from author's implementation](https://github.com/phillipi/pix2pix)
- Suppose you downloaded "facades" dataset in ```/path/to/facades```

# Train
- **pix2pixGAN**
- ```CUDA_VISIBLE_DEVICES=x python main_pix2pixgan.py --dataroot /path/to/facades/train --valDataroot /path/to/facades/val --exp /path/to/a/directory/for/checkpoints```
- **pix2pixBEGAN**
- ```CUDA_VISIBLE_DEVICES=x python main_pix2pixBEGAN.py --dataroot /path/to/facades/train --valDataroot /path/to/facades/val --exp /path/to/a/directory/for/checkpoints```
- [Most of the parameters are the same](https://github.com/taey16/pix2pixBEGAN.pytorch/blob/master/main_pix2pixBEGAN.py#L37-L48) for a fair comparision.
- The original pix2pix is modelled as a conditional GAN, however we didn't. Input samples are not given in D([Only target samples are given](https://github.com/taey16/pix2pixBEGAN.pytorch/blob/master/main_pix2pixBEGAN.py#L175))
- We used the [image-buffer](https://github.com/taey16/pix2pixBEGAN.pytorch/blob/master/main_pix2pixBEGAN.py#L178)(analogyous to replay-buffer in DQN) in training D.
- Try other datasets as your need. Similar results will be found.

# Training Curve(pix2pixBEGAN)
- **L_D and L_G**

![loss](https://github.com/taey16/pix2pixBEGAN.pytorch/blob/master/imgs/BEGAN_loss_niter500.png)

- We found out both L_D and L_G are balanced consistancely and converged quickly, even thought network D and G are different in terms of model capacity and detailed layer specification.

- **M_global**

![Mglobal](https://github.com/taey16/pix2pixBEGAN.pytorch/blob/master/imgs/BEGAN_Mglobal_niter500.png)

- As the author said, M_global is a good indicator for monitoring convergence.
- [Parsing log](http://htmlpreview.github.io/?https://github.com/taey16/pix2pixBEGAN.pytorch/blob/master/imgs/pix2pixBEGAN.html)

# Comaprison
- **pix2pixGAN vs. pix2pixBEGAN**
- ```CUDA_VISIBLE_DEVICES=x python compare.py --netG_GAN /path/to/netG.pth --netG_BEGAN /path/to/netG.pth --exp /path/to/a/dir/for/saving --tstDataroot /path/to/facades/test/```
![failure](https://github.com/taey16/pix2pixBEGAN.pytorch/blob/master/imgs/failure.png)
![GANvsBEGAN](https://github.com/taey16/pix2pixBEGAN.pytorch/blob/master/imgs/pix2pixGAN_vs_pix2pixBEGAN.png)
- [Checkout more results](https://github.com/taey16/pix2pixBEGAN.pytorch/blob/master/imgs/comparison.png)(order in input, real-target, fake(pix2pixBEGAN), fake(pix2pixGAN))

# Showing reconstruction from D and generation from G
- (order in input, real-target, reconstructed-real, fake, reconstructed-fake) 
![reconDandGenG](https://github.com/taey16/pix2pixBEGAN.pytorch/blob/master/imgs/generated_epoch_00000499_iter00200000.png)

# Reference
- [pix2pix.pytorch](https://github.com/taey16/pix2pix.pytorch)
- [BEGAN in pytorch](https://github.com/sunshineatnoon/Paper-Implementations/tree/master/BEGAN)
- fantastic [pytorch](http://pytorch.org/docs/)
