# SAMPLE IMAGES - IS RESEARCH DAY 2022
**Do not download, share, or distribute as these images are under review!**
Images will be taken down April 29, 2022 evening

### Samples from TFC-GAN 4P Variant: Top: Visible, Middle: Generated, Bottom: Real
<img src=./pics/tfc_gan_samples.png>

### Magnitude Spectra Samples
Using MSE to measure error between real thermal image and magnitude spectra

<img src=./pics/eur_mag.png>

<img src=./pics/dev_mag.png>

### Individual Patches of 4 Regions

<img src=./pics/TFCGAN_patches.png>

### Zoomed-in Patches Comparing TFC-GAN and runner up, pix2pix
Here we also show that LPIPS is the best measure for evaluating perceptual quality of generated thermal faces. Not SSIM or PSNR, as these metrics increase as image quality decreases.

<img src=./pics/zoom_in_patches.png>

### Ablation Study
We use the TFC-GAN 4P variant. From it, we ablate all stability parameters. We then ablate temperature loss. We finally ablate all three - stability, temperature, and contrastive patch loss.

<img src=./pics/ablation.png>
