# favtGAN - Facial Visible Translation GAN

`pip install requirements.txt`

Python Scripts to train favtGAN using four different architectures: 
- Baseline
- No Noise
- Noisy Labels
- Gaussian 

Scripts located under `favtGAN/favtGAN`

Pix2pix Scripts for comparison from the https://github.com/eriklindernoren/PyTorch-GAN#pix2pix repository also provided. 

Quantitative Evaluation of SSIM, PSNR, in addition to Bhattacharrya and FID scores provided under `quant_eval`. For both Eurecom and Iris, scripts are provided to evaluate each dataset. 

## Datasets: 

- Eurecom dataset can only be acquired by permission from authors Mallat et al. <i>"Mallat, Khawla, and Jean-Luc Dugelay. "A benchmark database of visible and thermal paired face images across multiple variations." 2018 International Conference of the Biometrics Special Interest Group (BIOSIG). IEEE, 2018."</i>
- Iris dataset can be downloaded here: http://vcipl-okstate.org/pbvs/bench/. For preprocessing scripts, please contact me at cordun1@umbc.edu and I will provide you a preprocessing notebook with functions to format and align for best results.
- OSU dataset can also be downloaded here: http://vcipl-okstate.org/pbvs/bench/
- ADAS dataset must be acquired by FLIR https://www.flir.com/oem/adas/adas-dataset-form/.
- Labels are provided under `labels` where the Dataloader accepts the .csv file.



