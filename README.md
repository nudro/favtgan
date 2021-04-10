# anon

This repo includes:

<p>

GAN implementation in PyTorch to include the code for:
- Under `favtGAN/favtGAN`, the favtGAN baseline, no noise, noisy labels, and Gaussian designs, in addition to the data loaders, and sample bash scripts to call parameters. As an example to run training open terminal and enter the prompt `bash train_EI_sensor_baseline.sh`
- Under `favtGAN/pix2pix`, the implementation by Erik Lindernoren at https://github.com/eriklindernoren/PyTorch-GAN#pix2pix, with the only modification of one-sided smooth labels for the valid tensor.

Preprocessing for Crops:
- We provide a sample of Eurecom images in `crops/Eurecom_SAMPLE`. There are whole face Eurecom images of 5 generated thermal images from the "EIO_sensor_V4" experiment in `crops/EIO_sensor_fake_B`.
- Real ground-truth thermal cropped examples for the eye, mouth, and nose are provided in `crops/crops_real_B_SAMPLE` for the same 5 images.
- In `crops/EIO_sensor_V4` are the crops of the eye, mouth, and nose taken from the generated faces. the script used to take the crops is shown in the notebook `Eurecom Official Crops.ipynb`.
- Running the cells in the notebook, will call the Bhattacharrya, PSNR, and SSIM python scripts.
- This outputs csv's under `crops/EIO_sensor_V4` with all image quality scores for the eyes, mouth, and nose.
- The same procedure would be applied to Iris images using `Iris Official Crops.ipynb`.

Person Re-Identification:
- All person re-identification tasks were performed using the torchreid library at https://github.com/KaiyangZhou/deep-person-reid.
- An example of a script for the experiment `eur_EI_GAND_data.py` which was the dataloader for the Eurecom + Iris (EI) favtGAN baseline experiment is provided under `person_reid`.
- An example of how the MuDeep algorithm was called and run under evaluation model is provided in `torchreid.ipynb`.

Data Preprocessing:
- Jupyter Notebooks for how the Flir ADAS data was processed as well as the Iris dataset, is provided in `data_proc`
- The majority of functions focus on resizing images, and dealing w.r.t Iris, the removal of artificial framing and text, in addition to scaling of the visible faces.

