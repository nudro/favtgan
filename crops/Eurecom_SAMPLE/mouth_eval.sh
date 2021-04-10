while getopts "f": OPTION

do

  python evaluation_bhatt_crops.py --real_dir crops_real_B/crop_mouth --fake_dir ${OPTARG}/crops_fake_B/crop_mouth --experiment ${OPTARG} --roi mouth

  python evaluation_psnr_ssim_crops.py --real_dir crops_real_B/crop_mouth --fake_dir ${OPTARG}/crops_fake_B/crop_mouth --experiment ${OPTARG} --roi mouth

done
