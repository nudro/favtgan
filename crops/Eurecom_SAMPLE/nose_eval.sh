while getopts "f": OPTION

do

  python evaluation_bhatt_crops.py --real_dir crops_real_B/crop_nose --fake_dir ${OPTARG}/crops_fake_B/crop_nose --experiment ${OPTARG} --roi nose

  python evaluation_psnr_ssim_crops.py --real_dir crops_real_B/crop_nose --fake_dir ${OPTARG}/crops_fake_B/crop_nose --experiment ${OPTARG} --roi nose

done
