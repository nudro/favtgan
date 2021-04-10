while getopts "f": OPTION

do

  python evaluation_bhatt_crops.py --real_dir crops_real_B/crop_eyes --fake_dir ${OPTARG}/crops_fake_B/crop_eyes --experiment ${OPTARG} --roi eyes

  python evaluation_psnr_ssim_crops.py --real_dir crops_real_B/crop_eyes --fake_dir ${OPTARG}/crops_fake_B/crop_eyes --experiment ${OPTARG} --roi eyes

done
