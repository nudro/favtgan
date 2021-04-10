set -ex
python pix2pix-smooth.py --dataset_name EAI --n_epochs 2000 --batch_size 12 --gpu_num 1 --out_file EAI_V3 --sample_interval 200 --checkpoint_interval 10 --experiment EAI_V3