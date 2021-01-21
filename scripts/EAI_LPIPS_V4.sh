set -ex
python favtgan_LPIPS_V4.py --dataset_name EAI \
        --annots_csv /home/local/AD/cordun1/experiments/data/labels/EAI_s.csv \
        --n_epochs 501 \
        --batch_size 12 \
        --gpu_num 0 \
        --out_file EAI_LPIPS_V4 \
        --sample_interval 100 \
        --checkpoint_interval 100 \
        --experiment EAI_LPIPS_V4 \
        --n_classes 3
