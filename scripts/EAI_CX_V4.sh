set -ex
python favtgan_CX_V4.py --dataset_name EAI \
        --annots_csv /home/local/AD/cordun1/experiments/data/labels/EAI_s.csv \
        --n_epochs 501 \
        --batch_size 12 \
        --gpu_num 1 \
        --out_file EAI_CX \
        --sample_interval 100 \
        --checkpoint_interval 100 \
        --experiment EAI_CX \
        --n_classes 3
