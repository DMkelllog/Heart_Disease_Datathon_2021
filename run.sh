## pretrain


# Pretrain
python main.py --model_type 'caranet' --data_type 'both' --augmentation_type 3 --learning_rate 1e-3 --memo 1e-3

# Fine-tune (A2C) or (A4C)
## pretrained_path: pre_both_{aug_type}_{model}

for lr in 1e-4 1e-5
    do
    for epoch in 4 9 full
        do
        python main.py --model_type 'caranet' --data_type '' --augmentation_type 3 --learning_rate $lr --memo $lr --pretrained_path 'pre_both_3_caranet' --pretrained_epoch $epoch
    done
done