## pretrain


# Pretrain
# python main.py --model_type 'caranet' --data_type 'both' --augmentation_type 3 --learning_rate 1e-3 --num_epochs=1

# Fine-tune (A2C) or (A4C)
## pretrained_path: pre_both_{aug_type}_{model}_{lr}
for d_type in 'A2C' 'A4C'
    do
    for lr in 1e-4 1e-5
        do
        for epoch in 4 9 'full'
            do
            python main.py --model_type 'caranet' --pretrained_path 'pre_both_3_caranet_0.001' --data_type $d_type --augmentation_type 3 --learning_rate $lr  --pretrained_epoch $epoch --memo $epoch --num_epochs=1
        done
    done
done