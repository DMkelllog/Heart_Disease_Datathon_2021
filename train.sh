# pretrain

for aug_type in 1 3
    do
    for model in 'unet' 'caranet'
        do
        # Pretrain
        python main.py --model_type $model --data_type 'both' --augmentation_type 3 --learning_rate 1e-3
        # Fine-tune (A2C) or (A4C)
        ## pretrained_path: pre_both_{aug_type}_{model}_{lr}
        for d_type in 'A2C' 'A4C'
            do
            for lr in 1e-4 1e-5
                do
                for epoch in 4 9 'full'
                    do
                    python main.py --model_type $model --pretrained_path 'pre_both_3_unet_0.001' --data_type $d_type --augmentation_type $aug_type --learning_rate $lr  --pretrained_epoch $epoch --memo $epoch
                done
            done
        done
    done
done