CUDA_VISIBLE_DEVICES=1 python train_mix_v2.py \
--name guided_DDIM_v2/mix_commen_bc0.4_decay_F0.4_multiple_multi1.0_Faug_mix_v3_.7.3_0.4_bin1.0 \
--fake_data_name 'DDIM,guided' \
--real_data_name real \
--data_root ./data \
--flip \
--dataset_mode "normal" \
--batch_size 128 \
--lr 0.00008 \
# --pretrained_path ./checkpoints/your_pretrained/model_epoch_best.pth \

