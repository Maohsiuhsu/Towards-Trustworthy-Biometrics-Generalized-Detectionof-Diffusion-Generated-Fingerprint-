CUDA_VISIBLE_DEVICES=1 python visualize_features.py \
--model_path ./checkpoints/your_experiment/model_epoch_best.pth \
--batch_size 128 \
--eval \
--data_root ./data \
--name final \
--real_data_name real \

