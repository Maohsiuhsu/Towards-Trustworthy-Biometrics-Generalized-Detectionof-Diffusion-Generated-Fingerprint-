# Run evaluation script without cropping. Model weights need to be downloaded.
CUDA_VISIBLE_DEVICES=0 python eval.py \
--model_path ./checkpoints/your_experiment/model_epoch_best_last.pth \
--batch_size 128 \
--eval \
--data_root ./data \
--name eval/default_run \
--real_data_name real \
--dataset_mode "normal"

