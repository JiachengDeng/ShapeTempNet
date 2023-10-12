CUDA_VISIBLE_DEVICES=8 python train_point_corr.py \
--dataset_name shrec \
--train_batch_size 4 \
--do_train true \
--optimizer adam \
--arch LuckPointCorr_const_mae \
--ckpt_period 5 \
--train_val_split 0.8 \
--val_batch_size 1 \
--test_batch_size 1 \
--layer_list ssss \
--d_feedforward 512 \
--steplr2 \
--test_on_shrec true \
# --learnedmask \
# --matrix_post_process ot \
# --resume_from_checkpoint "/ssd/djc/PointCorrespondence/DPC_mae/output/shape_corr/LuckPointCorr/arch_LuckPointCorr_fps_mae/dataset_name_smal/latent_dim_768/22_08:21:20:21/epoch=309.ckpt" \
# --ot_loss_lambda 0.2 \
# --compute_perm_loss \
# --perm_loss_lambda 0.001 \
# --optimizer adam_angle \
# --steplr2 \