CUDA_VISIBLE_DEVICES=0 python train_point_corr.py \
--dataset_name tosca \
--train_batch_size 4 \
--do_train true \
--optimizer adam \
--arch TemplatePointCorr \
--ckpt_period 25 \
--train_val_split 0.8 \
--val_batch_size 1 \
--test_batch_size 1 \
--layer_list ssss \
--d_feedforward 512 \
--steplr2 \
--test_on_tosca true \
--max_epochs 600 \
--template_neigh_lambda 0.0 \
--template_cross_lambda 0.0 \
# --p_aug \
# --simi_metric embed \
# --template_div_lambda 0.0 \
# --simi_metric pos \
# --learnedmask \
# --matrix_post_process ot \
# --resume_from_checkpoint "/ssd/djc/PointCorrespondence/DPC_mae/output/shape_corr/LuckPointCorr/arch_LuckPointCorr_fps_mae/dataset_name_smal/latent_dim_768/22_08:21:20:21/epoch=309.ckpt" \
# --ot_loss_lambda 0.2 \
# --compute_perm_loss \
# --perm_loss_lambda 0.001 \
# --optimizer adam_angle \
# --steplr2 \
