CUDA_VISIBLE_DEVICES=2 python train_point_corr.py \
--dataset_name tosca \
--train_batch_size 4 \
--do_train false \
--optimizer adam \
--arch LuckPointCorr_fps_mae \
--ckpt_period 5 \
--train_val_split 0.8 \
--layer_list ssss \
--steplr2 \
--d_feedforward 512 \
--resume_from_checkpoint "/ssd/djc/PointCorrespondence/DPC_mae/output/shape_corr/LuckPointCorr/arch_LuckPointCorr_select_mae/dataset_name_tosca/latent_dim_768/23_09:11:05:41/epoch=499.ckpt" \
--test_on_tosca \
# --matrix_post_process offline_ot \
# --compute_perm_loss \
# --steplr2 \
#
# --optimizer adam_angle \
# --steplr2 \

# /ssd/djc/PointCorrespondence/DPC_mae/output/shape_corr/LuckPointCorr/arch_LuckPointCorr_fps_mae/dataset_name_shrec/latent_dim_768/31_07:19:04:29/epoch=224.ckpt