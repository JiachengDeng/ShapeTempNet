from visualization.visualize_api import visualize_pair_corr, visualize_reconstructions
from data.point_cloud_db.point_cloud_dataset import PointCloudDataset

from models.sub_models.dgcnn.dgcnn_modular import DGCNN_MODULAR
from models.sub_models.dgcnn.dgcnn import get_graph_feature

from models.sub_models.cross_attention.transformers import FlexibleTransformerEncoder, LuckTransformerEncoder,Decoder,Encoder
from models.sub_models.cross_attention.transformers import TransformerSelfLayer, TransformerCrossLayer, LuckSelfLayer
from models.sub_models.cross_attention.position_embedding import PositionEmbeddingCoordsSine, \
    PositionEmbeddingLearned
from models.sub_models.cross_attention.warmup import WarmUpScheduler
from models.sub_models.pointnet import PointcloudTokenizer
from models.sub_models.feature_upsampling import PointNetFeatureUpsampling
#from models.sub_models.ssw_loss.ssw_loss import StereoWhiteningLoss, ShapeWhiteningLoss

import numpy as np

import math

from torch.optim.lr_scheduler import MultiStepLR, StepLR

from torch_cluster import knn
import pointnet2_ops._ext as _ext
from models.correspondence_utils import get_s_t_neighbors
from models.shape_corr_trainer import ShapeCorrTemplate
from models.metrics.metrics import AccuracyAssumeEye, AccuracyAssumeEyeSoft, uniqueness

from utils import switch_functions

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sub_models.dgcnn.dgcnn import DGCNN as non_geo_DGCNN

from utils.argparse_init import str2bool

from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
 


class GroupingOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx):
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        ctx.for_backwards = (idx, N)
        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, N = ctx.for_backwards
        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None

grouping_operation = GroupingOperation.apply

class LuckPointCorr(ShapeCorrTemplate):
    
    def __init__(self, hparams, **kwargs):
        """Stub."""
        super(LuckPointCorr, self).__init__(hparams, **kwargs)
        
        self.encoder_norm = nn.LayerNorm(self.hparams.d_embed) if self.hparams.pre_norm else None
        hparams.in_features_dim = self.hparams.d_embed
        self.encoder_CROSS = LuckTransformerEncoder(hparams, self.hparams.layer_list, self.encoder_norm, True)
        self.tokenizer = PointcloudTokenizer(
            num_groups=128,
            group_size=16,
            group_radius=None,
            token_dim=self.hparams.d_embed,
        )
        self.upsampling = PointNetFeatureUpsampling(in_channel=self.hparams.d_embed+3, mlp=[self.hparams.d_embed, self.hparams.d_embed])
        self.pos_embed = PositionEmbeddingCoordsSine(3, self.hparams.d_embed, scale= 1.0)
        #self.encoder = Encoder(4, self.hparams.d_embed, 8, True)
        self.decoder = Decoder(4, self.hparams.d_embed, 8)
        ###
        #self.decoder_CROSS = LuckTransformerEncoder(hparams, self.hparams.layer_list, self.encoder_norm, True)
        ###Shape Selective whitening
        # if self.hparams.old_ssw:
        #     self.loss_stereo_ssw = StereoWhiteningLoss()
        # else:
        #     self.loss_stereo_ssw = ShapeWhiteningLoss()
        ###

        self.chamfer_dist_3d = dist_chamfer_3D.chamfer_3DDist()

        self.accuracy_assume_eye = AccuracyAssumeEye()
        self.accuracy_assume_eye_soft_0p01 = AccuracyAssumeEyeSoft(top_k=int(0.01 * self.hparams.num_points))
        self.accuracy_assume_eye_soft_0p05 = AccuracyAssumeEyeSoft(top_k=int(0.05 * self.hparams.num_points))


    def chamfer_loss(self, pos1, pos2):
        if not pos1.is_cuda:
            pos1 = pos1.cuda()

        if not pos2.is_cuda:
            pos2 = pos2.cuda()

        dist1, dist2, idx1, idx2 = self.chamfer_dist_3d(pos1, pos2)
        loss = torch.mean(dist1) + torch.mean(dist2)

        return loss

    def configure_optimizers(self):
        if self.hparams.warmup:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0, weight_decay=0.0001)
            self.scheduler = WarmUpScheduler(self.optimizer, [28800, 28800, 0.5], 0.0001)
        elif self.hparams.steplr:
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.0001)
            self.scheduler = StepLR(optimizer=self.optimizer, step_size=200, gamma=0.5)
        elif self.hparams.steplr2:
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.slr, weight_decay=self.hparams.swd)
            self.scheduler = StepLR(optimizer=self.optimizer, step_size=65, gamma=0.7)
        elif self.hparams.testlr:
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.slr, weight_decay=self.hparams.swd)
            self.scheduler = StepLR(optimizer=self.optimizer, step_size=30, gamma=0.8)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            self.scheduler = MultiStepLR(self.optimizer, milestones=[6, 9], gamma=0.1)
        return [self.optimizer], [self.scheduler]

    # def compute_deep_features(self, shape):
    #     if self.hparams.compute_ssw_loss:
    #         shape["dense_output_features"], shape["intermediate_features"] = self.encoder_DGCNN.forward_per_point(shape["pos"], start_neighs=shape["neigh_idxs"])
    #     else:
    #         shape["dense_output_features"] = self.encoder_DGCNN.forward_per_point(shape["pos"], start_neighs=shape["neigh_idxs"])
    #     return shape
    
    def compute_self_features(self, source, target): 
        tokens_src, centers_src = self.tokenizer(source["pos"])
        neigh_src = torch.stack([
            knn(centers_src[i], centers_src[i], 16)[1].reshape(centers_src.shape[1], -1)
            for i in range(centers_src.shape[0])
        ])
        centers_srcpos = self.pos_embed(centers_src)
        src_pos = self.pos_embed(source["pos"])
        tokens_tgt, centers_tgt = self.tokenizer(target["pos"])
        centers_tgtpos = self.pos_embed(centers_tgt)
        tgt_pos = self.pos_embed(target["pos"])
        neigh_tgt = torch.stack([
            knn(centers_tgt[i], centers_tgt[i], 16)[1].reshape(centers_tgt.shape[1], -1)
            for i in range(centers_tgt.shape[0])
        ])
        #source_pe=self.pos_embed(source["pos"].reshape(-1,3)).reshape(-1,1024,self.hparams.d_embed)
        #target_pe=self.pos_embed(target["pos"].reshape(-1,3)).reshape(-1,1024,self.hparams.d_embed)
        # src_out = self.encoder(tokens_src.transpose(0,1),centers_srcpos).permute(1,2,0)
        # tgt_out = self.encoder(tokens_tgt.transpose(0,1),centers_tgtpos).permute(1,2,0)
        src_out, tgt_out = self.encoder_CROSS(
            tokens_src.transpose(0,1),  tokens_tgt.transpose(0,1),
            src_xyz = centers_src,
            tgt_xyz = centers_tgt,
            src_neigh = neigh_src,
            tgt_neigh = neigh_tgt,
        )
        
        
        src_out1 = self.upsampling(source["pos"], centers_src, source["pos"], src_out.transpose(1,2))  # (B, N, C)
        src_out = self.decoder(src_out,centers_srcpos,src_out1.transpose(1,2),src_pos)
        #src_out = self.encoder(src_out1.transpose(0,1),src_pos)
        tgt_out1 = self.upsampling(target["pos"], centers_tgt, target["pos"], tgt_out.transpose(1,2))
        #tgt_out = self.encoder(tgt_out1.transpose(0,1),tgt_pos)
        # src_out, tgt_out = self.decoder_CROSS(
        #     src_out1.transpose(0,1),  tgt_out1.transpose(0,1),
        #     src_xyz = source["pos"],
        #     tgt_xyz = target["pos"],
        #     src_neigh = source["neigh_idxs"],
        #     tgt_neigh = target["neigh_idxs"],
        # )
        
        tgt_out = self.decoder(tgt_out,centers_tgtpos,tgt_out1.transpose(1,2),tgt_pos)
        source["dense_output_features"] = src_out.transpose(1,2)
        target["dense_output_features"] = tgt_out.transpose(1,2)
        
        return source, target

    def forward_source_target(self, source, target):
        
        ###transformers     
        source, target = self.compute_self_features(source, target)
        ###

        # measure cross similarity
        P_non_normalized = switch_functions.measure_similarity(self.hparams.similarity_init, source["dense_output_features"], target["dense_output_features"])
        
        temperature = None
        P_normalized = P_non_normalized

        # cross nearest neighbors and weights
        source["cross_nn_weight"], source["cross_nn_sim"], source["cross_nn_idx"], target["cross_nn_weight"], target["cross_nn_sim"], target["cross_nn_idx"] =\
            get_s_t_neighbors(self.hparams.k_for_cross_recon, P_normalized, sim_normalization=self.hparams.sim_normalization)

        # cross reconstruction
        source["cross_recon"], source["cross_recon_hard"] = self.reconstruction(source["pos"], target["cross_nn_idx"], target["cross_nn_weight"], self.hparams.k_for_cross_recon)
        target["cross_recon"], target["cross_recon_hard"] = self.reconstruction(target["pos"], source["cross_nn_idx"], source["cross_nn_weight"], self.hparams.k_for_cross_recon)

        return source, target, P_normalized, temperature

    @staticmethod
    def reconstruction(pos, nn_idx, nn_weight, k):
        nn_pos = get_graph_feature(pos.transpose(1, 2), k=k, idx=nn_idx, only_intrinsic='neighs', permute_feature=False)
        nn_weighted = nn_pos * nn_weight.unsqueeze(dim=3)
        recon = torch.sum(nn_weighted, dim=2)

        recon_hard = nn_pos[:, :, 0, :]
 
        return recon, recon_hard

    def forward_shape(self, shape):
        P_self = switch_functions.measure_similarity(self.hparams.similarity_init, shape["dense_output_features"], shape["dense_output_features"])

        # measure self similarity
        nn_idx = shape['neigh_idxs'][:,:,:self.hparams.k_for_self_recon + 1] if self.hparams.use_euclidiean_in_self_recon else None
        shape["self_nn_weight"], _, shape["self_nn_idx"], _, _, _ = \
            get_s_t_neighbors(self.hparams.k_for_self_recon + 1, P_self, sim_normalization=self.hparams.sim_normalization, s_only=True, ignore_first=True,nn_idx=nn_idx)

        # self reconstruction
        shape["self_recon"], _ = self.reconstruction(shape["pos"], shape["self_nn_idx"], shape["self_nn_weight"], self.hparams.k_for_self_recon)

        return shape, P_self

    @staticmethod
    def batch_frobenius_norm_squared(matrix1, matrix2):
        loss_F = torch.sum(torch.pow(matrix1 - matrix2, 2), dim=[1, 2])

        return loss_F

    def get_perm_loss(self, P, do_soft_max=True):
        batch_size = P.shape[0]

        I = torch.eye(n=P.shape[1], device=P.device)
        I = I .unsqueeze(0).repeat(batch_size, 1, 1)

        if do_soft_max:
            P_normed = F.softmax(P, dim=2)
        else:
            P_normed = P

        perm_loss = torch.mean(self.batch_frobenius_norm_squared(torch.bmm(P_normed, P_normed.transpose(2, 1).contiguous()), I.float()))

        return perm_loss

    @staticmethod
    def get_neighbor_loss(source, source_neigh_idxs, target_cross_recon, k):
        # source.shape[1] is the number of points

        if k < source_neigh_idxs.shape[2]:
            neigh_index_for_loss = source_neigh_idxs[:, :, :k]
        else:
            neigh_index_for_loss = source_neigh_idxs

        source_grouped = grouping_operation(source.transpose(1, 2).contiguous(), neigh_index_for_loss.int()).permute(0, 2, 3, 1)
        source_diff = source_grouped[:, :, 1:, :] - torch.unsqueeze(source, 2)  # remove fist grouped element, as it is the seed point itself
        source_square = torch.sum(source_diff ** 2, dim=-1)

        target_cr_grouped = grouping_operation(target_cross_recon.transpose(1, 2).contiguous(), neigh_index_for_loss.int()).permute(0, 2, 3, 1)
        target_cr_diff = target_cr_grouped[:, :, 1:, :] - torch.unsqueeze(target_cross_recon, 2)  # remove fist grouped element, as it is the seed point itself
        target_cr_square = torch.sum(target_cr_diff ** 2, dim=-1)

        GAUSSIAN_HEAT_KERNEL_T = 8.0
        gaussian_heat_kernel = torch.exp(-source_square/GAUSSIAN_HEAT_KERNEL_T)
        neighbor_loss_per_neigh = torch.mul(gaussian_heat_kernel, target_cr_square)

        neighbor_loss = torch.mean(neighbor_loss_per_neigh)

        return neighbor_loss

    def forward(self, data):
        
        for shape in ["source", "target"]:
            data[shape]["edge_index"] = [
                knn(data[shape]["pos"][i], data[shape]["pos"][i], self.hparams.num_neighs,)
                for i in range(data[shape]["pos"].shape[0])
            ]
            data[shape]["neigh_idxs"] = torch.stack(
                [data[shape]["edge_index"][i][1].reshape(data[shape]["pos"].shape[1], -1) for i in range(data[shape]["pos"].shape[0])]
            )

        # dense features, similarity, and cross reconstruction

        data["source"], data["target"], data["P_normalized"], data["temperature"] = self.forward_source_target(data["source"], data["target"])
        
        # ### For visualizations
        # if self.hparams.mode == "val":
        #     p_cpu = data["P_normalized"].data.cpu().numpy()
        #     source_xyz = data["source"]["pos"].data.cpu().numpy()
        #     target_xyz = data["target"]["pos"].data.cpu().numpy()
        #     # label_cpu = label.data.cpu().numpy()
        #     np.save("./smal-val/p_{}".format(self.hparams.batch_idx), p_cpu)
        #     np.save("./smal-val/source_{}".format(self.hparams.batch_idx), source_xyz)
        #     np.save("./smal-val/target_{}".format(self.hparams.batch_idx), target_xyz)
        #     # np.save("./smal-test/label_{}".format(batch_idx), label_cpu)
        # ###

        
        # cross reconstruction losses
        self.losses[f"source_cross_recon_loss"] = self.hparams.cross_recon_lambda * self.chamfer_loss(data["source"]["pos"], data["source"]["cross_recon"])
        self.losses[f"target_cross_recon_loss"] =self.hparams.cross_recon_lambda * self.chamfer_loss(data["target"]["pos"], data["target"]["cross_recon"])

        # self reconstruction
        if self.hparams.use_self_recon:
            _, P_self_source = self.forward_shape(data["source"])
            _, P_self_target = self.forward_shape(data["target"])
            
            # self reconstruction losses
            data["source"]["self_recon_loss_unscaled"] = self.chamfer_loss(data["source"]["pos"], data["source"]["self_recon"])
            data["target"]["self_recon_loss_unscaled"] = self.chamfer_loss(data["target"]["pos"], data["target"]["self_recon"])

            self.losses[f"source_self_recon_loss"] = self.hparams.self_recon_lambda * data["source"]["self_recon_loss_unscaled"]
            self.losses[f"target_self_recon_loss"] = self.hparams.self_recon_lambda * data["target"]["self_recon_loss_unscaled"]

        if self.hparams.compute_perm_loss:
            data[f"perm_loss_fwd_unscaled"] = self.get_perm_loss(data["P_normalized"])
            data[f"perm_loss_bac_unscaled"] = self.get_perm_loss(data["P_normalized"].transpose(2, 1).contiguous())
            
            if self.hparams.perm_loss_lambda > 0.0:
                self.losses[f"perm_loss_fwd"] = self.hparams.perm_loss_lambda * data[f"perm_loss_fwd_unscaled"]
                self.losses[f"perm_loss_bac"] = self.hparams.perm_loss_lambda * data[f"perm_loss_bac_unscaled"]


        
        if self.hparams.compute_neigh_loss and self.hparams.neigh_loss_lambda > 0.0:
            data[f"neigh_loss_fwd_unscaled"] = \
                self.get_neighbor_loss(data["source"]["pos"], data["source"]["neigh_idxs"], data["target"]["cross_recon"], self.hparams.k_for_cross_recon)
            data[f"neigh_loss_bac_unscaled"] = \
                self.get_neighbor_loss(data["target"]["pos"], data["target"]["neigh_idxs"], data["source"]["cross_recon"], self.hparams.k_for_cross_recon)

            self.losses[f"neigh_loss_fwd"] = self.hparams.neigh_loss_lambda * data[f"neigh_loss_fwd_unscaled"]
            self.losses[f"neigh_loss_bac"] = self.hparams.neigh_loss_lambda * data[f"neigh_loss_bac_unscaled"]

        #TODO: change current epoch
        if self.hparams.compute_ssw_loss and self.hparams.ssw_loss_lambda > 0.0 and self.current_epoch >= self.hparams.max_epochs-100:
            cov_list=self.loss_stereo_ssw.cal_cov([data["source"]["intermediate_features"],data["target"]["intermediate_features"]])
            data[f"ssw_loss"] = self.loss_stereo_ssw(data["source"]["intermediate_features"], cov_list=cov_list, weight=0.01)
            
            self.losses[f"source_ssw_loss"] = self.hparams.ssw_loss_lambda * data[f"ssw_loss"]

        self.track_metrics(data)
        
        return data

    def test_step(self, test_batch, batch_idx):
        self.batch=test_batch
        self.hparams.mode = 'test'
        self.hparams.batch_idx=batch_idx
        
        label, pinput1, input2, ratio_list, soft_labels = self.extract_labels_for_test(test_batch)

        source = {"pos": pinput1, "id": self.batch['source']["id"]}
        target = {"pos": input2, "id": self.batch['target']["id"]}
        batch = {"source": source, "target": target}
        batch = self(batch)
        p = batch["P_normalized"].clone()
        
        # ### For visualization
        # p_cpu = p.data.cpu().numpy()
        # source_xyz = pinput1.data.cpu().numpy()
        # target_xyz = input2.data.cpu().numpy()
        # label_cpu = label.data.cpu().numpy()
        # np.save("./smal-test/p_{}".format(batch_idx), p_cpu)
        # np.save("./smal-test/source_{}".format(batch_idx), source_xyz)
        # np.save("./smal-test/target_{}".format(batch_idx), target_xyz)
        # np.save("./smal-test/label_{}".format(batch_idx), label_cpu)
        # ###
        
        if self.hparams.use_dualsoftmax_loss:
            temp = 0.0002
            p = p * F.softmax(p/temp, dim=0)*len(p) #With an appropriate temperature parameter, the model achieves higher performance
            p = F.log_softmax(p, dim=-1)


        _ = self.compute_acc(label, ratio_list, soft_labels, p,input2,track_dict=self.tracks,hparams=self.hparams)

        self.log_test_step()
        if self.vis_iter():
            self.visualize(batch, mode='test')

        return True


    def visualize(self, batch, mode="train"):
        visualize_pair_corr(self,batch, mode=mode)
        visualize_reconstructions(self,batch, mode=mode)
        
    @staticmethod
    def add_model_specific_args(parent_parser, task_name, dataset_name, is_lowest_leaf=False):
        parser = ShapeCorrTemplate.add_model_specific_args(parent_parser, task_name, dataset_name, is_lowest_leaf=False)
        parser = non_geo_DGCNN.add_model_specific_args(parser, task_name, dataset_name, is_lowest_leaf=True)
        parser = PointCloudDataset.add_dataset_specific_args(parser, task_name, dataset_name, is_lowest_leaf=False)
        parser.add_argument("--k_for_cross_recon", default=10, type=int, help="number of neighbors for cross reconstruction")

        parser.add_argument("--use_self_recon", nargs="?", default=True, type=str2bool, const=True, help="whether to use self reconstruction")
        parser.add_argument("--k_for_self_recon", default=10, type=int, help="number of neighbors for self reconstruction")
        parser.add_argument("--self_recon_lambda", type=float, default=10.0, help="weight for self reconstruction loss")
        parser.add_argument("--cross_recon_lambda", type=float, default=1.0, help="weight for cross reconstruction loss")

        parser.add_argument("--compute_perm_loss", nargs="?", default=False, type=str2bool, const=True, help="whether to compute permutation loss")
        parser.add_argument("--perm_loss_lambda", type=float, default=1.0, help="weight for permutation loss")

        parser.add_argument("--optimize_pos", nargs="?", default=False, type=str2bool, const=True, help="whether to compute neighbor smoothness loss")
        parser.add_argument("--compute_neigh_loss", nargs="?", default=True, type=str2bool, const=True, help="whether to compute neighbor smoothness loss")
        parser.add_argument("--neigh_loss_lambda", type=float, default=1.0, help="weight for neighbor smoothness loss")
        parser.add_argument("--num_angles", type=int, default=100,)

        parser.add_argument("--use_euclidiean_in_self_recon", nargs="?", default=False, type=str2bool, const=True, help="whether to use self reconstruction")
        parser.add_argument("--use_all_neighs_for_cross_reco", nargs="?", default=False, type=str2bool, const=True, help="whether to use self reconstruction")
        parser.add_argument("--use_all_neighs_for_self_reco", nargs="?", default=False, type=str2bool, const=True, help="whether to use self reconstruction")
        
        '''
        PreEncoder-related args
        '''
        parser.add_argument("--use_preenc", nargs="?", default=True, type=str2bool, const=False, help="whether to use DGCNN pre-encoder")
        
        '''
        Transformer-related args
        '''
        parser.add_argument("--enc_type", type=str, default="vanilla", help="attention mechanism type")
        parser.add_argument("--d_embed", type=int, default=512, help="transformer embedding dim")
        parser.add_argument("--nhead", type=int, default=8, help="transformer multi-head number")
        parser.add_argument("--d_feedforward", type=int, default=1024, help="feed forward dim")
        parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
        parser.add_argument("--transformer_act", type=str, default="relu", help="activation function in transformer")
        parser.add_argument("--pre_norm", nargs="?", default=True, type=str2bool, const=False, help="whether to use prenormalization")
        parser.add_argument("--sa_val_has_pos_emb", nargs="?", default=True, type=str2bool, const=False, help="position embedding in self-attention")
        parser.add_argument("--ca_val_has_pos_emb", nargs="?", default=True, type=str2bool, const=False, help="position embedding in cross-attention")
        parser.add_argument("--attention_type", type=str, default="dot_prod", help="attention mechanism type")
        parser.add_argument("--num_encoder_layers", type=int, default=6, help="the number of transformer encoder layers")
        parser.add_argument("--transformer_encoder_has_pos_emb", nargs="?", default=True, type=str2bool, const=False, help="whether to use position embedding in transformer encoder")
        parser.add_argument("--warmup", nargs="?", default=False, type=str2bool, const=True, help="whether to use warmup")
        parser.add_argument("--steplr", nargs="?", default=False, type=str2bool, const=True, help="whether to use StepLR")
        parser.add_argument("--steplr2", nargs="?", default=False, type=str2bool, const=True, help="whether to use StepLR2")
        parser.add_argument("--testlr", nargs="?", default=False, type=str2bool, const=True, help="whether to use test lr")
        parser.add_argument("--slr", type=float, default= 5e-4, help="steplr learning rate")
        parser.add_argument("--swd", default=5e-4, type=float, help="steplr2 weight decay")
        parser.add_argument("--layer_list", type=list, default=['s', 'c', 's', 'c', 's', 'c', 's', 'c'], help="encoder layer list")
        
        '''
        Shape Selective Whitening Loss-related args
        '''
        parser.add_argument("--old_ssw", nargs="?", default=False, type=str2bool, const=True, help="whether to use old shape whitening loss(similar to stereowhiteningloss)")
        parser.add_argument("--compute_ssw_loss", nargs="?", default=False, type=str2bool, const=True, help="whether to compute shape selective whitening loss")
        parser.add_argument("--ssw_loss_lambda", type=float, default=3.0, help="weight for SSW loss")
        
        '''
        Dual Softmax Loss-related args
        '''
        parser.add_argument("--use_dualsoftmax_loss", nargs="?", default=False, type=str2bool, const=True, help="whether to use dual softmax loss")

        parser.set_defaults(
            optimizer="adam",
            lr=0.0003,
            weight_decay=5e-4,
            max_epochs=300, 
            accumulate_grad_batches=2,
            latent_dim=768,
            bb_size=24,
            num_neighs=27,

            val_vis_interval=20,
            test_vis_interval=20,
        )

        return parser




    def track_metrics(self, data):
        self.tracks[f"source_cross_recon_error"] = self.chamfer_loss(data["source"]["pos"], data["source"]["cross_recon_hard"])
        self.tracks[f"target_cross_recon_error"] = self.chamfer_loss(data["target"]["pos"], data["target"]["cross_recon_hard"])

        
        if self.hparams.use_self_recon:
            self.tracks[f"source_self_recon_loss_unscaled"] = data["source"]["self_recon_loss_unscaled"]
            self.tracks[f"target_self_recon_loss_unscaled"] = data["target"]["self_recon_loss_unscaled"]


        if self.hparams.compute_neigh_loss and self.hparams.neigh_loss_lambda > 0.0:
            self.tracks[f"neigh_loss_fwd_unscaled"] = data[f"neigh_loss_fwd_unscaled"]
            self.tracks[f"neigh_loss_bac_unscaled"] = data[f"neigh_loss_bac_unscaled"]

        # nearest neighbors hit accuracy
        source_pred = data["P_normalized"].argmax(dim=2)
        target_neigh_idxs = data["target"]["neigh_idxs"]

        target_pred = data["P_normalized"].argmax(dim=1)
        source_neigh_idxs = data["source"]["neigh_idxs"]
        
        # uniqueness (number of unique predictions)
        self.tracks[f"uniqueness_fwd"] = uniqueness(source_pred)
        self.tracks[f"uniqueness_bac"] = uniqueness(target_pred)
