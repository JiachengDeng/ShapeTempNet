from visualization.visualize_api import visualize_pair_corr, visualize_reconstructions
from data.point_cloud_db.point_cloud_dataset import PointCloudDataset

from models.sub_models.dgcnn.dgcnn_modular import DGCNN_MODULAR
from models.sub_models.dgcnn.dgcnn import get_graph_feature

from models.sub_models.cross_attention.transformers import FlexibleTransformerEncoder, LuckTransformerEncoder, TemplateTransformerEncoder
from models.sub_models.cross_attention.transformers import TransformerSelfLayer, TransformerCrossLayer, LuckSelfLayer
from models.sub_models.cross_attention.position_embedding import PositionEmbeddingCoordsSine, \
    PositionEmbeddingLearned
from models.sub_models.cross_attention.warmup import WarmUpScheduler

from models.sub_models.autoencoder.auto_encoder import PointCloudAE


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

class TemplatePointCorr(ShapeCorrTemplate):
    
    def __init__(self, hparams, **kwargs):
        """Stub."""
        super(TemplatePointCorr, self).__init__(hparams, **kwargs)
        
        self.encoder_norm = nn.LayerNorm(self.hparams.d_embed) if self.hparams.pre_norm else None
        
        self.encoder = TemplateTransformerEncoder(hparams, self.hparams.layer_list, self.encoder_norm, True)
        
        self.autoencoder = PointCloudAE(self.hparams.ae_d_embed, 1024)
        self.accumulate_count = 0
        self.template_embed = torch.zeros((1,self.hparams.ae_d_embed)).cuda()
        
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
    

    def normalize_data(self, batch_data):
        """ Normalize the batch data, use coordinates of the block centered at origin,
            Input:
                BxNxC array
            Output:
                BxNxC array
        """
        B, N, C = batch_data.shape
        normal_data = torch.zeros((B, N, C)).cuda()
        for b in range(B):
            pc = batch_data[b]
            centroid = torch.mean(pc, axis=0)
            pc = pc - centroid
            m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
            pc = pc / m
            normal_data[b] = pc
        return normal_data

    def rotate_point_cloud_by_angle(self,batch_data):
        """ Rotate the point cloud along up direction with certain angle.
            Input:
            BxNx3 array, original batch of point clouds
            Return:
            BxNx3 array, rotated batch of point clouds
        """
        rotated_data = torch.zeros(batch_data.shape, dtype=torch.float32).cuda()
        rotated_gt = torch.zeros((batch_data.shape[0]), dtype=torch.long).cuda()
        for k in range(batch_data.shape[0]):
            toss = torch.randint(8,(1,))
            #toss = 6
            rotation_angle = self.ANGLE[toss]
            #rotation_angle = bimodal.sample(1)
            #rotation_angle = torch.rand((1)) * 2 * np.pi
            cosval = torch.cos(rotation_angle)
            sinval = torch.sin(rotation_angle)
            rotated_gt[k] = toss
            #rotated_gt[k,1] = cosval
            rotation_matrix = torch.tensor([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]]).cuda()
            shape_pc = batch_data[k,:,0:3]
            rotated_data[k,:,0:3] = torch.mm(shape_pc.reshape((-1, 3)), rotation_matrix)

        return rotated_data,rotated_gt

    def rotate_point_cloud_for_animal(self,batch_data, rotation_angle, axis = 'Y'):
        """ Rotate the point cloud along up direction with certain angle.
            Input:
            BxNx3 array, original batch of point clouds
            Return:
            BxNx3 array, rotated batch of point clouds
        """
        rotated_data = torch.zeros(batch_data.shape, dtype=torch.float32).cuda()
        for k in range(batch_data.shape[0]):
            cosval = torch.cos(rotation_angle)
            sinval = torch.sin(rotation_angle)
            if axis == 'X':
                rotation_matrix = torch.tensor([[1, 0, 0],
                                            [0, cosval, sinval],
                                            [0, -sinval, cosval]]).cuda()
            elif axis == 'Y':
                rotation_matrix = torch.tensor([[cosval, 0, sinval],
                                            [0, 1, 0],
                                            [-sinval, 0, cosval]]).cuda()
            elif axis == 'Z':
                rotation_matrix = torch.tensor([[cosval, sinval, 0],
                                            [-sinval, cosval, 0],
                                            [0, 0, 1]]).cuda()

            shape_pc = batch_data[k,:,0:3]
            rotated_data[k,:,0:3] = torch.mm(shape_pc.reshape((-1, 3)), rotation_matrix)

        return rotated_data
    
    def compute_self_features(self, source, target): 
        # if  self.hparams.mode == 'train' or self.hparams.mode == 'val':
        #   source["pos"] = self.rotate_point_cloud_for_animal(source["pos"], torch.tensor(-0.5*torch.pi), 'Z') #SMAL
        #   source["pos"] = self.rotate_point_cloud_for_animal(source["pos"], torch.tensor(-0.5*torch.pi), 'X') #SMAL
        #   target["pos"] = self.rotate_point_cloud_for_animal(target["pos"], torch.tensor(-0.5*torch.pi), 'Z') #SMAL
        #   target["pos"] = self.rotate_point_cloud_for_animal(target["pos"], torch.tensor(-0.5*torch.pi), 'X') #SMAL
        # else:
        #   #print('111')
        #   source["pos"] = self.rotate_point_cloud_for_animal(source["pos"], torch.tensor(-0.5*torch.pi), 'X') #TOSCA
        #   target["pos"] = self.rotate_point_cloud_for_animal(target["pos"], torch.tensor(-0.5*torch.pi), 'X') #TOSCA
        # source["pos"] = self.normalize_data(source["pos"])
        # target["pos"] = self.normalize_data(target["pos"])

        # if  self.hparams.mode == 'train' or self.hparams.mode == 'val':
        #     #src_pos_student,rotated_src_student= self.rotate_point_cloud_by_angle(source["pos"])
        #     target["pos"],rotated_gt_student = self.rotate_point_cloud_by_angle(target["pos"])
        template = {}
        src_ae_embed, src_dec_pc = self.autoencoder(source["pos"])
        tgt_ae_embed, tgt_dec_pc = self.autoencoder(target["pos"])
        
        source["ae_pos"] = src_dec_pc
        target["ae_pos"] = tgt_dec_pc
        
        # ! Bug: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
        self.template_embed = (self.template_embed.detach()*self.accumulate_count+ torch.mean(src_ae_embed, dim=0, keepdim=True) + torch.mean(tgt_ae_embed, dim=0, keepdim=True))/(self.accumulate_count+2)
        self.accumulate_count += 2
        
        _, template["pos"] = self.autoencoder(None, self.template_embed)  #template_shape = [1, 1024, 3]

        if self.hparams.batch_idx==0:
            templa = template["pos"].detach().cpu().numpy()
            np.save("./template-shape-tosca-traintemp/epoch_{}".format(self.current_epoch), templa)


        template["pos"] = template["pos"].repeat(source["pos"].shape[0],1,1)  #convenient for code but not efficient
    
        # * Compute template neigh idxs same as src and target before
        template["edge_index"], template["neigh_idxs"] = self.edge_neibor_compute(template["pos"])
        
        template_out, template_inter = self.encoder(
            template["pos"].transpose(0,1),  
            src_xyz = template["pos"],
            src_neigh = template["neigh_idxs"],
        )
        
        src_out, src_inter = self.encoder(
            source["pos"].transpose(0,1),  
            src_xyz = source["pos"],
            src_neigh = source["neigh_idxs"],
        )
        
        tgt_out, tgt_inter = self.encoder(
            target["pos"].transpose(0,1),
            src_xyz = target["pos"],
            src_neigh = target["neigh_idxs"],
        )
        
        template["dense_output_features"] = template_out.transpose(1,2)   #template["dense_output_features"] = [B, N, C]
        source["dense_output_features"] = src_out.transpose(1,2)
        target["dense_output_features"] = tgt_out.transpose(1,2)
        
        template["inter_features"] = template_inter  #template["inter_features"] = [[B,C1,N], [B,C2,N], [B,C3,N], [B,C4,N]]
        source["inter_features"] = src_inter
        target["inter_features"] = tgt_inter

        return source, target, template

    def forward_source_target(self, source, target):
        
        ###transformers     
        source, target, template = self.compute_self_features(source, target)
        ###

        # measure cross similarity
        P_non_normalized = switch_functions.measure_similarity(self.hparams.similarity_init, source["dense_output_features"], target["dense_output_features"])
        P_st_non_normalized = switch_functions.measure_similarity(self.hparams.similarity_init, source["dense_output_features"], template["dense_output_features"])
        P_tt_non_normalized = switch_functions.measure_similarity(self.hparams.similarity_init, target["dense_output_features"], template["dense_output_features"])
        
        temperature = None
        
        P_normalized = P_non_normalized
        P_st_normalized = P_st_non_normalized
        P_tt_normalized = P_tt_non_normalized

        # cross nearest neighbors and weights
        source["cross_nn_weight"], source["cross_nn_sim"], source["cross_nn_idx"], target["cross_nn_weight"], target["cross_nn_sim"], target["cross_nn_idx"] =\
            get_s_t_neighbors(self.hparams.k_for_cross_recon, P_normalized, sim_normalization=self.hparams.sim_normalization)
              
        if self.hparams.template_cross_lambda > 0.0:    
            source["t_cross_nn_weight"], source["t_cross_nn_sim"], source["t_cross_nn_idx"], template["s_cross_nn_weight"], template["s_cross_nn_sim"], template["s_cross_nn_idx"] =\
                get_s_t_neighbors(self.hparams.k_for_cross_recon, P_st_normalized, sim_normalization=self.hparams.sim_normalization)
            target["t_cross_nn_weight"], target["t_cross_nn_sim"], target["t_cross_nn_idx"], template["t_cross_nn_weight"], template["t_cross_nn_sim"], template["t_cross_nn_idx"] =\
                get_s_t_neighbors(self.hparams.k_for_cross_recon, P_tt_normalized, sim_normalization=self.hparams.sim_normalization)

        # cross reconstruction
        source["cross_recon"], source["cross_recon_hard"] = self.reconstruction(source["pos"], target["cross_nn_idx"], target["cross_nn_weight"], self.hparams.k_for_cross_recon)
        target["cross_recon"], target["cross_recon_hard"] = self.reconstruction(target["pos"], source["cross_nn_idx"], source["cross_nn_weight"], self.hparams.k_for_cross_recon)
        
        if self.hparams.template_cross_lambda > 0.0:
            source["t_cross_recon"], source["t_cross_recon_hard"] = self.reconstruction(source["pos"], template["s_cross_nn_idx"], template["s_cross_nn_weight"], self.hparams.k_for_cross_recon)
            template["s_cross_recon"], template["s_cross_recon_hard"] = self.reconstruction(template["pos"], source["t_cross_nn_idx"], source["t_cross_nn_weight"], self.hparams.k_for_cross_recon)
            target["t_cross_recon"], target["t_cross_recon_hard"] = self.reconstruction(target["pos"], template["t_cross_nn_idx"], template["t_cross_nn_weight"], self.hparams.k_for_cross_recon)
            template["t_cross_recon"], template["t_cross_recon_hard"] = self.reconstruction(template["pos"], target["t_cross_nn_idx"], target["t_cross_nn_weight"], self.hparams.k_for_cross_recon)

        return source, target, template, P_normalized, temperature

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
            data[shape]["edge_index"], data[shape]["neigh_idxs"] = self.edge_neibor_compute(data[shape]["pos"])
        # dense features, similarity, and cross reconstruction

        data["source"], data["target"], data["template"], data["P_normalized"], data["temperature"] = self.forward_source_target(data["source"], data["target"])
        
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

        # * Template-related losses
        #chamfer-loss for auto encoder
        if self.hparams.ae_lambda > 0.0:
            self.losses['ae_loss'] = self.hparams.ae_lambda*(self.chamfer_loss(data["source"]["pos"], data["source"]["ae_pos"])+self.chamfer_loss(data["target"]["pos"], data["target"]["ae_pos"]))
        #template cross reconstruction loss
        if self.hparams.template_cross_lambda > 0.0:
            self.losses["template_source_cross_recon_loss"] = self.hparams.template_cross_lambda * (self.chamfer_loss(data["source"]["pos"], data["source"]["t_cross_recon"])+self.chamfer_loss(data["template"]["pos"], data["template"]["s_cross_recon"]))
            self.losses["template_target_cross_recon_loss"] =self.hparams.template_cross_lambda * (self.chamfer_loss(data["target"]["pos"], data["target"]["t_cross_recon"])+self.chamfer_loss(data["template"]["pos"], data["target"]["t_cross_recon"]))

        #template mapping loss
        if self.hparams.template_neigh_lambda > 0.0:
            pass
        
        # * Template-related losses
        
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
        
    def edge_neibor_compute(self, pos):  #pos = [B, N, 3]
        edge_index = [
            knn(pos[i], pos[i], self.hparams.num_neighs,)
            for i in range(pos.shape[0])
        ]
        neigh_idxs = torch.stack(
            [edge_index[i][1].reshape(pos.shape[1], -1) for i in range(pos.shape[0])]
        )
        
        return edge_index, neigh_idxs
        
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
        parser.add_argument("--layer_list", type=list, default=['s', 's', 's', 's'], help="encoder layer list")
        
        '''
        Dual Softmax Loss-related args
        '''
        parser.add_argument("--use_dualsoftmax_loss", nargs="?", default=False, type=str2bool, const=True, help="whether to use dual softmax loss")


        '''
        Template-related args
        '''
        parser.add_argument("--ae_d_embed", type=int, default=512, help="auto encoder embedding dim")
        parser.add_argument("--ae_lambda", type=float, default=1.0, help="weight for auto encoder loss")
        parser.add_argument("--template_cross_lambda", type=float, default=1.0, help="weight for cross reconstruction loss between template and source/target")
        parser.add_argument("--template_neigh_lambda", type=float, default=1.0, help="weight for neighbor smoothness loss between template and source/target")
        
        parser.set_defaults(
            optimizer="adam",
            lr=0.0003,
            weight_decay=5e-4,
            max_epochs=600, 
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
