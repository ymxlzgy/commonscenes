import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from model.graph import GraphTripleConvNet, GraphTripleConvNet2, _init_weights, make_mlp
from model.sdfusion_txt2shape_model import SDFusionText2ShapeModel
from .diff_utils.visualizer import Visualizer
from .diff_utils.distributed import get_rank
import numpy as np
from helpers.util import bool_flag, _CustomDataParallel
from helpers.lr_scheduler import *
from fvcore.common.param_scheduler import MultiStepParamScheduler


class Sg2ScVAEModel(nn.Module):
    """
    VAE-based network for scene generation and manipulation from a scene graph.
    It has a separate embedding of shape and bounding box latents.
    """
    def __init__(self, vocab, diff_opt, diffusion_bs=8, embedding_dim=128, batch_size=32,
                 train_3d=True,
                 decoder_cat=False,
                 num_box_params=6,
                 distribution_before=True,
                 gconv_pooling='avg', gconv_num_layers=5,
                 mlp_normalization='none',
                 vec_noise_dim=0,
                 use_E2=False,
                 use_AE=False,
                 replace_latent=False,
                 residual=False,
                 use_angles=False,
                 clip=True):
        super(Sg2ScVAEModel, self).__init__()
        gconv_dim = embedding_dim
        gconv_hidden_dim = gconv_dim * 4
        box_embedding_dim = int(embedding_dim)
        if use_angles:
            angle_embedding_dim = int(embedding_dim / 4)
            box_embedding_dim = int(embedding_dim * 3 / 4)
            Nangle = 24
        obj_embedding_dim = embedding_dim
        self.dist_before = distribution_before
        self.replace_all_latent = replace_latent
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.train_3d = train_3d
        self.decoder_cat = decoder_cat
        self.vocab = vocab
        self.vec_noise_dim = vec_noise_dim
        self.use_AE = use_AE
        self.use_angles = use_angles
        # self.lr = lr_full
        self.clip = clip
        add_dim = 0
        if clip:
            add_dim = 512
        self.obj_classes_grained = list(set(vocab['object_idx_to_name_grained']))
        self.edge_list = list(set(vocab['pred_idx_to_name']))
        self.obj_classes_list = list(set(vocab['object_idx_to_name']))
        self.classes = dict(zip(sorted(self.obj_classes_list),range(len(self.obj_classes_list))))
        self.classes_r = dict(zip(self.classes.values(), self.classes.keys()))
        num_objs = len(self.obj_classes_list)
        num_preds = len(self.edge_list)

        # build encoder and decoder nets
        self.obj_embeddings_ec = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.pred_embeddings_ec = nn.Embedding(num_preds, embedding_dim * 2)
        self.obj_embeddings_dc = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.pred_embeddings_dc = nn.Embedding(num_preds, embedding_dim)
        if self.decoder_cat:
            self.pred_embeddings_dc = nn.Embedding(num_preds, embedding_dim * 2)
            self.pred_embeddings_man_dc = nn.Embedding(num_preds, embedding_dim * 3)
        self.d3_embeddings = nn.Linear(num_box_params, box_embedding_dim)
        if self.use_angles:
            self.angle_embeddings = nn.Embedding(Nangle, angle_embedding_dim)
        # weight sharing of mean and var
        self.mean_var = make_mlp([embedding_dim * 2 + add_dim, gconv_hidden_dim, embedding_dim * 2],
                                     batch_norm=mlp_normalization)
        self.mean = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        self.var = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        if self.use_angles:
            self.angle_mean_var = make_mlp([embedding_dim * 2 + add_dim, gconv_hidden_dim, embedding_dim * 2],
                                           batch_norm=mlp_normalization)
            self.angle_mean = make_mlp([embedding_dim * 2, angle_embedding_dim], batch_norm=mlp_normalization, norelu=True)
            self.angle_var = make_mlp([embedding_dim * 2, angle_embedding_dim], batch_norm=mlp_normalization, norelu=True)        # graph conv net

        self.diffusion_bs = diffusion_bs
        self.use_E2 = use_E2
        self.diff_cfg = OmegaConf.load(diff_opt)
        self.diffusion_bs = diffusion_bs if self.diff_cfg.hyper.batch_size is None else self.diff_cfg.hyper.batch_size
        self.Diff = SDFusionText2ShapeModel(self.diff_cfg)
        # visualizer
        self.visualizer = Visualizer(self.diff_cfg)
        if get_rank() == 0:
            self.visualizer.setup_io()

        gconv_kwargs_ec_box = {
            'input_dim_obj': gconv_dim * 2 + add_dim,
            'input_dim_pred': gconv_dim * 2 + add_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        gconv_kwargs_dc = {
            'input_dim_obj': gconv_dim + add_dim,
            'input_dim_pred': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        gconv_kwargs_manipulation = {
            'input_dim_obj': embedding_dim * 3 + add_dim,
            'input_dim_pred': embedding_dim * 3 + add_dim,
            'hidden_dim': gconv_hidden_dim,
            'output_dim': embedding_dim,
            'pooling': gconv_pooling,
            'num_layers': min(gconv_num_layers, 5),
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        gconv_kwargs_ec_rel = {
            'input_dim_obj': gconv_dim * 2 + add_dim,
            'input_dim_pred': gconv_dim * 2 + add_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        if self.decoder_cat:
            gconv_kwargs_dc['input_dim_obj'] = gconv_dim * 2 + add_dim
            gconv_kwargs_dc['input_dim_pred'] = gconv_dim * 2 + add_dim

        self.gconv_net_ec_box = GraphTripleConvNet(**gconv_kwargs_ec_box)

        self.gconv_net_dc = GraphTripleConvNet(**gconv_kwargs_dc)
        self.gconv_net_manipulation = GraphTripleConvNet(**gconv_kwargs_manipulation)

        if self.use_E2:
            self.gconv_net_ec_rel = GraphTripleConvNet2(**gconv_kwargs_ec_rel)

        net_layers = [gconv_dim * 2 + add_dim, gconv_hidden_dim, num_box_params]
        self.d3_net = make_mlp(net_layers, batch_norm=mlp_normalization, norelu=True)

        net_rel_layers = [gconv_dim * 2 + add_dim, 960, 1280]
        if self.Diff.df.conditioning_key == 'concat':
            net_rel_layers = [gconv_dim * 2 + add_dim, 1280, 4096]
        self.rel_mlp = make_mlp(net_rel_layers, batch_norm=mlp_normalization, norelu=True)


        if self.use_angles:
            # angle prediction net
            angle_net_layers = [gconv_dim * 2 + add_dim, gconv_hidden_dim, Nangle]
            self.angle_net = make_mlp(angle_net_layers, batch_norm=mlp_normalization, norelu=True)

        # initialization
        self.d3_embeddings.apply(_init_weights)
        self.mean_var.apply(_init_weights)
        self.mean.apply(_init_weights)
        self.var.apply(_init_weights)
        self.d3_net.apply(_init_weights)
        self.rel_mlp.apply(_init_weights)

        if self.use_angles:
            self.angle_mean_var.apply(_init_weights)
            self.angle_mean.apply(_init_weights)
            self.angle_var.apply(_init_weights)

    def set_cuda(self):
        # set reachable parameters cuda
        #self.cuda()
        self = _CustomDataParallel(self)

        # call unreachable sdfusion set cuda
        # self.Diff.cuda()
        #self.Diff = _CustomDataParallel(self.Diff)

    def encoder(self, objs, triples, boxes_gt, attributes, enc_text_feat, enc_rel_feat, angles_gt=None):
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_ec(objs)
        pred_vecs = self.pred_embeddings_ec(p)
        d3_vecs = self.d3_embeddings(boxes_gt)
        if self.clip:
            obj_vecs_ = torch.cat([enc_text_feat, obj_vecs], dim=1)
            pred_vecs_ = torch.cat([enc_rel_feat, pred_vecs], dim=1)
        else:
            obj_vecs_, pred_vecs_ = obj_vecs, pred_vecs
        if self.use_angles:
            angle_vecs = self.angle_embeddings(angles_gt)
            obj_vecs_ = torch.cat([obj_vecs_, d3_vecs, angle_vecs], dim=1)
        else:
            obj_vecs_ = torch.cat([obj_vecs_, d3_vecs], dim=1)

        obj_vecs_, pred_vecs_ = self.gconv_net_ec_box(obj_vecs_, pred_vecs_, edges)

        obj_vecs_3d = self.mean_var(obj_vecs_)
        mu = self.mean(obj_vecs_3d)
        logvar = self.var(obj_vecs_3d)

        if self.use_angles:
            obj_vecs_angle = self.angle_mean_var(obj_vecs_)
            mu_angle = self.angle_mean(obj_vecs_angle)
            logvar_angle = self.angle_var(obj_vecs_angle)
            mu = torch.cat([mu, mu_angle], dim=1)
            logvar = torch.cat([logvar, logvar_angle], dim=1)

        return mu, logvar

    def encoder_2(self, z, objs, triples, dec_text_feat, dec_rel_feat, attributes, manipulate=False):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_dc(objs)
        pred_vecs = self.pred_embeddings_dc(p)
        if self.clip:
            obj_vecs_ = torch.cat([dec_text_feat, obj_vecs], dim=1)
            pred_vecs_ = torch.cat([dec_rel_feat, pred_vecs], dim=1)
        else:
            obj_vecs_, pred_vecs_=obj_vecs,pred_vecs
        # concatenate noise first
        rel_vecs_ = torch.cat([obj_vecs_, z], dim=1)
        rel_vecs_2 = None
        if self.use_E2:
            rel_vecs_2, _ = self.gconv_net_ec_rel(rel_vecs_, pred_vecs_, edges)
            rel_vecs_2 = self.rel_mlp(rel_vecs_2)
            rel_vecs_2 = torch.unsqueeze(rel_vecs_2, dim=1)
        rel_vecs_ = self.rel_mlp(rel_vecs_)
        rel_vecs_ = torch.unsqueeze(rel_vecs_, dim=1)

        return rel_vecs_, rel_vecs_2

    def manipulate(self, z, objs, triples, dec_text_feat, dec_rel_feat, attributes):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_dc(objs)
        pred_vecs = self.pred_embeddings_man_dc(p)
        if self.clip:
            obj_vecs_ = torch.cat([dec_text_feat, obj_vecs], dim=1)
            pred_vecs_ = torch.cat([dec_rel_feat, pred_vecs], dim=1)
        else:
            obj_vecs_, pred_vecs_= obj_vecs, pred_vecs
        man_z = torch.cat([z, obj_vecs_], dim=1)
        man_z, _ = self.gconv_net_manipulation(man_z, pred_vecs_, edges)

        return man_z

    def decoder(self, z, objs, triples, dec_text_feat, dec_rel_feat, attributes, manipulate=False):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_dc(objs)
        pred_vecs = self.pred_embeddings_dc(p)
        if self.clip:
            obj_vecs_ = torch.cat([dec_text_feat, obj_vecs], dim=1)
            pred_vecs_ = torch.cat([dec_rel_feat, pred_vecs], dim=1)
        else:
            obj_vecs_, pred_vecs_ = obj_vecs, pred_vecs

        # concatenate noise first
        if self.decoder_cat:
            obj_vecs_ = torch.cat([obj_vecs_, z], dim=1)
            obj_vecs_, pred_vecs_ = self.gconv_net_dc(obj_vecs_, pred_vecs_, edges)

        # concatenate noise after gconv
        else:
            obj_vecs_, pred_vecs_ = self.gconv_net_dc(obj_vecs_, pred_vecs_, edges)
            obj_vecs_ = torch.cat([obj_vecs_, z], dim=1)

        d3_pred = self.d3_net(obj_vecs_)
        if self.use_angles:
            angles_pred = F.log_softmax(self.angle_net(obj_vecs_), dim=1)
            return d3_pred, angles_pred
        else:
            return d3_pred

    def decoder_with_additions(self, z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes, distribution=None,gen_shape=False):
        nodes_added = []
        if distribution is not None:
            mu, cov = distribution

        for i in range(len(missing_nodes)):
            ad_id = missing_nodes[i] + i
            nodes_added.append(ad_id)
            noise = np.zeros(z.shape[1])  # np.random.normal(0, 1, 64)
            if distribution is not None:
                zeros = torch.from_numpy(np.random.multivariate_normal(mu, cov, 1)).float().cuda()
            else:
                zeros = torch.from_numpy(noise.reshape(1, z.shape[1]))
            zeros.requires_grad = True
            zeros = zeros.float().cuda()
            z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)

        gen_sdf = None
        if gen_shape:
            un_rel_feat, rel_feat = self.encoder_2(z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)
            sdf_candidates = dec_sdfs
            length = objs.size(0)
            zeros_tensor = torch.zeros_like(sdf_candidates[0])
            mask = torch.ne(sdf_candidates, zeros_tensor)
            ids = torch.unique(torch.where(mask)[0])
            obj_selected = objs[ids]
            diff_dict = {'sdf': dec_sdfs[ids], 'rel': rel_feat[ids], 'uc': un_rel_feat[ids]}
            gen_sdf = self.Diff.rel2shape(diff_dict, uc_scale=3.)
        dec_man_enc_pred = self.decoder(z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)

        keep = []
        for i in range(len(z)):
            if i not in nodes_added and i not in manipulated_nodes:
                keep.append(1)
            else:
                keep.append(0)

        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()

        return dec_man_enc_pred, gen_sdf, keep

    def decoder_with_changes(self, z, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes, distribution=None,gen_shape=False):
        # append zero nodes
        if distribution is not None:
            (mu, cov) = distribution
        nodes_added = []
        for i in range(len(missing_nodes)):
          ad_id = missing_nodes[i] + i
          nodes_added.append(ad_id)
          noise = np.zeros(self.embedding_dim) # np.random.normal(0, 1, 64)
          if distribution is not None:
            zeros = torch.from_numpy(np.random.multivariate_normal(mu, cov, 1)).float().cuda()
          else:
            zeros = torch.from_numpy(noise.reshape(1, z.shape[1]))
          zeros.requires_grad = True
          zeros = zeros.float().cuda()
          z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)

        # mark changes in nodes
        change_repr = []
        for i in range(len(z)):
            if i not in nodes_added and i not in manipulated_nodes:
                noisechange = np.zeros(self.embedding_dim)
            else:
                noisechange = np.random.normal(0, 1, self.embedding_dim)
            change_repr.append(torch.from_numpy(noisechange).float().cuda())
        change_repr = torch.stack(change_repr, dim=0)
        z_prime = torch.cat([z, change_repr], dim=1)
        z_prime = self.manipulate(z_prime, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)

        if not self.replace_all_latent:
            # take original nodes when untouched
            touched_nodes = torch.tensor(sorted(nodes_added + manipulated_nodes)).long()
            for touched_node in touched_nodes:
                z = torch.cat([z[:touched_node], z_prime[touched_node:touched_node + 1], z[touched_node + 1:]], dim=0)
        else:
            z = z_prime

        gen_sdf = None
        if gen_shape:
            un_rel_feat, rel_feat = self.encoder_2(z, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)
            sdf_candidates = dec_sdfs
            length = dec_objs.size(0)
            zeros_tensor = torch.zeros_like(sdf_candidates[0])
            mask = torch.ne(sdf_candidates, zeros_tensor)
            ids = torch.unique(torch.where(mask)[0])
            obj_selected = dec_objs[ids]
            diff_dict = {'sdf': dec_sdfs[ids], 'rel': rel_feat[ids], 'uc': un_rel_feat[ids]}
            gen_sdf = self.Diff.rel2shape(diff_dict,uc_scale=3.)
        dec_man_enc_pred = self.decoder(z, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat,
                                        attributes)
        if self.use_angles:
            num_dec_objs = len(dec_man_enc_pred[0])
        else:
            num_dec_objs = len(dec_man_enc_pred)

        keep = []
        for i in range(num_dec_objs):
          if i not in nodes_added and i not in manipulated_nodes:
            keep.append(1)
          else:
            keep.append(0)

        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()

        return dec_man_enc_pred, gen_sdf, keep

    def balance_objects(self, id_list, object_list, n):
        assert len(id_list) == len(object_list), "id_list and object_list must have the same length"

        unique_ids = torch.unique(id_list)
        selected_object_indices = []

        # find n fine-grained objects to ensure diffusion meet all fine-grained classes in the scene
        if len(unique_ids) >= n:
            sampled_unique_ids = random.sample(unique_ids.tolist(), n)

        # fine-grained classes less than n, we take all fine-grained classes and randomly obtain the rest
        else:
            sampled_unique_ids = unique_ids.tolist()
            remaining_n = n - len(unique_ids)
            sampled_unique_ids += random.choices(id_list.tolist(), k=remaining_n)

        # find the corresponding ids in the coarse object classes
        for selected_id in sampled_unique_ids:
            selected_indices = (id_list == selected_id).nonzero(as_tuple=True)[0]
            selected_object_idx = selected_indices[random.choice(range(len(selected_indices)))]
            selected_object_indices.append(selected_object_idx)

        return torch.tensor(selected_object_indices)

    def select_sdfs(self, dec_objs_to_scene, dec_objs, dec_objs_grained, dec_sdfs, uc_rel_feat, c_rel_feat, random=False):
        dec_objs_to_scene = dec_objs_to_scene.detach().cpu().numpy()
        batch_size = np.max(dec_objs_to_scene) + 1
        sdf_selected = []
        uc_rel_selected = []
        c_rel_selected = []
        obj_cat_selected = []
        num_obj = int(np.ceil(self.diffusion_bs / batch_size))
        for i in range(batch_size):
            sdf_candidates = dec_sdfs[np.where(dec_objs_to_scene == i)[0]]
            obj_cat = dec_objs[np.where(dec_objs_to_scene == i)[0]]
            obj_cat_grained = dec_objs_grained[np.where(dec_objs_to_scene == i)[0]]
            uc_rel = uc_rel_feat[np.where(dec_objs_to_scene == i)[0]]
            c_rel = c_rel_feat[np.where(dec_objs_to_scene == i)[0]]
            length = obj_cat.size(0)
            zeros_tensor = torch.zeros_like(sdf_candidates[0])
            mask = torch.ne(sdf_candidates, zeros_tensor)
            # mask = torch.any(mask, dim=1)
            ids = torch.unique(torch.where(mask)[0])
            if random:
                perm = torch.randperm(len(ids))
                random_elements = ids[perm[:num_obj]]
                # for random_element in random_elements:
                sdf_selected.append(sdf_candidates[random_elements])
                uc_rel_selected.append(uc_rel[random_elements])
                c_rel_selected.append(c_rel[random_elements])
                obj_cat_selected.append(obj_cat[random_elements])
            else:
                # TODO: balance every object.
                # essential_names = self.obj_classes_grained
                selected_cat_ids = self.balance_objects(obj_cat_grained[ids], obj_cat[ids], num_obj)
                sdf_selected.append(sdf_candidates[selected_cat_ids])
                uc_rel_selected.append(uc_rel[selected_cat_ids])
                c_rel_selected.append(c_rel[selected_cat_ids])
                obj_cat_selected.append(obj_cat[selected_cat_ids])

        sdf_selected = torch.cat(sdf_selected, dim=0).cuda()
        uc_rel_selected = torch.cat(uc_rel_selected, dim=0).cuda()
        c_rel_selected = torch.cat(c_rel_selected, dim=0).cuda()
        obj_cat_selected = torch.cat(obj_cat_selected, dim=0)
        diff_dict = {'sdf': sdf_selected[:self.diffusion_bs], 'uc': uc_rel_selected[:self.diffusion_bs], 'rel': c_rel_selected[:self.diffusion_bs]}
        return obj_cat_selected[:self.diffusion_bs], diff_dict

    def forward(self, enc_objs, enc_triples, enc, enc_text_feat, enc_rel_feat, attributes, enc_objs_to_scene, dec_objs, dec_objs_grained,
                dec_triples, dec, dec_text_feat, dec_rel_feat, dec_attributes, dec_objs_to_scene, missing_nodes, manipulated_nodes, dec_sdfs,
                enc_angles=None, dec_angles=None):

        mu, logvar = self.encoder(enc_objs, enc_triples, enc, attributes, enc_text_feat, enc_rel_feat, enc_angles)
        if self.use_AE:
            z = mu
        else:
            # reparameterization
            std = torch.exp(0.5*logvar)
            # standard sampling
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)

        # append zero nodes
        nodes_added = []
        for i in range(len(missing_nodes)):
          ad_id = missing_nodes[i] + i
          nodes_added.append(ad_id)
          noise = np.zeros(self.embedding_dim) # np.random.normal(0, 1, 64)
          zeros = torch.from_numpy(noise.reshape(1, self.embedding_dim))
          zeros.requires_grad = True
          zeros = zeros.float().cuda()
          z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)

        # mark changes in nodes
        change_repr = []
        for i in range(len(z)):
            if i not in nodes_added and i not in manipulated_nodes:
                noisechange = np.zeros(self.embedding_dim)
            else:
                noisechange = np.random.normal(0, 1, self.embedding_dim)
            change_repr.append(torch.from_numpy(noisechange).float().cuda())
        change_repr = torch.stack(change_repr, dim=0)
        z_prime = torch.cat([z, change_repr], dim=1)
        z_prime = self.manipulate(z_prime, dec_objs, dec_triples, dec_text_feat, dec_rel_feat, attributes)

        if not self.replace_all_latent:
            # take original nodes when untouched
            touched_nodes = torch.tensor(sorted(nodes_added + manipulated_nodes)).long()
            for touched_node in touched_nodes:
                z = torch.cat([z[:touched_node], z_prime[touched_node:touched_node + 1], z[touched_node + 1:]], dim=0)
        else:
            z = z_prime

        # TODO: z, clip, dec_objs, dec_triples to GCN -> relation embeddings -> diffusion
        uc_rel_feat, c_rel_feat = self.encoder_2(z, dec_objs, dec_triples, dec_text_feat, dec_rel_feat, attributes)
        if c_rel_feat == None:
            c_rel_feat = uc_rel_feat
        obj_selected, diff_dict = self.select_sdfs(dec_objs_to_scene, dec_objs, dec_objs_grained, dec_sdfs, uc_rel_feat, c_rel_feat, random=False)
        # if False:
        #     id_ = torch.where(dec_objs==31)
        #     obj_selected = torch.tensor([31]).to(torch.int64)
        #     diff_dict = {'sdf': dec_sdfs[id_],'rel':c_rel_feat[id_],'uc':uc_rel_feat[id_]}
        self.Diff.set_input(diff_dict)
        self.Diff.set_requires_grad([self.Diff.df], requires_grad=True)
        self.Diff.forward()
        dec_shapes_selected = None


        if self.use_angles:
            dec_man_enc_pred, angles_pred = self.decoder(z, dec_objs, dec_triples, dec_text_feat, dec_rel_feat, attributes)
            orig_angles = []
            orig_gt_angles = []
        else:
            dec_man_enc_pred = self.decoder(z, dec_objs, dec_triples, attributes)

        orig_d3 = []
        orig_gt_d3 = []
        orig_gt_shapes = []
        keep = []
        for i in range(len(dec_man_enc_pred)):
          if i not in nodes_added and i not in manipulated_nodes:
            orig_d3.append(dec_man_enc_pred[i:i+1])
            orig_gt_d3.append(dec[i:i+1])
            orig_gt_shapes.append(dec_sdfs[i:i + 1])
            if self.use_angles:
                orig_angles.append(angles_pred[i:i+1])
                orig_gt_angles.append(dec_angles[i:i+1])
            keep.append(1)
          else:
            keep.append(0)

        orig_d3 = torch.cat(orig_d3, dim=0)
        orig_gt_shapes = torch.cat(orig_gt_shapes, dim=0)
        orig_gt_d3 = torch.cat(orig_gt_d3, dim=0)
        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()
        if self.use_angles:
            orig_angles = torch.cat(orig_angles, dim=0)
            orig_gt_angles = torch.cat(orig_gt_angles, dim=0)
        else:
            orig_angles, orig_gt_angles, angles_pred = None, None, None

        return mu, logvar, orig_gt_d3, orig_gt_angles, orig_gt_shapes, orig_d3, orig_angles, dec_man_enc_pred, angles_pred, [obj_selected, dec_shapes_selected], keep

    def forward_no_mani(self, objs, triples, enc, attributes):
        mu, logvar = self.encoder(objs, triples, enc, attributes)
        # reparameterization
        std = torch.exp(0.5 * logvar)
        # standard sampling
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        keep = []
        dec_man_enc_pred = self.decoder(z, objs, triples, attributes)
        for i in range(len(dec_man_enc_pred)):
            keep.append(1)
        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()
        return mu, logvar, dec_man_enc_pred, keep

    def sampleShape(self, point_classes_idx, point_ae, mean_est_shape, cov_est_shape, dec_objs, dec_triplets,
                    attributes=None):
        with torch.no_grad():
            z_shape = []
            for idxz in dec_objs:
                idxz = int(idxz.cpu())
                if idxz in point_classes_idx:
                    z_shape.append(torch.from_numpy(
                        np.random.multivariate_normal(mean_est_shape[idxz], cov_est_shape[idxz], 1)).float().cuda())
                else:
                    z_shape.append(torch.from_numpy(np.random.multivariate_normal(mean_est_shape[-1],
                                                                                  cov_est_shape[-1],
                                                                                  1)).float().cuda())
            z_shape = torch.cat(z_shape, 0)

            dc_shapes = self.decoder(z_shape, dec_objs, dec_triplets, attributes)
            points = point_ae.forward_inference_from_latent_space(dc_shapes, point_ae.get_grid())
        return points, dc_shapes

    def sampleBoxes(self, mean_est, cov_est, dec_objs, dec_triplets, attributes=None):
        with torch.no_grad():
            z = torch.from_numpy(
            np.random.multivariate_normal(mean_est, cov_est, dec_objs.size(0))).float().cuda()

            return self.decoder(z, dec_objs, dec_triplets, attributes)

    def sample(self, point_classes_idx, mean_est, cov_est, dec_objs, dec_triplets, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, attributes=None, gen_shape=False):
        with torch.no_grad():
            z = torch.from_numpy(np.random.multivariate_normal(mean_est, cov_est, dec_objs.size(0))).float().cuda()
            gen_sdf = None
            if gen_shape:
                un_rel_feat, rel_feat = self.encoder_2(z, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)
                sdf_candidates = dec_sdfs
                length = dec_objs.size(0)
                zeros_tensor = torch.zeros_like(sdf_candidates[0])
                mask = torch.ne(sdf_candidates, zeros_tensor)
                ids = torch.unique(torch.where(mask)[0])
                obj_selected = dec_objs[ids]
                if rel_feat == None:
                    rel_feat = un_rel_feat
                diff_dict = {'sdf': dec_sdfs[ids], 'rel': rel_feat[ids], 'uc': un_rel_feat[ids]}
                gen_sdf = self.Diff.rel2shape(diff_dict,uc_scale=3.)


            return self.decoder(z, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes), gen_sdf

    # 0-20k->20k-60k->60k-100k->100k-
    # 1e-4 -> 5e-5 -> 1e-5 -> 5e-6
    def lr_lambda(self, counter):
        # 10000
        if counter < 20000:
            return 1.0
        # 40000
        elif counter < 60000:
            return 5e-5 / 1e-4
        # 80000
        elif counter < 100000:
            return 1e-5 / 1e-4
        else:
            return 5e-6 / 1e-4

    def optimizer_ini(self):
        # optimizer_box for model
        #if self.isTrain:
        # initialize optimizers

        # first branch
        #params = filter(lambda p: p.requires_grad, list(self.parameters()))
        params = [p for p in self.parameters() if p.requires_grad == True]
        # diffusion
        df_params = self.Diff.trainable_params

        trainable_params = params + df_params


        self.optimizerFULL = optim.AdamW(trainable_params, lr=1e-4)  # self.lr) #TODO: replace with opt.training.lr
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizerFULL, lr_lambda=self.lr_lambda)

        # self.optimizerFULL = optim.AdamW(trainable_params, lr=1e-5)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizerFULL, 30000, 0.9)
        # self.scheduler = LRMultiplier(
        #     self.optimizerFULL,
        #     WarmupParamScheduler(
        #         MultiStepParamScheduler(
        #             [1, 0.1, 0.01],
        #             milestones=[20000, 100000],
        #             num_updates=200000,
        #         ),
        #         warmup_factor=0.001,
        #         warmup_length=1000 / 200000,  # 1000 / total_steps_16bs,
        #         warmup_method="linear",
        #     ),
        #     max_iter=200000
        # )

        self.optimizers = [self.optimizerFULL]
        # self.schedulers = [self.scheduler]

        #self.print_networks(verbose=False)

        #self.optimizer_box = optim.Adam(params, lr=self.lr)
        #self.optimizer_box.step()

        # update learning rate (called once every epoch)
    def update_learning_rate(self):
        # update lr
        # for scheduler in self.schedulers:
        self.scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('[*] learning rate = %.7f' % lr)

        return lr

    def state_dict(self, epoch, counter):
        state_dict_1 = super(Sg2ScVAEModel, self).state_dict()
        state_dict_2 = {
            'epoch': epoch,
            'counter': counter,
            'vqvae': self.Diff.vqvae_module.state_dict(),
            # 'cond_model': self.cond_model_module.state_dict(),
            'df': self.Diff.df_module.state_dict(),
            #'diff_opt': self.Diff.optimizer.state_dict()
            'opt': self.optimizerFULL.state_dict()
        }
        state_dict_1.update(state_dict_2)
        return state_dict_1
    def collect_train_statistics(self, train_loader, with_points=False):
        # model = model.eval()
        mean_cat = None
        if with_points:
            means, vars = {}, {}
            for idx in train_loader.dataset.point_classes_idx:
                means[idx] = []
                vars[idx] = []
            means[-1] = []
            vars[-1] = []

        for idx, data in enumerate(train_loader):
            if data == -1:
                continue
            try:
                objs, triples, tight_boxes, objs_to_scene, triples_to_scene = data['decoder']['objs'], \
                                                                              data['decoder']['tripltes'], \
                                                                              data['decoder']['boxes'], \
                                                                              data['decoder']['obj_to_scene'], \
                                                                              data['decoder']['triple_to_scene']

                enc_text_feat, enc_rel_feat = None, None
                if 'feats' in data['decoder']:
                    encoded_points = data['decoder']['feats']
                    encoded_points = encoded_points.cuda()
                if 'text_feats' in data['decoder'] and 'rel_feats' in data['decoder']:
                    enc_text_feat, enc_rel_feat = data['decoder']['text_feats'], data['decoder']['rel_feats']
                    enc_text_feat, enc_rel_feat = enc_text_feat.cuda(), enc_rel_feat.cuda()

            except Exception as e:
                print('Exception', str(e))
                continue

            objs, triples, tight_boxes = objs.cuda(), triples.cuda(), tight_boxes.cuda()
            boxes = tight_boxes[:, :6]
            angles = tight_boxes[:, 6].long() - 1
            angles = torch.where(angles > 0, angles, torch.zeros_like(angles))
            attributes = None


            mean, logvar = self.encoder(objs, triples, boxes, attributes, enc_text_feat, enc_rel_feat, angles)
            mean, logvar = mean.cpu().clone(), logvar.cpu().clone()

            mean = mean.data.cpu().clone()
            if mean_cat is None:
                mean_cat = mean
            else:
                mean_cat = torch.cat([mean_cat, mean], dim=0)

        mean_est = torch.mean(mean_cat, dim=0, keepdim=True)  # size 1*embed_dim
        mean_cat = mean_cat - mean_est
        cov_est_ = np.cov(mean_cat.numpy().T)
        n = mean_cat.size(0)
        d = mean_cat.size(1)
        cov_est = np.zeros((d, d))
        for i in range(n):
            x = mean_cat[i].numpy()
            cov_est += 1.0 / (n - 1.0) * np.outer(x, x)
        mean_est = mean_est[0]

        return mean_est, cov_est_
