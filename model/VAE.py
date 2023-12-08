import json

import torch
import torch.nn as nn

import pickle
import os
import glob

import trimesh
from termcolor import colored
from model.VAEGAN_V1BOX import Sg2ScVAEModel as v1_box
from model.VAEGAN_V1FULL import Sg2ScVAEModel as v1_full
from model.VAEGAN_V2BOX import Sg2ScVAEModel as v2_box
from model.VAEGAN_V2FULL import Sg2ScVAEModel as v2_full


class VAE(nn.Module):

    def __init__(self, root="../GT",type='v1_box', diff_opt = '../config/v2_full.yaml', vocab=None, replace_latent=False, with_changes=True, distribution_before=True,
                 residual=False, gconv_pooling='avg', with_angles=False, num_box_params=6, lr_full=None, deepsdf=False, clip=True, with_E2=True):
        super().__init__()
        assert type in ['v1_box', 'v1_full', 'v2_box', 'v2_full'], '{} is not included'.format(type)

        self.type_ = type
        self.vocab = vocab
        self.with_angles = with_angles
        self.epoch = 0
        self.v1full_database = os.path.join(root, "DEEPSDF_reconstruction")

        if self.type_ == 'v1_box':
            assert replace_latent is not None
            self.vae_box = v1_box(vocab, embedding_dim=64, decoder_cat=True, mlp_normalization="batch",
                               input_dim=num_box_params, replace_latent=replace_latent, use_angles=with_angles,
                               residual=residual, gconv_pooling=gconv_pooling, gconv_num_layers=5)
        elif self.type_ == 'v1_full':
            self.classes_ = sorted(list(set(self.vocab['object_idx_to_name'])))
            self.v1code_base = os.path.join(self.v1full_database, 'Codes')
            self.v1mesh_base = os.path.join(self.v1full_database, 'Meshes')
            # self.code_dict_path = os.path.join(self.v1full_database, 'deepsdf_code.json')
            id_names = os.listdir(self.v1code_base)
            self.code_dict = {}
            for id_name in id_names:
                latent_code = torch.load(os.path.join(self.v1code_base, id_name, 'sdf.pth'), map_location="cpu")[0]
                latent_code = latent_code.detach().numpy()
                self.code_dict[id_name] = latent_code[0]
            assert distribution_before is not None and replace_latent is not None and with_changes is not None
            self.vae = v1_full(vocab, embedding_dim=128, decoder_cat=True, mlp_normalization="batch",
                              gconv_num_layers=5, gconv_num_shared_layer=5, with_changes=with_changes, use_angles=with_angles,
                              distribution_before=distribution_before, replace_latent=replace_latent,
                              num_box_params=num_box_params, residual=residual, shape_input_dim=256 if deepsdf else 128)
        elif self.type_ == 'v2_box':
            assert replace_latent is not None
            self.vae_box = v2_box(vocab, embedding_dim=64, decoder_cat=True, mlp_normalization="batch",
                               input_dim=num_box_params, replace_latent=replace_latent, use_angles=with_angles,
                               residual=residual, gconv_pooling=gconv_pooling, gconv_num_layers=5)
        elif self.type_ == 'v2_full':
            self.diff_opt = diff_opt
            assert distribution_before is not None and replace_latent is not None and with_changes is not None
            self.vae_v2 = v2_full(vocab, self.diff_opt, diffusion_bs=16, embedding_dim=64, decoder_cat=True, mlp_normalization="batch",
                              gconv_num_layers=5, use_angles=with_angles, distribution_before=distribution_before, use_E2=with_E2, replace_latent=replace_latent,
                              num_box_params=num_box_params, residual=residual, clip=clip)
            self.vae_v2.optimizer_ini()
        self.counter = 0

    def set_cuda(self):
        self.vae_v2.set_cuda()

    def forward_mani(self, enc_objs, enc_triples, enc_boxes, enc_angles, enc_shapes, encoded_enc_text_feat, encoded_enc_rel_feat, attributes, enc_objs_to_scene, dec_objs, dec_objs_grained,
                     dec_triples, dec_boxes, dec_angles, dec_sdfs, dec_shapes, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes, dec_objs_to_scene, missing_nodes,
                     manipulated_nodes):

        if self.type_ == 'v1_full':
            mu, logvar, orig_gt_boxes, orig_gt_angles, orig_gt_shapes, orig_boxes, orig_angles, orig_shapes, boxes, angles, obj_and_shape, keep = \
                self.vae.forward(enc_objs, enc_triples, enc_boxes, enc_angles, enc_shapes, attributes, enc_objs_to_scene,
                                 dec_objs, dec_triples, dec_boxes, dec_angles, dec_shapes, dec_attributes, dec_objs_to_scene,
                                 missing_nodes, manipulated_nodes)

            return mu, logvar, mu, logvar, orig_gt_boxes, orig_gt_angles, orig_gt_shapes, orig_boxes, orig_angles, orig_shapes, boxes, angles, obj_and_shape, keep

        elif self.type_ == 'v1_box':
            mu_boxes, logvar_boxes, orig_gt_boxes, orig_gt_angles, orig_boxes, orig_angles, boxes, angles, keep = self.vae_box.forward(
                enc_objs, enc_triples, enc_boxes, attributes, encoded_enc_text_feat, encoded_enc_rel_feat, enc_objs_to_scene, dec_objs, dec_triples, dec_boxes,
                dec_attributes, dec_objs_to_scene, missing_nodes, manipulated_nodes, enc_angles, dec_angles)

            return mu_boxes, logvar_boxes, None, None, orig_gt_boxes, orig_gt_angles, None, orig_boxes, orig_angles, None, boxes, angles, None, keep

        elif self.type_ == 'v2_box':
            mu_boxes, logvar_boxes, orig_gt_boxes, orig_gt_angles, orig_boxes, orig_angles, boxes, angles, keep = self.vae_box.forward(
                enc_objs, enc_triples, enc_boxes, encoded_enc_text_feat, encoded_enc_rel_feat, attributes, enc_objs_to_scene, dec_objs, dec_triples, dec_boxes,
                encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes, dec_objs_to_scene, missing_nodes, manipulated_nodes, enc_angles, dec_angles)

            return mu_boxes, logvar_boxes, None, None, orig_gt_boxes, orig_gt_angles, None, orig_boxes, orig_angles, None, boxes, angles, None, keep

        elif self.type_ == 'v2_full':
            mu, logvar, orig_gt_boxes, orig_gt_angles, orig_gt_shapes, orig_boxes, orig_angles, boxes, angles, obj_and_shape, keep = self.vae_v2.forward(
                enc_objs, enc_triples, enc_boxes, encoded_enc_text_feat, encoded_enc_rel_feat, attributes, enc_objs_to_scene, dec_objs, dec_objs_grained, dec_triples, dec_boxes,
                encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes, dec_objs_to_scene, missing_nodes, manipulated_nodes, dec_sdfs, enc_angles, dec_angles)

            return mu, logvar, None, None, orig_gt_boxes, orig_gt_angles, orig_gt_shapes, orig_boxes, orig_angles, None, boxes, angles, obj_and_shape, keep

    def load_networks(self, exp, epoch, strict=True, restart_optim=False):
        if self.type_ == 'v1_box':
            self.vae_box.load_state_dict(
                torch.load(os.path.join(exp, 'checkpoint', 'model_box_{}.pth'.format(epoch))),
                strict=strict
            )
        elif self.type_ == 'v1_full':
            print()
            ckpt = torch.load(os.path.join(exp, 'checkpoint', 'model{}.pth'.format(epoch))).state_dict()
            self.vae.load_state_dict(
                ckpt,
                strict=strict
            )
        elif self.type_ == 'v2_box':
            self.vae_box.load_state_dict(
                torch.load(os.path.join(exp, 'checkpoint', 'model_box_{}.pth'.format(epoch))),
                strict=strict
            )
        elif self.type_ == 'v2_full':
            from omegaconf import OmegaConf
            diff_cfg = OmegaConf.load(self.diff_opt)
            ckpt = torch.load(os.path.join(exp, 'checkpoint', 'model{}.pth'.format(epoch)))
            diff_state_dict = {}
            diff_state_dict['vqvae'] = ckpt.pop('vqvae')
            diff_state_dict['df'] = ckpt.pop('df')
            diff_state_dict['opt'] = ckpt.pop('opt')
            try:
                self.epoch = ckpt.pop('epoch')
                self.counter = ckpt.pop('counter')
            except:
                print('no epoch or counter record.')
            self.vae_v2.load_state_dict(
                ckpt,
                strict=strict
            )
            print(colored('[*] v2_box successfully restored from: %s' % os.path.join(exp, 'checkpoint',
                                                                                        'model{}.pth'.format(epoch)),
                          'blue'))
            self.vae_v2.Diff.vqvae.load_state_dict(diff_state_dict['vqvae'])
            self.vae_v2.Diff.df.load_state_dict(diff_state_dict['df'])

            if not restart_optim:
                import torch.optim as optim
                self.vae_v2.optimizerFULL.load_state_dict(diff_state_dict['opt'])
                # self.vae_v2.scheduler = optim.lr_scheduler.StepLR(self.vae_v2.optimizerFULL, 10000, 0.9)
                self.vae_v2.scheduler = optim.lr_scheduler.LambdaLR(self.vae_v2.optimizerFULL, lr_lambda=self.vae_v2.lr_lambda,
                                                        last_epoch=int(self.counter - 1))

            # for multi-gpu (deprecated)
            if diff_cfg.hyper.distributed:
                self.vae_v2.Diff.make_distributed(diff_cfg)
                self.vae_v2.Diff.df_module = self.vae_v2.Diff.df.module
                self.vae_v2.Diff.vqvae_module = self.vae_v2.Diff.vqvae.module
            else:
                self.vae_v2.Diff.df_module = self.vae_v2.Diff.df
                self.vae_v2.Diff.vqvae_module = self.vae_v2.Diff.vqvae
            print(colored('[*] v2_shape successfully restored from: %s' % os.path.join(exp, 'checkpoint', 'model{}.pth'.format(epoch)), 'blue'))

    def compute_statistics(self, exp, epoch, stats_dataloader, force=False):
        box_stats_f = os.path.join(exp, 'checkpoint', 'model_stats_box_{}.pkl'.format(epoch))
        stats_f = os.path.join(exp, 'checkpoint', 'model_stats_{}.pkl'.format(epoch))
        if self.type_ == 'v1_box':
            if os.path.exists(box_stats_f) and not force:
                stats = pickle.load(open(box_stats_f, 'rb'))
                self.mean_est_box, self.cov_est_box = stats[0], stats[1]
            else:
                self.mean_est_box, self.cov_est_box = self.vae_box.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est_box, self.cov_est_box], open(box_stats_f, 'wb'))

        elif self.type_ == 'v1_full':
            if os.path.exists(stats_f) and not force:
                stats = pickle.load(open(stats_f, 'rb'))
                self.mean_est, self.cov_est = stats[0], stats[1]
            else:
                self.mean_est, self.cov_est = self.vae.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est, self.cov_est], open(stats_f, 'wb'))
        elif self.type_ == 'v2_box':
            if os.path.exists(box_stats_f) and not force:
                stats = pickle.load(open(box_stats_f, 'rb'))
                self.mean_est_box, self.cov_est_box = stats[0], stats[1]
            else:
                self.mean_est_box, self.cov_est_box = self.vae_box.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est_box, self.cov_est_box], open(box_stats_f, 'wb'))
        elif self.type_ == 'v2_full':
            if os.path.exists(stats_f) and not force:
                stats = pickle.load(open(stats_f, 'rb'))
                self.mean_est, self.cov_est = stats[0], stats[1]
            else:
                self.mean_est, self.cov_est = self.vae_v2.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est, self.cov_est], open(stats_f, 'wb'))

    def decoder_with_changes_boxes_and_shape(self, z_box, z_shape, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes, box_data=None, gen_shape=False):
        if self.type_ == 'v1_full':
            boxes, feats, keep = self.vae.decoder_with_changes(z_box, objs, triples, attributes, missing_nodes, manipulated_nodes)
            points, _ = self.decode_g2sv1(objs, feats, box_data, retrieval=True)
        elif self.type_ == 'v1_box' or self.type_ == 'v2_box':
            boxes, keep = self.decoder_with_changes_boxes(z_box, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes, manipulated_nodes)
            points = None
        elif self.type_ == 'v2_full':
            boxes, sdfs, keep = self.vae_v2.decoder_with_changes(z_box, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes,
                                                               manipulated_nodes, gen_shape=gen_shape)
            return boxes, sdfs, keep

        return boxes, points, keep

    def decoder_with_changes_boxes(self, z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes, manipulated_nodes):
        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            return self.vae_box.decoder_with_changes(z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes, manipulated_nodes)
        if self.type_ == 'v1_full':
            return None, None

    def decoder_with_changes_shape(self, z, objs, triples, attributes, missing_nodes, manipulated_nodes, atlas):
        if self.type_ == 'v1_full':
            return None, None
    def decoder_boxes_and_shape(self, z_box, z_shape, objs, triples, attributes, atlas=None):
        angles = None
        if self.type_ == 'v1_full':
            boxes, angles, feats = self.vae.decoder(z_box, objs, triples, attributes)
            points = atlas.forward_inference_from_latent_space(feats, atlas.get_grid()) if atlas is not None else feats
        elif self.type_ == 'v1_box':
            boxes, angles = self.decoder_boxes(z_box, objs, triples, attributes)
            points = None
        return boxes, angles, points

    def decoder_boxes(self, z, objs, triples, attributes):
        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            if self.with_angles:
                return self.vae_box.decoder(z, objs, triples, attributes)
            else:
                return self.vae_box.decoder(z, objs, triples, attributes), None
        elif self.type_ == 'v1_full':
            return None, None

    def decoder_with_additions_boxes_and_shape(self, z_box, z_shape, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes,
                                               manipulated_nodes, gen_shape=False):
        if self.type_ == 'v1_full':
            outs, keep = self.vae.decoder_with_additions(z_box, objs, triples, attributes, missing_nodes, manipulated_nodes)
            return outs[:2], None, outs[2], keep
        elif self.type_ == 'v1_box' or self.type_ == 'v2_box':
            boxes, keep = self.decoder_with_additions_boxs(z_box, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes,
                                                                manipulated_nodes)
            return boxes, None, keep
        elif self.type_ == 'v2_full':
            boxes, sdfs, keep = self.vae_v2.decoder_with_additions(z_box, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes,
                                                         manipulated_nodes, gen_shape=gen_shape)
            return boxes, sdfs, keep
        else:
            print("error, no this type")

    def decoder_with_additions_boxs(self, z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes, manipulated_nodes):
        boxes, angles, keep = None, None, None
        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            boxes, keep = self.vae_box.decoder_with_additions(z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes,
                                                            manipulated_nodes, (self.mean_est_box, self.cov_est_box))

        elif self.type_ == 'v1_full':
            return  None, None, None
        return boxes, angles, keep

    def encode_box_and_shape(self, objs, triples, encoded_enc_text_feat, encoded_enc_rel_feat, feats, boxes, angles=None, attributes=None):
        if not self.with_angles:
            angles = None
        if self.type_ == 'v1_box' or self.type_ == 'v2_box' or self.type_ == 'v2_full':
            return self.encode_box(objs, triples, encoded_enc_text_feat, encoded_enc_rel_feat, boxes, angles, attributes), (None, None)
        elif self.type_ == 'v1_full':
            with torch.no_grad():
                z, log_var = self.vae.encoder(objs, triples, boxes, feats, attributes, angles)
                return (z, log_var), (z, log_var)

    def encode_shape(self, objs, triples, feats, attributes=None):
        if self.type_ == 'v1_full':
            return None, None

    def encode_box(self, objs, triples, encoded_enc_text_feat, encoded_enc_rel_feat, boxes, angles=None, attributes=None):

        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            z, log_var = self.vae_box.encoder(objs, triples, boxes, attributes, encoded_enc_text_feat, encoded_enc_rel_feat, angles)
        elif self.type_ == 'v2_full':
            z, log_var = self.vae_v2.encoder(objs, triples, boxes, attributes, encoded_enc_text_feat,
                                              encoded_enc_rel_feat, angles)
        elif self.type_ == 'v1_full':
            return None, None
        return z, log_var

    def sample_box_and_shape(self, point_classes_idx, dec_objs, dec_triplets, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, attributes=None, gen_shape=False):
        if self.type_ == 'v1_full':
            return self.vae.sample_3dfront(point_classes_idx, self.mean_est, self.cov_est, dec_objs, dec_triplets, attributes)
        elif self.type_ == 'v2_full':
            return self.vae_v2.sample(point_classes_idx, self.mean_est, self.cov_est, dec_objs, dec_triplets, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat,
                                   attributes, gen_shape=gen_shape)
        boxes = self.sample_box(dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)
        shapes = self.sample_shape(point_classes_idx, dec_objs, dec_triplets, attributes)
        return boxes, shapes

    def get_closest_vec(self, class_name, shape_vec, box_data):
        import numpy as np

        obj_ids = list(box_data[class_name].keys())
        # names = list(self.code_dict.keys())
        codes = np.vstack([self.code_dict[obj_id] for obj_id in obj_ids])
        mses = np.sum((codes - shape_vec.detach().cpu().numpy()) ** 2, axis=-1)
        id_min = np.argmin(mses)
        return obj_ids[id_min], codes[id_min]

    def decode_g2sv1(self, cats, shape_vecs, box_data, retrieval=False):
        if retrieval:
            vec_list = []
            mesh_list= []
            for (cat, shape_vec) in zip(cats, shape_vecs):
                class_name = self.classes_[cat].strip('\n')
                if class_name == 'floor' or class_name == '_scene_':
                    continue
                name_, vec_ = self.get_closest_vec(class_name, shape_vec, box_data)
                vec_list.append(vec_)
                obj = trimesh.load(os.path.join(self.v1mesh_base,name_,'sdf.ply'))
                mesh_list.append(obj)

        return mesh_list, vec_list

    def sample_box(self, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes=None):
        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            return self.vae_box.sampleBoxes(self.mean_est_box, self.cov_est_box, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)
        elif self.type_ == 'v1_full':
            return self.vae.sample(self.mean_est, self.cov_est, dec_objs, dec_triplets, attributes)[0]
        elif self.type_ == 'v2_full':
            return self.vae_v2.sample(self.mean_est, self.cov_est, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)[0]

    def sample_shape(self, point_classes_idx, dec_objs, dec_triplets, attributes=None):
        if self.type_ == 'v1_full':
            return self.vae.sample(self.mean_est, self.cov_est, dec_objs, dec_triplets, attributes)[1]


    def save(self, exp, outf, epoch, counter=None):
        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            torch.save(self.vae_box.state_dict(), os.path.join(exp, outf, 'model_box_{}.pth'.format(epoch)))
        elif self.type_ == 'v1_full':
            torch.save(self.vae, os.path.join(exp, outf, 'model{}.pth'.format(epoch)))
        elif self.type_ == 'v2_full':
            torch.save(self.vae_v2.state_dict(epoch, counter), os.path.join(exp, outf, 'model{}.pth'.format(epoch)))
