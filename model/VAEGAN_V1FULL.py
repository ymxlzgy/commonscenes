import torch
import torch.nn as nn
import torch.nn.functional as F
from model.graph import GraphTripleConvNet, _init_weights, make_mlp
import numpy as np


class Sg2ScVAEModel(nn.Module):
    """
    VAE-based network for scene generation and manipulation from a scene graph.
    It has a shared embedding of shape and bounding box latents.
    """
    def __init__(self, vocab,
                 embedding_dim=128,
                 batch_size=32,
                 train_3d=True,
                 decoder_cat=False,
                 Nangle=24,
                 gconv_pooling='avg',
                 gconv_num_layers=5,
                 gconv_num_shared_layer=3,
                 distribution_before=True,
                 mlp_normalization='none',
                 vec_noise_dim=0,
                 use_AE=False,
                 with_changes=True,
                 replace_latent=True,
                 use_angles=False,
                 num_box_params=6,
                 residual=False,
                 shape_input_dim=128):

        super(Sg2ScVAEModel, self).__init__()
        self.replace_latent = replace_latent
        self.with_changes = with_changes
        self.dist_before = distribution_before

        self.use_angles = use_angles

        gconv_dim = embedding_dim
        gconv_hidden_dim = gconv_dim * 4

        box_embedding_dim = int(embedding_dim)
        shape_embedding_dim = int(embedding_dim)
        if use_angles:
            angle_embedding_dim = int(embedding_dim / 4)
            box_embedding_dim = int(embedding_dim * 3 / 4)
            Nangle = 24

        obj_embedding_dim = embedding_dim

        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.train_3d = train_3d
        self.decoder_cat = decoder_cat
        self.vocab = vocab
        self.vec_noise_dim = vec_noise_dim
        self.use_AE = use_AE

        num_objs = len(list(set(vocab['object_idx_to_name'])))
        num_preds = len(list(set(vocab['pred_idx_to_name'])))

        # build network components for encoder and decoder
        self.obj_embeddings_ec_box = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.obj_embeddings_ec_shape = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.pred_embeddings_ec_box = nn.Embedding(num_preds, 2 * embedding_dim)
        self.pred_embeddings_ec_shape = nn.Embedding(num_preds, 2 * embedding_dim)

        self.obj_embeddings_dc_box = nn.Embedding(num_objs + 1, 2*obj_embedding_dim)
        self.obj_embeddings_dc_man = nn.Embedding(num_objs + 1, 2*obj_embedding_dim)
        self.obj_embeddings_dc_shape = nn.Embedding(num_objs + 1, 2*obj_embedding_dim)
        self.pred_embeddings_dc_box = nn.Embedding(num_preds, 4*embedding_dim)
        self.pred_embeddings_dc_shape = nn.Embedding(num_preds, 4*embedding_dim)

        if self.decoder_cat:
            self.pred_embeddings_dc = nn.Embedding(num_preds, embedding_dim * 2)
            self.pred_embeddings_man_dc = nn.Embedding(num_preds, embedding_dim * 6)
        if self.train_3d:
            self.box_embeddings = nn.Linear(num_box_params, box_embedding_dim)
            self.shape_embeddings = nn.Linear(shape_input_dim, shape_embedding_dim)
        else:
            self.box_embeddings = nn.Linear(4, box_embedding_dim)
        if self.use_angles:
            self.angle_embeddings = nn.Embedding(Nangle, angle_embedding_dim)
        # weight sharing of mean and var
        self.box_mean_var = make_mlp([embedding_dim * 2, gconv_hidden_dim, embedding_dim * 2],
                                     batch_norm=mlp_normalization)
        self.box_mean = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        self.box_var = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)

        self.shape_mean_var = make_mlp([embedding_dim * 2, gconv_hidden_dim, embedding_dim * 2],
                                     batch_norm=mlp_normalization)
        self.shape_mean = make_mlp([embedding_dim * 2, shape_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        self.shape_var = make_mlp([embedding_dim * 2, shape_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        if self.use_angles:
            self.angle_mean_var = make_mlp([embedding_dim * 2, gconv_hidden_dim, embedding_dim * 2],
                                           batch_norm=mlp_normalization)
            self.angle_mean = make_mlp([embedding_dim * 2, angle_embedding_dim], batch_norm=mlp_normalization, norelu=True)
            self.angle_var = make_mlp([embedding_dim * 2, angle_embedding_dim], batch_norm=mlp_normalization, norelu=True)        # graph conv net

        self.gconv_net_ec = None
        self.gconv_net_dc = None

        gconv_kwargs_ec = {
            'input_dim_obj': gconv_dim * 2,
            'input_dim_pred': gconv_dim * 2,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        gconv_kwargs_dc = {
            'input_dim_obj': gconv_dim * 2,
            'input_dim_pred': gconv_dim * 2,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        gconv_kwargs_shared = {
            'input_dim_obj': gconv_hidden_dim,
            'input_dim_pred': gconv_hidden_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_shared_layer,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        if self.with_changes:
            gconv_kwargs_manipulation = {
                'input_dim_obj': embedding_dim * 6,
                'input_dim_pred': embedding_dim * 6,
                'hidden_dim': gconv_hidden_dim * 2,
                'output_dim': embedding_dim * 2,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers,
                'mlp_normalization': mlp_normalization,
                'residual': residual
            }
        if self.decoder_cat:
            gconv_kwargs_dc['input_dim_obj'] = gconv_dim * 4
            gconv_kwargs_dc['input_dim_pred'] = gconv_dim * 4
        if not self.dist_before:
            gconv_kwargs_shared['input_dim_obj'] = gconv_hidden_dim * 2
            gconv_kwargs_shared['input_dim_pred'] = gconv_hidden_dim * 2

        self.gconv_net_ec_box = GraphTripleConvNet(**gconv_kwargs_ec)
        self.gconv_net_ec_shape = GraphTripleConvNet(**gconv_kwargs_ec)

        self.gconv_net_dec_box = GraphTripleConvNet(**gconv_kwargs_dc)
        self.gconv_net_dec_shape = GraphTripleConvNet(**gconv_kwargs_dc)
        self.gconv_net_shared = GraphTripleConvNet(**gconv_kwargs_shared)

        if self.with_changes:
            self.gconv_net_manipulation = GraphTripleConvNet(**gconv_kwargs_manipulation)

        # box prediction net
        if self.train_3d:
            box_net_dim = num_box_params
        else:
            box_net_dim = 4
        box_net_layers = [gconv_dim * 4, gconv_hidden_dim, box_net_dim]

        self.box_net = make_mlp(box_net_layers, batch_norm=mlp_normalization, norelu=True)

        if self.use_angles:
            # angle prediction net
            angle_net_layers = [gconv_dim * 4, gconv_hidden_dim, Nangle]
            self.angle_net = make_mlp(angle_net_layers, batch_norm=mlp_normalization, norelu=True)
        shape_net_layers = [gconv_dim * 4, gconv_hidden_dim, 256]
        self.shape_net = make_mlp(shape_net_layers, batch_norm=mlp_normalization, norelu=True)

        # initialization
        self.box_embeddings.apply(_init_weights)
        self.box_mean_var.apply(_init_weights)
        self.box_mean.apply(_init_weights)
        self.box_var.apply(_init_weights)
        if self.use_angles:
            self.angle_mean_var.apply(_init_weights)
            self.angle_mean.apply(_init_weights)
            self.angle_var.apply(_init_weights)
        self.shape_mean_var.apply(_init_weights)
        self.shape_mean.apply(_init_weights)
        self.shape_var.apply(_init_weights)
        self.shape_net.apply(_init_weights)
        self.box_net.apply(_init_weights)

    def encoder(self, objs, triples, boxes_gt, shapes_gt, attributes, angles_gt=None):
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs_box = self.obj_embeddings_ec_box(objs)
        obj_vecs_shape = self.obj_embeddings_ec_shape(objs)

        shape_vecs = self.shape_embeddings(shapes_gt)
        pred_vecs_box = self.pred_embeddings_ec_box(p)
        pred_vecs_shape = self.pred_embeddings_ec_shape(p)

        boxes_vecs = self.box_embeddings(boxes_gt)

        if self.use_angles:
            angle_vecs = self.angle_embeddings(angles_gt)
            obj_vecs_box = torch.cat([obj_vecs_box, boxes_vecs, angle_vecs], dim=1)
        else:
            obj_vecs_box = torch.cat([obj_vecs_box, boxes_vecs], dim=1)

        obj_vecs_shape = torch.cat([obj_vecs_shape, shape_vecs], dim=1)

        if self.gconv_net_ec_box is not None:
            obj_vecs_box, pred_vecs_box = self.gconv_net_ec_box(obj_vecs_box, pred_vecs_box, edges)
            obj_vecs_shape, pred_vecs_shapes = self.gconv_net_ec_shape(obj_vecs_shape, pred_vecs_shape, edges)

        if self.dist_before:
            obj_vecs_shared = torch.cat([obj_vecs_box, obj_vecs_shape], dim=1)
            pred_vecs_shared = torch.cat([pred_vecs_box, pred_vecs_shapes], dim=1)
            obj_vecs_shared, pred_vecs_shared =self.gconv_net_shared(obj_vecs_shared, pred_vecs_shared, edges)

            obj_vecs_box, obj_vecs_shape = obj_vecs_shared[:, :obj_vecs_box.shape[1]], obj_vecs_shared[:, obj_vecs_box.shape[1]:]
            obj_vecs_box_norot = self.box_mean_var(obj_vecs_box)
            mu_box = self.box_mean(obj_vecs_box_norot)
            logvar_box = self.box_var(obj_vecs_box_norot)

            if self.use_angles:
                obj_vecs_angle = self.angle_mean_var(obj_vecs_box)
                mu_angle = self.angle_mean(obj_vecs_angle)
                logvar_angle = self.angle_var(obj_vecs_angle)
                mu_box = torch.cat([mu_box, mu_angle], dim=1)
                logvar_box = torch.cat([logvar_box, logvar_angle], dim=1)

            obj_vecs_shape = self.shape_mean_var(obj_vecs_shape)
            mu_shape = self.shape_mean(obj_vecs_shape)
            logvar_shape = self.shape_var(obj_vecs_shape)

        else:
            obj_vecs_box_norot = self.box_mean_var(obj_vecs_box)
            mu_box = self.box_mean(obj_vecs_box_norot)
            logvar_box = self.box_var(obj_vecs_box_norot)

            if self.use_angles:
                obj_vecs_angle = self.angle_mean_var(obj_vecs_box)
                mu_angle = self.angle_mean(obj_vecs_angle)
                logvar_angle = self.angle_var(obj_vecs_angle)
                mu_box = torch.cat([mu_box, mu_angle], dim=1)
                logvar_box = torch.cat([logvar_box, logvar_angle], dim=1)

            obj_vecs_shape = self.shape_mean_var(obj_vecs_shape)
            mu_shape = self.shape_mean(obj_vecs_shape)
            logvar_shape = self.shape_var(obj_vecs_shape)

        mu = torch.cat([mu_box, mu_shape], dim=1)
        logvar = torch.cat([logvar_box, logvar_shape], dim=1)
        return mu, logvar

    def manipulate(self, z, objs, triples, attributes):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_dc_man(objs)
        pred_vecs = self.pred_embeddings_man_dc(p)

        man_z = torch.cat([z, obj_vecs], dim=1)
        man_z, _ = self.gconv_net_manipulation(man_z, pred_vecs, edges)

        return man_z

    def decoder(self, z, objs, triples, attributes, manipulate=False):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs_box = self.obj_embeddings_dc_box(objs)
        obj_vecs_shape = self.obj_embeddings_dc_shape(objs)
        pred_vecs_box = self.pred_embeddings_dc_box(p)
        pred_vecs_shape = self.pred_embeddings_dc_shape(p)
        if not self.dist_before:
            obj_vecs_box = torch.cat([obj_vecs_box, z], dim=1)
            obj_vecs_shape = torch.cat([obj_vecs_shape, z], dim=1)
            obj_vecs_shared = torch.cat([obj_vecs_box, obj_vecs_shape], dim=1)
            pred_vecs_shared = torch.cat([pred_vecs_box, pred_vecs_shape], dim=1)

            obj_vecs_shared, pred_vecs_shared = self.gconv_net_shared(obj_vecs_shared, pred_vecs_shared, edges)
            obj_vecs_box, obj_vecs_shape = obj_vecs_shared[:, :obj_vecs_box.shape[1]], obj_vecs_shared[:, obj_vecs_box.shape[1]:]
            pred_vecs_box, pred_vecs_shape = pred_vecs_shared[:, :pred_vecs_box.shape[1]], pred_vecs_shared[:, pred_vecs_box.shape[1]:]

        if self.decoder_cat:
            if self.dist_before:
                obj_vecs_box = torch.cat([obj_vecs_box, z], dim=1)
                obj_vecs_shape = torch.cat([obj_vecs_shape, z], dim=1)

            obj_vecs_box, pred_vecs_box = self.gconv_net_dec_box(obj_vecs_box, pred_vecs_box, edges)
            obj_vecs_shape, pred_vecs_shape = self.gconv_net_dec_shape(obj_vecs_shape, pred_vecs_shape, edges)
        else:
            raise NotImplementedError

        boxes_pred = self.box_net(obj_vecs_box)
        shapes_pred = self.shape_net(obj_vecs_shape)

        if self.use_angles:
            angles_pred = F.log_softmax(self.angle_net(obj_vecs_box), dim=1)
            return boxes_pred, angles_pred, shapes_pred
        else:
            return boxes_pred, shapes_pred

    def decoder_with_changes(self, z, dec_objs, dec_triples, attributes, missing_nodes, manipulated_nodes):
        # append zero nodes
        nodes_added = []
        for i in range(len(missing_nodes)):
          ad_id = missing_nodes[i] + i
          nodes_added.append(ad_id)
          noise = np.zeros(self.embedding_dim* 2) # np.random.normal(0, 1, 64)
          zeros = torch.from_numpy(noise.reshape(1, self.embedding_dim* 2))
          zeros.requires_grad = True
          zeros = zeros.float().cuda()
          z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)

        # mark changes in nodes
        change_repr = []
        for i in range(len(z)):
            if i not in nodes_added and i not in manipulated_nodes:
                noisechange = np.zeros(self.embedding_dim* 2)
            else:
                noisechange = np.random.normal(0, 1, self.embedding_dim* 2)
            change_repr.append(torch.from_numpy(noisechange).float().cuda())
        change_repr = torch.stack(change_repr, dim=0)
        z_prime = torch.cat([z, change_repr], dim=1)
        z_prime = self.manipulate(z_prime, dec_objs, dec_triples, attributes)

        # take original nodes when untouched

        if self.replace_latent:
            # take original nodes when untouched
            touched_nodes = torch.tensor(sorted(nodes_added + manipulated_nodes)).long()
            for touched_node in touched_nodes:
                z = torch.cat([z[:touched_node], z_prime[touched_node:touched_node + 1], z[touched_node + 1:]], dim=0)
        else:
            z = z_prime

        dec_man_enc_boxes_pred = self.decoder(z, dec_objs, dec_triples, attributes)
        if self.use_angles:
            dec_man_enc_boxes_pred, dec_man_enc_shapes_pred = dec_man_enc_boxes_pred[:-1], dec_man_enc_boxes_pred[-1]
            num_dec_objs = len(dec_man_enc_boxes_pred[0])
        else:
            num_dec_objs = len(dec_man_enc_boxes_pred)

        keep = []
        for i in range(num_dec_objs):
          if i not in nodes_added and i not in manipulated_nodes:
            keep.append(1)
          else:
            keep.append(0)

        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()

        return dec_man_enc_boxes_pred, dec_man_enc_shapes_pred, keep

    def decoder_with_additions(self, z, objs, triples, attributes, missing_nodes, manipulated_nodes, distribution=None):
        nodes_added = []
        if distribution is not None:
            mu, cov = distribution

        for i in range(len(missing_nodes)):
            ad_id = missing_nodes[i] + i
            nodes_added.append(ad_id)
            noise = np.zeros(z.shape[1])
            if distribution is not None:
                zeros = torch.from_numpy(np.random.multivariate_normal(mu, cov, 1)).float().cuda()
            else:
                zeros = torch.from_numpy(noise.reshape(1, z.shape[1]))
            zeros.requires_grad = True
            zeros = zeros.float().cuda()
            z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)

        keep = []
        for i in range(len(z)):
            if i not in nodes_added and i not in manipulated_nodes:
                keep.append(1)
            else:
                keep.append(0)

        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()

        return self.decoder(z, objs, triples, attributes), keep

    def forward(self, enc_objs, enc_triples, enc_boxes, enc_angles, enc_shapes, attributes, enc_objs_to_scene, dec_objs,
                dec_triples, dec_boxes, dec_angles, dec_shapes, dec_attributes, dec_objs_to_scene, missing_nodes, manipulated_nodes):

        mu, logvar = self.encoder(enc_objs, enc_triples, enc_boxes, enc_shapes, attributes, enc_angles)

        if self.use_AE:
            z = mu
        else:
            # reparameterization
            std = torch.exp(0.5*logvar)
            # standard sampling
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)

        if self.with_changes:
            # append zero nodes
            nodes_added = []
            for i in range(len(missing_nodes)):
              ad_id = missing_nodes[i] + i
              nodes_added.append(ad_id)
              noise = np.zeros(self.embedding_dim * 2) # np.random.normal(0, 1, 64)
              zeros = torch.from_numpy(noise.reshape(1, self.embedding_dim * 2))
              zeros.requires_grad = True
              zeros = zeros.float().cuda()
              z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)

            # mark changes in nodes
            change_repr = []
            for i in range(len(z)):
                if i not in nodes_added and i not in manipulated_nodes:
                    noisechange = np.zeros(self.embedding_dim * 2)
                else:
                    noisechange = np.random.normal(0, 1, self.embedding_dim * 2)
                change_repr.append(torch.from_numpy(noisechange).float().cuda())
            change_repr = torch.stack(change_repr, dim=0)
            z_prime = torch.cat([z, change_repr], dim=1)
            z_prime = self.manipulate(z_prime, dec_objs, dec_triples, attributes)
            if self.replace_latent:
                # take original nodes when untouched
                touched_nodes = torch.tensor(sorted(nodes_added + manipulated_nodes)).long()
                for touched_node in touched_nodes:
                    z = torch.cat([z[:touched_node], z_prime[touched_node:touched_node + 1], z[touched_node + 1:]],
                                  dim=0)
            else:
                z = z_prime


        if self.use_angles:
            dec_man_enc_boxes_pred, angles_pred, dec_man_enc_shapes_pred = self.decoder(z, dec_objs, dec_triples, attributes)
            orig_angles = []
            orig_gt_angles = []
        else:
            dec_man_enc_boxes_pred, dec_man_enc_shapes_pred = self.decoder(z, dec_objs, dec_triples, attributes)

        if not self.with_changes:
            return mu, logvar, dec_man_enc_boxes_pred, dec_man_enc_shapes_pred
        else:
            orig_boxes = []
            orig_shapes = []
            orig_gt_boxes = []
            orig_gt_shapes = []
            keep = []
            # keep becomes 1 in the unchanged nodes and 0 otherwise
            for i in range(len(dec_man_enc_boxes_pred)):
              if i not in nodes_added and i not in manipulated_nodes:
                orig_boxes.append(dec_man_enc_boxes_pred[i:i+1])
                orig_shapes.append(dec_man_enc_shapes_pred[i:i+1])
                orig_gt_boxes.append(dec_boxes[i:i+1])
                orig_gt_shapes.append(dec_shapes[i:i+1])
                if self.use_angles:
                    orig_angles.append(angles_pred[i:i+1])
                    orig_gt_angles.append(dec_angles[i:i+1])
                keep.append(1)
              else:
                keep.append(0)

            orig_boxes = torch.cat(orig_boxes, dim=0)
            orig_shapes = torch.cat(orig_shapes, dim=0)
            orig_gt_boxes = torch.cat(orig_gt_boxes, dim=0)
            orig_gt_shapes = torch.cat(orig_gt_shapes, dim=0)
            keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()
            if self.use_angles:
                orig_angles = torch.cat(orig_angles, dim=0)
                orig_gt_angles = torch.cat(orig_gt_angles, dim=0)
            else:
                orig_angles, orig_gt_angles, angles_pred = None, None, None

            return mu, logvar, orig_gt_boxes, orig_gt_angles, orig_gt_shapes, orig_boxes, orig_angles, \
                   orig_shapes, dec_man_enc_boxes_pred, angles_pred, [dec_objs, dec_man_enc_shapes_pred], keep

    def sample(self, point_classes_idx, point_ae, mean_est, cov_est, dec_objs,  dec_triples, attributes=None):
        with torch.no_grad():
            z = torch.from_numpy(
                np.random.multivariate_normal(mean_est, cov_est, dec_objs.size(0))).float().cuda()
            samples = self.decoder(z, dec_objs, dec_triples, attributes)
            points = point_ae.forward_inference_from_latent_space(samples[-1], point_ae.get_grid())
            return samples[:-1], (points, samples[-1])

    def sample_3dfront(self, point_classes_idx, mean_est, cov_est, dec_objs,  dec_triples, attributes=None):
        with torch.no_grad():
            z = torch.from_numpy(
                np.random.multivariate_normal(mean_est, cov_est, dec_objs.size(0))).float().cuda()
            samples = self.decoder(z, dec_objs, dec_triples, attributes)
            return samples[:-1], samples[-1]

    def collect_train_statistics(self, train_loader):
        # model = model.eval()
        mean_cat = None

        for idx, data in enumerate(train_loader):
            if data == -1:
                continue

            try:
                dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene = data['decoder']['objs'], \
                                                                                                  data['decoder'][
                                                                                                      'tripltes'], \
                                                                                                  data['decoder']['boxes'], \
                                                                                                  data['decoder'][
                                                                                                      'obj_to_scene'], \
                                                                                                  data['decoder'][
                                                                                                      'triple_to_scene']

                encoded_dec_points = data['decoder']['feats']
                encoded_dec_points = encoded_dec_points.cuda()

            except Exception as e:
                print('Exception', str(e))
                continue

            dec_objs, dec_triples, dec_tight_boxes = dec_objs.cuda(), dec_triples.cuda(), dec_tight_boxes.cuda()
            dec_boxes = dec_tight_boxes[:, :6]
            angles = dec_tight_boxes[:, 6].long() - 1
            angles = torch.where(angles > 0, angles, torch.zeros_like(angles))
            attributes = None

            mean, logvar = self.encoder(dec_objs, dec_triples, dec_boxes, encoded_dec_points, attributes, angles)
            mean, logvar = mean, logvar

            mean = mean.data.cpu().clone()
            if mean_cat is None:
                    mean_cat = mean.numpy()
            else:
                mean_cat = np.concatenate([mean_cat, mean.numpy()], axis=0)

        mean_est = np.mean(mean_cat, axis=0, keepdims=True)  # size 1*embed_dim
        mean_cat = mean_cat - mean_est
        n = mean_cat.shape[0]
        d = mean_cat.shape[1]
        cov_est = np.zeros((d, d))
        for i in range(n):
            x = mean_cat[i]
            cov_est += 1.0 / (n - 1.0) * np.outer(x, x)
        mean_est = mean_est[0]

        return mean_est, cov_est
