from traitlets import dlink
from models.texture import MultiTexture
from models.smpl_body import *
from models.offset_mlp import OffsetMLP
from models.texture_mlp import NormalEncoder, TextureMLP
from models.pose_mlp import RefineMLP
from optimization.criterions import LeakyHingeLoss, MaskedCriterion, poseCriterion
from optimization.perceptual_loss import ResNetLOSS
from optimization.holefilling_segmentation_loss import calc_holefilling_segmentation_loss
# from data.real import digitize_segmap
from util.render import (
    normalize_image_points,
    batch_project,
    create_camera_objects,
    hard_feature_blend,
    render_shaded_mesh,
    create_renderer
)
from util.general import (
    fill_tensor_background,
    seperated_gaussian_blur,
    masked_gaussian_blur,
    dict_2_device,
    DecayScheduler,
    erode_mask,
    softmask_gradient,
    IoU,
    NoSubmoduleWrapper,
    stack_dicts,
)
from util.screen_grad import screen_grad
from util.log import get_logger
from util.lbs import batch_rodrigues

from pytorch3d.renderer import TexturesVertex, rasterize_meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import *

from pathlib import Path
from copy import copy
from argparse import ArgumentParser
from torchvision.transforms.functional import gaussian_blur
from torch.utils.data.dataloader import DataLoader

import shutil
import warnings
import torch
import torchvision
import pytorch_lightning as pl
import json
import numpy as np

from util.meshes import vertex_normals


logger = get_logger(__name__)

class MDGOptimizer(pl.LightningModule):
    """
    Main Class for Optimizing Neural Head Avatars from RGB sequences.
    """
    @staticmethod
    def add_argparse_args(parser):
        parser = ArgumentParser(parents=[parser], add_help=False)

        # specific arguments for combined module

        combi_args = [
            # texture settings
            dict(name_or_flags="--texture_hidden_feats", default=256, type=int),
            dict(name_or_flags="--texture_hidden_layers", default=8, type=int),
            dict(name_or_flags="--texture_d_hidden_dynamic", type=int, default=128),
            dict(name_or_flags="--texture_n_hidden_dynamic", type=int, default=1),
            dict(name_or_flags="--glob_rot_noise", type=float, default=5.0),
            dict(name_or_flags="--d_normal_encoding", type=int, default=32),
            dict(name_or_flags="--d_normal_encoding_hidden", type=int, default=128),
            dict(name_or_flags="--n_normal_encoding_hidden", type=int, default=2),
            dict(name_or_flags="--smpl_noise", type=float, default=0.0),
            dict(name_or_flags="--soft_clip_sigma", type=float, default=-1.0),

            # geometry refinement mlp settings
            dict(name_or_flags="--offset_hidden_layers", default=8, type=int),
            dict(name_or_flags="--offset_hidden_feats", default=256, type=int),

            # smpl_body settings
            dict(name_or_flags="--subdivide_mesh", type=int, default=1),
            dict(name_or_flags="--semantics_blur", default=3, type=int, required=False),
            dict(name_or_flags="--spatial_blur_sigma", type=float, default=0.01),

            # training timeline settings
            dict(name_or_flags="--epochs_offset", type=int, default=50,
                 help="Until which epoch to train smpl_body parameters and offsets jointly"),
            dict(name_or_flags="--epochs_texture", type=int, default=500,
                 help="Until which epoch to train texture while keeping model fixed"),
            dict(name_or_flags="--epochs_joint", type=int, default=500,
                 help="Until which epoch to train model jointly while keeping model fixed"),
            dict(name_or_flags="--image_log_period", type=int, default=10),

            # lr settings
            dict(name_or_flags="--smpl_lr", default=0.005, type=float, nargs=3),
            dict(name_or_flags="--offset_lr", default=0.005, type=float, nargs=3),
            dict(name_or_flags="--tex_lr", default=0.01, type=float, nargs=3),

            # loss weights
            dict(name_or_flags="--body_part_weights", type=str, required=True),
            dict(name_or_flags="--w_rgb", type=float, default=1, nargs=3),
            dict(name_or_flags="--w_perc", default=0, type=float, nargs=3),
            dict(name_or_flags="--w_edge", default=1, type=float, nargs=3),
            dict(name_or_flags="--w_norm", default=1, type=float, nargs=3),
            dict(name_or_flags="--w_lap", type=json.loads, nargs="*"),
            dict(name_or_flags="--w_shape_reg", default=1e-4, type=float, nargs=3),
            dict(name_or_flags="--w_expr_reg", default=1e-4, type=float, nargs=3),
            dict(name_or_flags="--w_pose_reg", default=1e-4, type=float, nargs=3),
            dict(name_or_flags="--w_surface_reg", default=1e-4, type=float, nargs=3),
            dict(name_or_flags="--texture_weight_decay", type=float, default=5e-6, nargs=3),
            dict(name_or_flags="--w_silh", type=json.loads, nargs="*"),
            

        ]
        for f in combi_args:
            parser.add_argument(f.pop("name_or_flags"), **f)

        return parser


    def __init__(self, max_frame_id, w_lap, w_silh, body_part_weights, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.callbacks = [pl.callbacks.ModelCheckpoint(filename="{epoch:02d}", save_last=True)]

      
        self._smpl = SMPLBody(
            SMPL_N_SHAPE,
            SMPL_N_POSE,
            smpl_model_path=SMPL_MODEL_PATH,
            smpl_template_mesh_path=SMPL_MESH_PATH,
            smpl_parts_path=SMPL_PARTS_PATH,
            spatial_blur_sigma=self.hparams["spatial_blur_sigma"],
        )

        # self._shape = torch.nn.Parameter(torch.zeros(4707, SMPL_N_SHAPE), requires_grad=True)
        # self._pose = torch.nn.Parameter(torch.zeros(523, SMPL_N_POSE), requires_grad=True)
        # self._translation = torch.nn.Parameter(torch.zeros(523, SMPL_N_TRANS), requires_grad=True)



        # restrict offsets to anything but eyeballs
        indices = np.arange(len(self._smpl._v_template))
        self._offset_indices = indices
        self._offset_face_indices = self._smpl._faces
        self._vert_feats = torch.nn.Parameter(torch.zeros(1, len(self._offset_indices), 32), requires_grad=True)
        self._vert_feats.data.normal_(0.0, 0.02)

        cond_feats = 72
        in_feats = 3 + self._vert_feats.shape[-1]
        # in_feats = 3

        self._offset_mlp = OffsetMLP(
            in_feats=in_feats,
            cond_feats=cond_feats,
            hidden_feats=self.hparams["offset_hidden_feats"],
            hidden_layers=self.hparams["offset_hidden_layers"],
        )

        # appearance
        self._normal_encoder = NormalEncoder(
            input_dim=3,
            output_dim=self.hparams["d_normal_encoding"],
            hidden_layers_feature_size=self.hparams["d_normal_encoding_hidden"],
            hidden_layers=self.hparams["n_normal_encoding_hidden"],
        )

        teeth_res = 64
        face_res = 256
        d_feat = 64
        self._texture = TextureMLP(
            d_input=3 + d_feat,
            hidden_layers=self.hparams["texture_hidden_layers"],
            hidden_features=self.hparams["texture_hidden_feats"],
            d_dynamic= 72,
            d_hidden_dynamic_head=self.hparams["texture_d_hidden_dynamic"],
            n_dynamic_head=self.hparams["texture_n_hidden_dynamic"],
            d_dynamic_head=self.hparams["d_normal_encoding"],
        )

        pose_in_feats = 72
        pose_cond_feats = 72

        self._pose_mlp= RefineMLP(
            in_feats=pose_in_feats,
            cond_feats=pose_cond_feats,
            hidden_feats=128,
            hidden_layers=6,
            output_dim = pose_in_feats
        )


        self._explFeatures = MultiTexture(
            torch.randn(d_feat, face_res, face_res) * 0.01,
            torch.randn(d_feat, teeth_res, teeth_res) * 0.01,
        )

        self._decays = {
            "lap": DecayScheduler(*w_lap, geometric=True),
            "silh": DecayScheduler(*w_silh, geometric=False),
        }

        # body part weights
        with open(body_part_weights, "r") as f:
            key = "mlp"
            self._body_part_loss_weights = json.load(f)[key]
        # self.semantic_labels = list(self._smpl.get_body_parts().keys())

        # loss definitions
        self._leaky_hinge = LeakyHingeLoss(0.0, 1.0, 0.3)
        self._masked_L1 = MaskedCriterion(torch.nn.L1Loss(reduction="none"))
        self._masked_L1_mask = MaskedCriterion(torch.nn.L1Loss(reduction="none"))
        # self._L1 = NoMaskedCriterion(torch.nn.L1Loss(reduction="none"))
        self._loss_pose  = poseCriterion(torch.nn.MSELoss(reduction='none'))
        # self._loss_pose = torch.nn.MSELoss(reduction='none')
        

        if Path("assets/InsightFace/backbone.pth").exists():
            self._perceptual_loss = NoSubmoduleWrapper(ResNetLOSS())  # don't store perc_loss weights as model weights
        else:
            self._perceptual_loss = None

        # training stage
        self.fit_residuals = False
        self.is_train = False

        # learning rate for smpl_body translation
        self._trans_lr = [0.1 * i for i in self.hparams["smpl_lr"]]

        self._semantic_thr = 0.99
        # self._blurred_vertex_labels = self._smpl.spatially_blurred_vert_labels


        # limits for pose and expression conditioning
        self.register_buffer("shape_min", torch.zeros(10) - 100)
        self.register_buffer("shape_max", torch.zeros(10) + 100)
        self.register_buffer("pose_min", torch.zeros(72) - 100)
        self.register_buffer("pose_max", torch.zeros(72) + 100)
        self.register_buffer("translation_min", torch.zeros(3) - 100)
        self.register_buffer("translation_max", torch.zeros(3) + 100)


    def on_train_start(self) -> None:

        # Copying config and body part weights to checkpoint dir
        logdir = Path(self.trainer.log_dir)
        conf_path = logdir / "config.ini"

        if not conf_path.exists():
            shutil.copy(self.hparams["config"], conf_path, follow_symlinks=True)

        # moves perceptual loss to own device
        try:
            self._perceptual_loss.to(self.device)
        except AttributeError:
            raise AttributeError("You have to download the backbone weights for the perceptual loss. Please refer"
                                 "to the 'Installation' section of the README")

        # hard setting lr
        # smpl_body_optim, offset_optim, tex_optim, joint_smpl_body_optim = self.optimizers()
        offset_optim, tex_optim = self.optimizers()

        lrs = self.get_current_lrs_n_lossweights()

        # smpl_body_optim.param_groups[0]["lr"] = lrs["smpl_lr"]
        # smpl_body_optim.param_groups[1]["lr"] = lrs["trans_lr"]

        for pg in offset_optim.param_groups:
            pg["lr"] = lrs["offset_lr"]

        # joint_smpl_body_optim.param_groups[0]["lr"] = lrs["smpl_lr"]
        # joint_smpl_body_optim.param_groups[1]["lr"] = lrs["trans_lr"]

        for pg in tex_optim.param_groups:
            pg["lr"] = lrs["tex_lr"]
            pg["weight_decay"] = lrs["texture_weight_decay"]
            
    
    def on_train_end(self):
        # determining dynamic condition extrema for validation
        # self.get_dyn_cond_extrema(self.trainer.train_dataloader.dataset.datasets)
        print("training is ending")



    @torch.no_grad()
    def get_dyn_cond_extrema(self, dataset=None):
        dataset = dataset if dataset is not None else self.trainer.train_dataloader.dataset.datasets
        dataloader = DataLoader(dataset, batch_size=48, num_workers=4)

        for i, batch in enumerate(dataloader):
            batch = dict_2_device(batch, self.device)
            smpl_params_offsets = self._create_smpl_param_batch(batch, ignore_offsets=True)
            
            shape = smpl_params_offsets["shape"]
            pose = smpl_params_offsets["pose"]
            translation = smpl_params_offsets["translation"]
        
            if i == 0:
                self.shape_min = torch.min(shape, dim=0).values
                self.shape_max = torch.max(shape, dim=0).values
                self.pose_min = torch.min(pose, dim=0).values
                self.pose_max = torch.max(pose, dim=0).values
                self.translation_min = torch.min(translation, dim=0).values
                self.translation_max = torch.max(translation, dim=0).values
            else:
                self.shape_min = torch.min(self.shape_min, torch.min(shape, dim=0).values)
                self.shape_max = torch.max(self.shape_max, torch.max(shape, dim=0).values)
                self.pose_min = torch.min(self.pose_min, torch.min(pose, dim=0).values)
                self.pose_max = torch.max(self.pose_max, torch.max(pose, dim=0).values)
                self.translation_min = torch.min(self.translation_min, torch.min(translation, dim=0).values)
                self.translation_max = torch.max(self.translation_max, torch.max(translation, dim=0).values)

        return (
            (self.shape_min, self.shape_max),
            (self.pose_min, self.pose_max),
            (self.translation_min, self.translation_max),
        )

    
    
    
    def _get_current_optimizer(self, epoch=None):
        if epoch is None:
            epoch = self.current_epoch
 
        # smpl_body_optim, offset_optim, tex_optim, joint_smpl_body_optim = self.optimizers()
        offset_optim, tex_optim = self.optimizers()

        if epoch < self.hparams["epochs_offset"]:
            optim = [offset_optim]
        elif epoch < self.hparams["epochs_texture"] + self.hparams["epochs_offset"]:
            optim = [tex_optim]
        else:
            optim = [offset_optim, tex_optim]


        return optim




    def _predict_offsets(self, pose, vertices):

        # conditions = pose
        # mlp_out = self._offset_mlp(vertices, conditions)
        # final_offsets = mlp_out

        batch_size = len(pose)
        conditions = pose
        vert_feats = self._vert_feats.expand(batch_size, -1, -1)
        v_temp = self._smpl._v_template_normed.expand(batch_size, -1, -1)
        # v_temp = vertices
        x = torch.cat([v_temp, vert_feats], dim= -1)

        mlp_out = self._offset_mlp(x, conditions)
        final_offsets = mlp_out


        return final_offsets


    # def _predict_offsets(self, pose, vertices):
    
    #     # conditions = pose
    #     # mlp_out = self._offset_mlp(vertices, conditions)
    #     # final_offsets = mlp_out

    #     batch_size = len(pose)
    #     conditions = pose
    #     vert_feats = self._vert_feats.expand(batch_size, -1, -1)

    #     x = torch.cat([vertices, vert_feats], dim= -1)
    #     mlp_out = self._offset_mlp(x, conditions)
    #     final_offsets = mlp_out


    #     return final_offsets
    

    def _create_smpl_param_batch(self, batch, vertices=None, ignore_shape=False, ignore_expr=False, ignore_pose=False,
                                  ignore_offsets=False):


        N = len(iter(batch.values()).__next__())
        indices = batch.get("frame", None)

        p = {}

        # p["shape"] = self._shape[indices]
        # p["translation"] = self._translation[indices]s

        # self._pose[indices] = 0

        
        # temp = self._pose_mlp(self._pose[indices], batch["pose"])
        # temp  = temp.squeeze(1)
        # p["pose"] = batch["pose"].squeeze(1)
        
        # p["pose"] = self._pose[indices]
        # p["pose"] = temp
        # p["shape"] = 0
        # p["translation"] = 0
        # p["pose"] = 0
        
 
        # # adding parameters from dataset
        # p["shape"] = p["shape"] + batch["shape"]
        p["pose"] = batch["pose"]
        # p["translation"] = p["translation"] + batch["smpl_trans"]


        if ignore_offsets:
            p["offsets"] = None
        else:
            p["offsets"] = self._predict_offsets(p["pose"], vertices)


        return p

    def _forward_smpl(self, smpl_params):
        verts = self._smpl(
            **smpl_params,
            zero_centered=True,
            use_rotation_limits=True,
            return_landmarks="static",
        )
        
        return verts


    def _rasterize(self, meshes, cameras, image_size, center_prediction=False):

        assert len(meshes) == len(cameras) 
        
        eps = None
        
        verts_world = meshes.verts_padded()
        verts_view = cameras.get_world_to_view_transform().transform_points(verts_world, eps=eps)   
        projection_trafo = cameras.get_projection_transform().compose(cameras.get_ndc_camera_transform()) ## extri


        verts_ndc = projection_trafo.transform_points(verts_view, eps=eps)
        verts_ndc[..., 2] = verts_view[..., 2]
        meshes_ndc = meshes.update_padded(new_verts_padded=verts_ndc)
        verts_packed = meshes_ndc.verts_packed()
        faces_packed = meshes_ndc.faces_packed()
        face_verts = verts_packed[faces_packed]
        
        
        perspective_correct = cameras.is_perspective()
        znear = cameras.get_znear()
        if isinstance(znear, torch.Tensor):
            znear = znear.min().item()
        z_clip = None if not perspective_correct or znear is None else znear / 2

        
        # verts_packed = meshes.verts_packed()
        # faces_packed = meshes.faces_packed()
        # face_verts = verts_packed[faces_packed]

  
        with torch.no_grad():
            fragments = rasterize_meshes(
                meshes_ndc, #meshes
                image_size=image_size,
                blur_radius=0,
                faces_per_pixel=2,
                bin_size=-1,
                max_faces_per_bin=None,
                clip_barycentric_coords=False,
                perspective_correct=perspective_correct,
                cull_backfaces=False,
                z_clip_value=z_clip,
                cull_to_frustum=False,
            )

            pix_to_face = fragments[0][..., [1]]
            zbuf = fragments[1][..., [1]]
            bary_coords = fragments[2][..., [1], :]
            dists = fragments[3][..., [1]]

            fragments = Fragments(pix_to_face=pix_to_face, zbuf=zbuf, bary_coords=bary_coords, dists=dists)

        pix2face, bary_coords = fragments.pix_to_face, fragments.bary_coords
        is_visible = pix2face[..., 0] > -1

        # shape (sum(is_visible), 3, 3)
        visible_faces = pix2face[is_visible][:, 0]
        visible_face_verts = face_verts[visible_faces]
        # shape (sum(is_visible), 3, 1)
        visible_bary_coords = bary_coords[is_visible][:, 0][..., None]

        visible_surface_point = visible_face_verts * visible_bary_coords
        visible_surface_point = visible_surface_point.sum(dim=1)

        screen_coords = torch.zeros(*pix2face.shape[:3], 2).to(self.device)
        screen_coords[is_visible] = visible_surface_point[:, :2]  # now have gradient

        # if images are not-squared we need to adjust the screen coordinates
        # by the aspect ratio => coords given as [-1,1] for shorter edge and
        # [-s,s] for longer edge where s is the aspect ratio
        H, W = image_size
        if H > W:
            s = H / W
            screen_coords[..., 1] *= 1 / s
        elif H < W:
            s = W / H
            screen_coords[..., 0] *= 1 / s

        return fragments, screen_coords



    def _rasterize_smpl(self, batch, verts):


        H, W = batch["rgb"].shape[-2:]
        batch_size = batch["rgb"].shape[0]
        cam_parmas = batch["smpl_trans"]

        manual_cam = batch["manual_rendering"]
        ### from SMPL parameters(PARE)
        # verts = verts*cam_parmas[:,0:1].view(verts.shape[0],1,1)
        # verts[:,:,0:1]=verts[:,:,0:1]+cam_parmas[:,1:2].view(verts.shape[0],1,1)*cam_parmas[:,0:1].view(verts.shape[0],1,1)
        # verts[:,:,1:2]=verts[:,:,1:2]+cam_parmas[:,2:3].view(verts.shape[0],1,1)*cam_parmas[:,0:1].view(verts.shape[0],1,1)
        # verts[:,:,2:3]=verts[:,:,2:3]*(-1) 
        # verts[:,:,1:2]=verts[:,:,1:2]*(-1)        



        #### from SMPL G.T vertices
        render_verts = verts 
        # render_verts = verts.clone()
        # cam_intri= torch.tensor(batch["cam_intrinsic"]).float().repeat(verts.shape[0],1,1).to(self.device)
        # cam_r = torch.tensor(batch["camera_rotation"]).float().repeat(verts.shape[0],1,1).to(self.device)
        # cam_t = torch.tensor(batch["camera_translation"]).float().repeat(verts.shape[0],1).to(self.device)

        cam_intri= torch.tensor(batch["cam_intrinsic"]).float().to(self.device)
        cam_r = torch.tensor(batch["camera_rotation"]).float().to(self.device)
        cam_t = torch.tensor(batch["camera_translation"]).float().to(self.device)
        
        # cam_transpose = cam_transpose.repeat(verts.shape[0],1,1)
        cam_transpose = cam_t.unsqueeze(1).permute(0,2,1).to(self.device)

        ho = torch.ones((render_verts.shape[0], render_verts.shape[1],1)).float().to(self.device)
        homo_vertices = torch.cat((render_verts, ho), dim=2)
        
        mat = torch.cat((cam_r, cam_transpose), dim=2)
        ze = torch.Tensor([0,0,0,1]).float().to(self.device)
        ze = ze.unsqueeze(0)
        ze = ze.repeat(verts.shape[0],1, 1)

        homo_matrix = torch.cat((mat, ze), dim=1)
        mat_vertices = torch.matmul(homo_matrix, homo_vertices.permute(0,2,1)).permute(0,2,1)
        mat_vertices = mat_vertices[:,:,:3] / mat_vertices[:,:,-1].unsqueeze(2)
        aligned_vertices = torch.matmul(cam_intri, mat_vertices.permute(0,2,1)).permute(0,2,1)


        # [331, 990, 601, 1260]

        # final_vertices = aligned_vertices.clone()
        final_vertices = aligned_vertices.clone()
        final_vertices[:,:,:2]  =  aligned_vertices[:,:,:2] / aligned_vertices[:,:,-1].unsqueeze(2) 
        final_vertices[:,:,0] = final_vertices[:,:,0] - manual_cam[:,2].unsqueeze(1).repeat(1, final_vertices[:,:,0].shape[1])
        final_vertices[:,:,1] = final_vertices[:,:,1] - manual_cam[:,0].unsqueeze(1).repeat(1, final_vertices[:,:,0].shape[1])

        scaler = 512/(manual_cam[:,1]-manual_cam[:,0])
        final_vertices[:,:,0:2] = final_vertices[:,:,0:2] * scaler.unsqueeze(1).unsqueeze(2).repeat(1, final_vertices[:,:,0:2].shape[1], final_vertices[:,:,0:2].shape[2] )

        no_inplace = final_vertices.clone()
        max = no_inplace[:,:,2].max().clone()
        min = no_inplace[:,:,2].min().clone()

        final_vertices[:,:,2] = (final_vertices[:,:,2]-min)/(max-min)*79+10

        # aligned_vertices[:,:,2] = (aligned_vertices[:,:,2]-aligned_vertices[:,:,2].min())/(aligned_vertices[:,:,2].max()-aligned_vertices[:,:,2].min())*79+10


        final_vertices[:,:,0] = ((final_vertices[:,:,0] / 512) -0.5)*2
        final_vertices[:,:,1] = ((final_vertices[:,:,1] / 512) -0.5)*2

        # FILL SCENE
        cameras, renderer = create_renderer(batch_size,(H, W), self.device)
        smpl_body_meshes = Meshes(verts=final_vertices, faces=self._smpl._faces[None].expand(len(verts), -1, -1))
 

        # self.verts_rgb = self._tex_mlp(self.verts_rgb)
        # verts_rgb = torch.ones_like(self._smpl._v_template).float().cuda()
        # tex_color=np.array((0, 0, 0)) / 255
        # vertex_colors = torch.ones_like(self._smpl._v_template).float().cuda() * torch.tensor(tex_color, device=self.device).float().view(1, 3).float().cuda()
        # # define meshes and textures
        # vertex_colors = vertex_colors.repeat(verts.shape[0],1,1)
        # textures = TexturesVertex(vertex_colors)
        # # textures = TexturesVertex(tex)
        # textures.to(self.device)
        # smpl_body_meshes = Meshes(aligned_vertices, self._smpl._faces[None].expand(len(verts), -1, -1), textures)
        # img = renderer(smpl_body_meshes)
    
        # RASTERIZING FRAGMENTS
        return self._rasterize(smpl_body_meshes, cameras, (H, W))

    
    
    
    def _render_semantics(self, verts, H, W, return_rasterizer_results=False, rasterized_results=None,
                          return_confidence=False):

        # FILL SCENE
        batch_size = verts.shape[0]
        cameras,_ = create_renderer(batch_size, (H, W), self.device)
        smpl_body_meshes = Meshes(verts=verts, faces=self._smpl._faces[None].expand(len(verts), -1, -1))

        # RASTERIZING FRAGMENTS
        if rasterized_results is not None:
            fragments, screen_coords = rasterized_results
        else:
            fragments, screen_coords = self._rasterize(smpl_body_meshes, cameras, (H, W))
        N, H, W, K, _ = fragments.bary_coords.shape

        # get semantics labels and optionally confidence
        face_semantics = self._smpl.vert_labels


        C = face_semantics.shape[-1]
        face_semantics = face_semantics.repeat(N, 1)[smpl_body_meshes.faces_packed()]
        pixel_face_semantics = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords,
                                                           face_semantics)  # N x H x W x K x C

        blend_params = BlendParams(sigma=0, gamma=0, background_color=[0.0] * C)
        semantic_rendering = hard_feature_blend(pixel_face_semantics, fragments, blend_params)  # N x H x W x C+1
        semantic_rendering = semantic_rendering.permute(0, 3, 1, 2)  # N x C+1 x H x W


        if return_rasterizer_results:
            return semantic_rendering, dict(fragments=fragments, screen_coords=screen_coords)
        else:
            return semantic_rendering  # N x C+1 x H x W






    def _render_normals(self, verts, K, R, T, H, W, cameras=None, smpl_body_meshes=None, faces=None,
                        return_rasterizer_results=False, rasterized_results=None):
        """
        renders tensor of normals of shape N x 3 x H x W
        :param batch:
        :param verts:
        :param faces: faces of shape N x F x 3, falls back to smpl_body faces
        :return:
        """
        batch_size = verts.shape[0]

        # FILL SCENE
        if cameras is None:
            cameras, _ = create_renderer(batch_size, (H, W), self.device)
        if smpl_body_meshes is None:
            faces = faces if faces is not None else self._smpl._faces[None].expand(len(verts), -1, -1)
            smpl_body_meshes = Meshes(verts=verts, faces=faces)

        
        
        # RASTERIZING FRAGMENTS
        if rasterized_results is not None:
            fragments, screen_coords = rasterized_results
        else:
            fragments, screen_coords = self._rasterize(smpl_body_meshes, cameras, (H, W))
        N, H, W, K, _ = fragments.bary_coords.shape

        # NORMAL RENDERING


        
        face_normals = smpl_body_meshes.verts_normals_padded()  # N x V_max x 3
        face_normals = cameras.get_world_to_view_transform().transform_normals(face_normals)
        face_normals = face_normals.flatten(end_dim=1)[smpl_body_meshes.verts_padded_to_packed_idx()]  # N * V x 3
        face_normals = face_normals[smpl_body_meshes.faces_packed()]
        pixel_face_normals = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_normals)

        blend_params = BlendParams(sigma=0, gamma=0, background_color=[0.0] * 3)
        normal_rendering = hard_feature_blend(pixel_face_normals, fragments, blend_params)
        normal_rendering = normal_rendering.permute(0, 3, 1, 2)  # shape N x 3+1 x H x W

        # # flip orientations
        normal_rendering[:, :3] *= -1

        if return_rasterizer_results:
            return normal_rendering, dict(fragments=fragments, screen_coords=screen_coords)

        return normal_rendering   # shape N x 3+1 x H x W



    def _render_rgba(self, verts, K, R, T, H, W, pose, return_rasterizer_results=False,
                     rasterized_results=None, rendered_normals=None, center_prediction=False):


        # FILL SCENE
        batch_size = verts.shape[0]
        cameras, renderer = create_renderer(batch_size,(H, W), self.device)
        
        
        smpl_body_meshes = Meshes(verts=verts, faces=self._smpl._faces[None].expand(len(verts), -1, -1))

        # RASTERIZING FRAGMENTS
        # if rasterized_results is not None and not center_prediction:
        #     fragments, screen_coords = rasterized_results
        # else:
        #     fragments, screen_coords = self._rasterize(smpl_body_meshes, cameras, (H, W), center_prediction=center_prediction)
        
        fragments, screen_coords = self._rasterize(smpl_body_meshes, cameras, (H, W), center_prediction=center_prediction)
            
            
        N, H, W, faces_per_pix, _ = fragments.bary_coords.shape

        assert faces_per_pix == 1  # otherwise explicit texels might get messed up

        # FACE COORD RENDERING
        face_coords = self._smpl._face_coords_normed.repeat(N, 1, 1)
        # shape: N x H x W x K x 3
        pixel_face_coords = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_coords)

        # EXPL TEXTURE SAMPLING
        mask = fragments.pix_to_face != -1
        uv_coords = self._smpl._face_uvcoords.repeat(N, 1, 1)

        # shape: N x H x W x K x 2
        pixel_uv_coords = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, uv_coords)[..., :2]
        pixel_uv_ids = self._smpl._face_uvmap.repeat(N)[fragments.pix_to_face]  # N x H x W x K
        pixel_uv_ids[~mask] = -1



        mouth_frequencies, mouth_phase_shifts = self._texture.predict_frequencies_phase_shifts(pose.float())
        stat_frequencies, stat_phase_shifts = self._texture.predict_frequencies_phase_shifts(
            torch.zeros_like(pose.float()))

        frequencies = torch.stack((mouth_frequencies, stat_frequencies), dim=1)  # N x 3 x C
        phase_shifts = torch.stack((mouth_phase_shifts, stat_phase_shifts), dim=1)  # N x 3 x C

        # rendering semantic map
        # semantics = self._render_semantics(verts, H, W, rasterized_results=[fragments, screen_coords],
        #                                    return_confidence=True).detach()

        # with torch.no_grad():
        #     mouth_weights = semantics
        #     face_weights = (semantics[:, -1] - mouth_weights).clip(min=0)
        #     region_weights = torch.stack((mouth_weights, face_weights), dim=-1)  # N x H x W x 3
        #     region_weights = region_weights.view(N, H, W, 1, -1)  # N x H x W x K x 3

        if rendered_normals is None:
            rendered_normals = self._render_normals(verts, K, R, T, H, W, cameras=cameras, smpl_body_meshes=smpl_body_meshes,
                                                    rasterized_results=[fragments, screen_coords], )[:, :3]

        # MLP TEXTURE SAMPLING
        pixel_face_coords_masked = pixel_face_coords[mask]
        pixel_uv_coords_masked = pixel_uv_coords[mask]
        pixel_uv_ids_masked = pixel_uv_ids[mask]

        if getattr(self, "n_upsample", 1) != 1:
            rendered_normals = torchvision.transforms.functional.resize(rendered_normals, (
                    torch.tensor(rendered_normals.shape[-2:]) / self.n_upsample).int().tolist())
        normal_encoding = self._normal_encoder(rendered_normals)  # N x C x H x W

        if getattr(self, "n_upsample", 1) != 1:
            normal_encoding = torchvision.transforms.functional.resize(normal_encoding, (H, W))

        pixel_normal_encoding = normal_encoding.permute(0, 2, 3, 1)
        pixel_normal_encoding = pixel_normal_encoding.unsqueeze(-2).expand(-1, -1, -1, faces_per_pix,
                                                                           -1)  # N x H x W x K x D
        pixel_normal_encoding_masked = pixel_normal_encoding[mask]
        pixel_expl_features_masked = self._explFeatures(pixel_uv_coords_masked.view(1, 1, -1, 2),
                                                        pixel_uv_ids_masked.view(1, 1, -1))
        pixel_expl_features_masked = pixel_expl_features_masked[0, :, 0, :].permute(1, 0)
        static_mlp_conditions_masked = torch.cat((pixel_face_coords_masked, pixel_expl_features_masked), dim=-1)
        n_masked = torch.arange(N).view(N, 1, 1, 1).expand(N, H, W, faces_per_pix)[mask]
        
        
        frequencies_masked = torch.sum(frequencies[n_masked], dim=1)
        phase_shifts_masked = torch.sum(phase_shifts[n_masked], dim=1)
        # N' x C

        mlp_pixel_rgb = torch.zeros(N, H, W, faces_per_pix, 3, dtype=fragments.bary_coords.dtype,
                                    device=fragments.bary_coords.device)

        mlp_pixel_rgb[mask] = self._texture.forward(static_mlp_conditions_masked, frequencies_masked,
                                                    phase_shifts_masked,
                                                    dynamic_conditions=pixel_normal_encoding_masked)
        ##
        ## region weight로 semantic 별로 주기 (추가)
        ##


        blend_params = BlendParams(sigma=0, gamma=0, background_color=[1.0] * 3)
        mlp_rgba_pred = hard_feature_blend(mlp_pixel_rgb, fragments, blend_params)
        # N x H x W x 3 + 1

        # if random_select:
        #     feature_img[~mask.expand(-1, -1, -1, 4)] = 1

        
        #rgh 이미지가 나옴
        mlp_rgba_pred = mlp_rgba_pred.permute(0, 3, 1, 2)

        if return_rasterizer_results:
            return mlp_rgba_pred, dict(fragments=fragments, screen_coords=screen_coords)
        else:
            return mlp_rgba_pred  # N x 4 x H x W



    def _compute_laplacian_smoothing_loss(self, vertices, offset_vertices):

        batch_faces = self._smpl._faces.repeat(vertices.shape[0],1,1)
        
        basis_meshes = Meshes(verts=vertices, faces=batch_faces)
        offset_meshes = Meshes(verts=offset_vertices, faces=batch_faces)

        # compute weights
        N = len(offset_meshes)
        basis_verts_packed = basis_meshes.verts_packed()  # (sum(V_n), 3)
        offset_verts_packed = offset_meshes.verts_packed()

        with torch.no_grad():
            L = basis_meshes.laplacian_packed()
            basis_lap = L.mm(basis_verts_packed)  # .norm(dim=1)  * weights

        offset_lap = L.mm(offset_verts_packed)  # .norm(dim=1)  * weights
        diff = (offset_lap - basis_lap) ** 2
        diff = diff.sum(dim=1)
        diff = diff.view(N, -1)

        return diff.sum() / N



    def _compute_silhouette_loss(self, batch, verts, rasterized_results=None, return_iou=False):
        gt_mask = batch["seg"]
        K = batch["cam_intrinsic"]
        R = batch["cam_extrinsic"]
        T = batch["cam_extrinsic"]
        N, V, _ = verts.shape
        H, W = gt_mask.shape[-2:]
        cam_parmas = batch["smpl_trans"]
        
        batch_size = verts.shape[0]
        
        cameras, renderer = create_renderer(batch_size, (H, W), self.device)

        
        semantics_pred, raster_res =self._render_normals(verts, K, R, T, H, W, return_rasterizer_results=True,
                                                       rasterized_results=rasterized_results)

        # semantics_pred, raster_res =self._render_semantics(verts,  H, W, return_rasterizer_results=True, rasterized_results=rasterized_results)

        pred_mask = semantics_pred[:, [-1]]

        screen_coords = raster_res["screen_coords"] * (-1)  # N x H_r x W_r x 2

        loss = self._masked_L1_mask(gt_mask, pred_mask)
        loss, iou = calc_holefilling_segmentation_loss(gt_mask.float(), pred_mask.float(), screen_coords,
                                                       return_iou=True, sigma=2.0)

        if return_iou:
            return loss  #, iou.mean()
        else:
            return loss


    def _compute_rgb_losses(self, batch, verts,  pose, rasterized_results=None,
                            rendered_normals=None):


        rgb_gt = batch["rgb"]
        mask = batch["seg"].detach()

        K = batch["cam_intrinsic"]
        R = batch["camera_rotation"]
        T = batch["cam_extrinsic"]
        cam_parmas = batch["smpl_trans"]

        H, W = rgb_gt.shape[-2:]
        batch_size = batch["rgb"].shape[0]
        cam_parmas = batch["smpl_trans"]
        manual_cam = batch["manual_rendering"]



        #### from SMPL G.T vertices
        render_verts = verts.clone()

        cam_intri= torch.tensor(batch["cam_intrinsic"]).float().to(self.device)
        cam_r = torch.tensor(batch["camera_rotation"]).float().to(self.device)
        cam_t = torch.tensor(batch["camera_translation"]).float().to(self.device)
        
        # cam_transpose = cam_transpose.repeat(verts.shape[0],1,1)
        cam_transpose = cam_t.unsqueeze(1).permute(0,2,1).to(self.device)

        ho = torch.ones((render_verts.shape[0], render_verts.shape[1],1)).float().to(self.device)
        homo_vertices = torch.cat((render_verts, ho), dim=2)
        
        mat = torch.cat((cam_r, cam_transpose), dim=2)
        ze = torch.Tensor([0,0,0,1]).float().to(self.device)
        ze = ze.unsqueeze(0)
        ze = ze.repeat(verts.shape[0],1, 1)

        homo_matrix = torch.cat((mat, ze), dim=1)
        mat_vertices = torch.matmul(homo_matrix, homo_vertices.permute(0,2,1)).permute(0,2,1)
        mat_vertices = mat_vertices[:,:,:3] / mat_vertices[:,:,-1].unsqueeze(2)
        aligned_vertices = torch.matmul(cam_intri, mat_vertices.permute(0,2,1)).permute(0,2,1)


        # [331, 990, 601, 1260]

        final_vertices = aligned_vertices.clone()
        final_vertices[:,:,:2]  =  aligned_vertices[:,:,:2] / aligned_vertices[:,:,-1].unsqueeze(2) 
        final_vertices[:,:,0] = final_vertices[:,:,0] - manual_cam[:,2].unsqueeze(1).repeat(1, final_vertices[:,:,0].shape[1])
        final_vertices[:,:,1] = final_vertices[:,:,1] - manual_cam[:,0].unsqueeze(1).repeat(1, final_vertices[:,:,0].shape[1])

        scaler = 512/(manual_cam[:,1]-manual_cam[:,0])
        final_vertices[:,:,0:2] = final_vertices[:,:,0:2] * scaler.unsqueeze(1).unsqueeze(2).repeat(1, final_vertices[:,:,0:2].shape[1], final_vertices[:,:,0:2].shape[2] )

        no_inplace = final_vertices.clone()
        max = no_inplace[:,:,2].max().clone()
        min = no_inplace[:,:,2].min().clone()

        final_vertices[:,:,2] = (final_vertices[:,:,2]-min)/(max-min)*79+10

        # aligned_vertices[:,:,2] = (aligned_vertices[:,:,2]-aligned_vertices[:,:,2].min())/(aligned_vertices[:,:,2].max()-aligned_vertices[:,:,2].min())*79+10


        final_vertices[:,:,0] = ((final_vertices[:,:,0] / 512) -0.5)*2
        final_vertices[:,:,1] = ((final_vertices[:,:,1] / 512) -0.5)*2
        
        rgba_pred, raster_dict = self._render_rgba(final_vertices, K, R, T, H, W,  pose=pose, rendered_normals=rendered_normals,
                                                   return_rasterizer_results=True,
                                                   rasterized_results=rasterized_results)
        
        predicted_images = rgba_pred[:, :3]
        predicted_seg = rgba_pred[:, [3]].detach()

        screen_coords = raster_dict["screen_coords"] * (-1)
        screen_colors = screen_grad(batch["rgb"], screen_coords)

        
        # predicted_images = mask * predicted_images
        # mask = predicted_seg * mask
        # use intersection mask
        rgb_loss = self._masked_L1(predicted_images, screen_colors, mask)
        # rgb_loss = self._masked_L1(predicted_images, screen_colors)

        # ATTENTION: perceptual loss only provides gradient to texture!
        # don't apply gradient to boundary of silhouette -> artifacts occur otherwise
        gradient_margin = int( H / 50)
        predicted_images = softmask_gradient(predicted_images, erode_mask(predicted_seg, gradient_margin))
        perc_loss = self._perceptual_loss(predicted_images, screen_colors.detach()).mean() if \
            self.get_current_lrs_n_lossweights()["w_perc"] >= 0 else 0.0


        return dict(rgb_loss=rgb_loss, perc_loss=perc_loss)

    


    # def _compute_rgb_losses(self, batch, verts,  pose, rasterized_results=None,
    #                         rendered_normals=None):
    #     """
    #     Computes photometric and perceptual loss
    #     :returns dict("rgb_loss"=scalar_rgb_loss_tensor, "perc_loss"=scalar_perc_loss_tensor)
    #     """

    #     rgb_gt = batch["rgb"]
    #     mask = batch["seg"].detach()

    #     K = batch["cam_intrinsic"]
    #     R = batch["camera_rotation"]
    #     T = batch["cam_extrinsic"]
    #     H, W = rgb_gt.shape[-2:]
    #     cam_parmas = batch["smpl_trans"]
        
        
    #     verts = verts*cam_parmas[:,0:1].view(verts.shape[0],1,1)
    #     verts[:,:,0:1]=verts[:,:,0:1]+cam_parmas[:,1:2].view(verts.shape[0],1,1)*cam_parmas[:,0:1].view(verts.shape[0],1,1)
    #     verts[:,:,1:2]=verts[:,:,1:2]+cam_parmas[:,2:3].view(verts.shape[0],1,1)*cam_parmas[:,0:1].view(verts.shape[0],1,1)

    #     verts[:,:,2:3]=verts[:,:,2:3]*(-1)
    #     verts[:,:,1:2]=verts[:,:,1:2]*(-1)
        
    #     rgba_pred, raster_dict = self._render_rgba(verts, K, R, T, H, W,  pose=pose, rendered_normals=rendered_normals,
    #                                                return_rasterizer_results=True,
    #                                                rasterized_results=rasterized_results)
        
    #     predicted_images = rgba_pred[:, :3]
    #     predicted_seg = rgba_pred[:, [3]].detach()

    #     screen_coords = raster_dict["screen_coords"] * (-1)
    #     screen_colors = screen_grad(batch["rgb"], screen_coords)

    #     mask = predicted_seg * mask
    #     # use intersection mask
    #     rgb_loss = self._masked_L1(predicted_images, screen_colors, mask)
    #     # rgb_loss = self._masked_L1(predicted_images, screen_colors)

    #     # ATTENTION: perceptual loss only provides gradient to texture!
    #     # don't apply gradient to boundary of silhouette -> artifacts occur otherwise
    #     gradient_margin = int(max(H, W) / 50)
    #     predicted_images = softmask_gradient(predicted_images, erode_mask(predicted_seg, gradient_margin))
    #     perc_loss = self._perceptual_loss(predicted_images, screen_colors.detach()).mean() if \
    #         self.get_current_lrs_n_lossweights()["w_perc"] >= 0 else 0.0


    #     return dict(rgb_loss=rgb_loss, perc_loss=perc_loss)




    # def _compute_smpl_body_reg_losses(self, batch):
    #     indices = torch.unique(batch["frame"])
    #     shape_reg = torch.sum(self._shape[indices] ** 2) / 2
    #     pose_reg = torch.sum(self._pose[indices] ** 2) / 2
        
    #     return shape_reg, pose_reg



    
    def _compute_surface_consistency(self, offsets):

        N = len(offsets)

        if N == 1:
            return 0

        # offsets = offsets.view(N, -1)

        # for each frame select another frame and compare their offsets
        all_indices = list(range(N))
        possible_partners = [all_indices[:i] + all_indices[i + 1:] for i in all_indices]
        partner_indices = [np.random.choice(p) for p in possible_partners]

        frame_i_offsets = offsets
        frame_j_offsets = offsets[partner_indices]

        diff = torch.abs(frame_i_offsets - frame_j_offsets.detach())

        return diff.sum() / N


    def _compute_edge_length_loss(self, offset_vertices):

        batch_faces = self._smpl._faces[None, ...]
        batch_faces = batch_faces.expand(offset_vertices.shape[0], *batch_faces.shape[1:])
        meshes = Meshes(verts=offset_vertices, faces=batch_faces)

        N = len(meshes)
        edges_packed = meshes.edges_packed()  # (sum(E_n), 2)
        verts_packed = meshes.verts_packed()  # (sum(V_n), 3)

        verts_edges = verts_packed[edges_packed]
        v0, v1 = verts_edges.unbind(1)
        lengths = (v0 - v1).norm(dim=1, p=2)

        # add batch dimension and filter scalp edges
        lengths = lengths.view(N, -1)
        mean_lengths = torch.mean(lengths, dim=1, keepdim=True).expand(lengths.shape).detach()
        deviation = torch.abs(lengths - mean_lengths)

        # calculate loss only when larger than 3x mean
        deviation[deviation < 1.5 * mean_lengths] *= 0

        return deviation.mean()
    
    
    
    def _compute_normal_loss(self, batch, vertices_offsets, rasterized_results=None,
                             return_normal_pred=False):
        """

        :param batch:
        :param vertices_offsets:
        :param rendered_semantics: tensor of shape N x C x H x W with rendered semantics of mesh_
        :param rasterized_results:
        :return:
        """

        normal_gt = batch["normal"]
        mask_gt = batch["seg"]
        N, _3, H, W = normal_gt.shape
        K = batch["cam_intrinsic"]
        
        #### 사용 X  나중에 지우기
        R = batch["cam_extrinsic"]
        T = batch["cam_extrinsic"]
        
        cam_parmas = batch["smpl_trans"]
        
        
        normal_pred, raster_res = self._render_normals(vertices_offsets, K, R, T, H, W, return_rasterizer_results=True,
                                                       rasterized_results=rasterized_results)
 

        normal_pred, seg_pred = normal_pred[:, :3], normal_pred[:, [3]]



        mask = mask_gt * seg_pred

        sigma = max(H, W) / 50
        with torch.no_grad():
            blurred_normal_gt = masked_gaussian_blur(normal_gt.detach(), mask_gt, sigma=sigma, kernel_size=None,
                                                     blur_fc=seperated_gaussian_blur)
            blurred_normal_pred = masked_gaussian_blur(normal_pred.detach(), seg_pred, sigma=sigma, kernel_size=None,
                                                       blur_fc=seperated_gaussian_blur)

        normal_gt_lapl = normal_gt - blurred_normal_gt
        normal_pred_lapl = normal_pred - blurred_normal_pred


        # shape [N, H, W, 2]
        screen_coords = raster_res["screen_coords"] * (-1)
        normal_gt_lapl = screen_grad(normal_gt_lapl, screen_coords) 

        # normal_gt_lapl = screen_grad(normal_gt_lapl, screen_coords) 
        # normal_gt_gr = screen_grad(normal_gt, screen_coords)
        # normal_pred_gr = screen_grad(normal_pred, screen_coords)
        # normal_pred = normal_pred * (-1)
        
        mask_normal = mask_gt-seg_pred
        # loss1 = self._masked_L1(normal_gt_lapl, normal_pred_lapl, mask_normal)

        loss1 = self._masked_L1(normal_gt, normal_pred)
        loss = self._masked_L1(normal_gt_lapl, normal_pred_lapl, mask)

        # loss1 = self._L1(normal_gt, normal_pred, mask)
        # loss2 = self._L1(normal_gt_lapl, normal_pred_lapl, mask)

        loss = loss1 + loss
        # loss = loss1
        

        if return_normal_pred:
            return loss, normal_pred
        else:
            return loss


    def get_normal_coord_system(self, normals):
        """
        returns tensor of basis vectors of coordinate system that moves with normals:

        e_x = always points horizontally_optim
        e_y = always pointing up
        e_z = normal vector

        returns tensor of shape N x 3 x 3

        :param normals: tensor of shape N x 3
        :return:
        """
        device = normals.device
        dtype = normals.dtype
        N = len(normals)

        assert len(normals.shape) == 2

        normals = normals.detach()
        e_y = torch.tensor([0., 1., 0.], device=device, dtype=dtype)
        e_x = torch.tensor([0., 0., 1.], device=device, dtype=dtype)

        basis = torch.zeros(len(normals), 3, 3, dtype=dtype, device=device)
        # e_z' = e_n
        basis[:, 2] = torch.nn.functional.normalize(normals, p=2, dim=-1)

        # e_x' = e_n x e_y except e_n || e_y then e_x' = e_x
        normal_parallel_ey_mask = ((basis[:, 2] * e_y[None]).sum(dim=-1).abs() == 1)
        basis[:, 0] = torch.cross(e_y.expand(N, 3), basis[:, 2], dim=-1)
        basis[normal_parallel_ey_mask][:, 0] = e_x[None]
        basis[:, 0] = torch.nn.functional.normalize(basis[:, 0], p=2, dim=-1)
        basis[normal_parallel_ey_mask][:, 0] = e_x[None]
        basis[:, 0] = torch.nn.functional.normalize(basis[:, 0], p=2, dim=-1)

        # e_y' = e_z' x e_x'
        basis[:, 1] = torch.cross(basis[:, 2], basis[:, 0], dim=-1)
        #basis[:, 1] = torch.nn.functional.normalize(basis[:, 1], p=2, dim=-1)

        assert torch.all(torch.norm(basis, dim=-1, p=2) > .99)

        return basis



    def _optimize_offsets(self, batch):
        """
        Optimize the smpl_body head model + offets based on silhouette, normal map and semantic map

        :param batch: input data batch as returned from self.prepare_batch()
        :return: scalar loss, log dict
        """

        # get loss term weights
        # w_lap = self._decays["lap"].get(self.current_epoch)
        # w_silh = self._decays["silh"].get(self.current_epoch)

        # computes losses
        init_vertices = batch["vertices"].clone()
        init_vertices = init_vertices.float().to(self.device)
        offsets=self._predict_offsets(batch["pose"], init_vertices)

        # smpl_body_params_offsets = self._create_smpl_param_batch(batch, init_vertices)
        # offsets = smpl_body_params_offsets["offsets"]
        # offsets_verts = init_vertices.clone()


        ### normal 방향으로 더하기 ###
        normals = vertex_normals(init_vertices, self._smpl._faces[None].expand(len(offsets), -1, -1))
        B, V, _3 = normals.shape
        normal_coord_sys = self.get_normal_coord_system(normals.view(-1, 3)).view(B, V, 3, 3)
        offsets = torch.matmul(normal_coord_sys.permute(0, 1, 3, 2), offsets.unsqueeze(-1)).squeeze(-1)
        offsets_verts = init_vertices + offsets


        # offsets_verts = init_vertices + offsets


        raster_res = self._rasterize_smpl(batch, offsets_verts)
        silh_loss = self._compute_silhouette_loss(batch, offsets_verts, rasterized_results=raster_res)
        # silh_loss = self._compute_silhouette_loss(batch, offsets_verts, pose_loss=False, rasterized_results=raster_res)


        normal_loss = self._compute_normal_loss(
            batch,
            offsets_verts,
            rasterized_results=raster_res,
        )
        
        lap_loss = self._compute_laplacian_smoothing_loss(init_vertices, offsets_verts)
        loss_weights = self.get_current_lrs_n_lossweights()

        # total_loss = loss_weights["w_norm"] * normal_loss + loss_weights["w_silh"] * silh_loss 
        total_loss = 0.0001*normal_loss + 0.001 * silh_loss 
        # + loss_weights["w_lap"] * lap_loss
        # + loss_weights["w_lap"] * lap_loss
        # total_loss = loss_weights["w_silh"] * silh_loss


        log_dict = {
            "normal_loss": normal_loss,
            "silh_loss": silh_loss,
            "lap_loss": lap_loss,
            # "edge_loss": edge_loss,
            # "shape_reg": shape_reg,
            # "pose_reg": pose_reg,
            # "surface_reg": surface_reg,
            "total_loss": total_loss,
        }

        # for k, v in self._decays.items():
        #     decay = v.get(self.current_epoch)
        #     log_dict[f"decay_{k}"] = decay

        return total_loss, log_dict
    



    def _optimize_texture(self, batch):

        # construct mesh
        smpl_body_params_offsets = self._create_smpl_param_batch(batch)
        offsets_verts = self._forward_smpl(smpl_body_params_offsets)

        # produce noisy smpl_body parameters for conditioning the texture (generalizes better)
        smpl_body_params_offsets_detached = dict()
        for key, val in smpl_body_params_offsets.items():
            smpl_body_params_offsets_detached[key] = val.detach()
            
        pose = batch["pose"]
        
        phot_lossdict = self._compute_rgb_losses(
            batch,
            offsets_verts,
            pose=pose,
        )
        
        rgb_loss = phot_lossdict["rgb_loss"]
        perc_loss = phot_lossdict["perc_loss"]

        weights = self.get_current_lrs_n_lossweights()

        total_loss = weights["w_rgb"] * rgb_loss + weights["w_perc"] * perc_loss

        log_dict = {
            "rgb_loss": rgb_loss,
            "perc_loss": perc_loss,
            "total_loss": total_loss,
        }

        for k, v in self._decays.items():
            decay = v.get(self.current_epoch)
            log_dict[f"decay_{k}"] = decay

        return total_loss, log_dict




    def _optimize_jointly(self, batch):

        # construct mesh
        # smpl_params = self._create_smpl_param_batch(batch, ignore_offsets=True)
        # vertices = self._forward_smpl(smpl_params)

        # smpl_body_params_offsets = self._create_smpl_param_batch(batch, vertices)
        # offsets_verts = self._forward_smpl(smpl_body_params_offsets)
        # raster_res = self._rasterize_smpl(batch, offsets_verts)

        

        # computes losses
        init_vertices = batch["vertices"].clone()
        init_vertices = init_vertices.float().to(self.device)
        smpl_body_params_offsets = self._create_smpl_param_batch(batch, init_vertices)

        offsets = smpl_body_params_offsets["offsets"]
        offsets_verts = init_vertices.clone()


        ### normal 방향으로 더하기 ###
        normals = vertex_normals(init_vertices, self._smpl._faces[None].expand(len(offsets), -1, -1))
        B, V, _3 = normals.shape
        normal_coord_sys = self.get_normal_coord_system(normals.view(-1, 3)).view(B, V, 3, 3)
        offsets = torch.matmul(normal_coord_sys.permute(0, 1, 3, 2), offsets.unsqueeze(-1)).squeeze(-1)
        offsets_verts = init_vertices + offsets

        # offsets_verts = self._forward_smpl(smpl_body_params_offsets)
        
        raster_res = self._rasterize_smpl(batch, offsets_verts)

        silh_loss = self._compute_silhouette_loss(batch, offsets_verts, rasterized_results=raster_res)

        
        normal_loss = self._compute_normal_loss(
            batch,
            offsets_verts,
            rasterized_results=raster_res,
        )

        edge_loss = self._compute_edge_length_loss(offsets_verts)

        # regularization terms
        lap_loss = self._compute_laplacian_smoothing_loss(init_vertices, offsets_verts)
        # surface_reg = self._compute_surface_consistency(smpl_body_params_offsets["offsets"])
        
        # shape_reg, pose_reg = self._compute_smpl_body_reg_losses(batch)
        weights = self.get_current_lrs_n_lossweights()
        
        
        pose = batch["pose"]
        # pose = smpl_body_params_offsets["pose"]

        phot_lossdict = self._compute_rgb_losses(
            batch,
            offsets_verts,
            pose=pose,
        )

        rgb_loss = phot_lossdict["rgb_loss"]
        perc_loss = phot_lossdict["perc_loss"]
        # total_loss =normal_loss

        # total_loss = weights["w_rgb"] * rgb_loss + weights["w_perc"] * perc_loss + weights["w_norm"] * normal_loss +\
        #               weights["w_lap"] * lap_loss + weights["w_silh"] * silh_loss
        
        # total_loss = 0.5* rgb_loss + 0.5 * perc_loss + 0.0001 * normal_loss +  silh_loss + 0.0001 * lap_loss + 0.0001*edge_loss
                     
                    #weights["w_edge"] * edge_loss + weights["w_shape_reg"] * shape_reg + weights["w_silh"] * silh_loss + weights["w_pose_reg"] * pose_reg + weights["w_surface_reg"] * surface_reg

                   
        total_loss = silh_loss + normal_loss + rgb_loss + perc_loss + 0.00001 * lap_loss

        log_dict = {
            "rgb_loss": rgb_loss,
            "normal_loss" : normal_loss,
            "silh_loss": silh_loss,
            "perc_loss": perc_loss,
            "lap_loss": lap_loss,
            # "edge_loss": edge_loss,
            # "shape_reg": shape_reg,
            # "surface_reg": surface_reg,
            # "pose_reg": pose_reg,
            "total_loss": total_loss,
        }

        for k, v in self._decays.items():
            decay = v.get(self.current_epoch)
            log_dict[f"decay_{k}"] = decay

        return total_loss, log_dict



    

    def toggle_optimizer(self, optimizers):
        """
        Makes sure only the gradients of the current optimizer's parameters are calculated
        in the training step to prevent dangling gradients in multiple-optimizer setup.

        .. note:: Only called when using multiple optimizers

        Override for your own behavior

        It works with ``untoggle_optimizer`` to make sure param_requires_grad_state is properly reset.

        Args:
            optimizer: Current optimizer used in training_loop
            optimizer_idx: Current optimizer idx in training_loop
        """

        # Iterate over all optimizer parameters to preserve their `requires_grad` information
        # in case these are pre-defined during `configure_optimizers`
        param_requires_grad_state = {}
        for opt in self.optimizers(use_pl_optimizer=False):
            for group in opt.param_groups:
                for param in group["params"]:
                    # If a param already appear in param_requires_grad_state, continue
                    if param in param_requires_grad_state:
                        continue
                    param_requires_grad_state[param] = param.requires_grad
                    param.requires_grad = False

        # Then iterate over the current optimizer's parameters and set its `requires_grad`
        # properties accordingly
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    param.requires_grad = param_requires_grad_state[param]
        self._param_requires_grad_state = param_requires_grad_state


    def untoggle_optimizer(self):
        for param, requires_grad in self._param_requires_grad_state.items():
            param.requires_grad = requires_grad

        # save memory
        self._param_requires_grad_state = dict()



    def step(self, batch, batch_idx, stage="train"):
        
        if stage == "train" or self.fit_residuals:
            with torch.autograd.set_detect_anomaly(True):
                optim = self._get_current_optimizer()

                self.toggle_optimizer(optim)  # automatically toggle right requires_grads
                # step 1: standard optimization of smpl_body model and offsets

                if self.current_epoch < self.hparams["epochs_offset"]:
                    loss, log_dict = self._optimize_offsets(batch)

                # step 2: optimization texture
                elif self.current_epoch < self.hparams["epochs_offset"] + self.hparams["epochs_texture"]:
                    loss, log_dict = self._optimize_texture(batch)

                # step 3: optimization texture and shape
                else:
                    loss, log_dict = self._optimize_jointly(batch)

                # for opt in optim:
                #     opt.zero_grad()

                self.manual_backward(loss)

                for opt in optim:
                    opt.step()

                for opt in optim:
                    opt.zero_grad(set_to_none=True)  # saving gpu memory

                self.untoggle_optimizer()

        else:
            with torch.no_grad():
                if self.current_epoch < self.hparams["epochs_offset"]:
                    loss, log_dict = self._optimize_offsets(batch)
                else:
                    loss, log_dict = self._optimize_jointly(batch)

        # give all keys in logdict val_ prefix
        for key in list(log_dict.keys()):
            val = log_dict.pop(key)
            log_dict[f"{stage}_{key}"] = val

        # log scores
        log_dict["step"] = self.current_epoch
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # log images
        log_images = self.current_epoch % self.hparams["image_log_period"] == 0
        # if batch_idx == 0 and log_images:_compute_normal_loss
        #     try:
        #         interesting_frames = [1455, 536] if stage == "train" else [826, 1399]
        #         dataset = self.trainer.train_dataloader.dataset.datasets if stage == "train" else \
        #             self.trainer.val_dataloaders[0].dataset
        #         interesting_samples = [dataset[dataset.frame_list.index(f)] for f in interesting_frames]
        #     except ValueError:
        #         interesting_samples = [dataset[i] for i in np.linspace(0, len(dataset), 4).astype(int)[1:3]]
        #     # vis_batch = dict_2_device(stack_dicts(*interesting_samples), self.device)
        #     vis_batch = dict_2_device(batch, self.device)
        #     self._visualize_head(vis_batch, title=stage)


        return loss

    ## training 때 step
    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        self.is_train = True
        self.fit_residuals = False

        ret = self.step(batch, batch_idx, stage="train")

        self.is_train = False
        
        return ret
    


    def validation_step(self, batch, batch_idx, **kwargs):
        self.fit_residuals = self.current_epoch < self.hparams["epochs_offset"] or self.current_epoch >= self.hparams[
            "epochs_texture"] + self.hparams["epochs_offset"]

        with torch.set_grad_enabled(self.fit_residuals):
            self.step(batch, batch_idx, stage="val")

        self.fit_residuals = False




    @torch.no_grad()
    def _visualize_head(self, batch, title=f"smpl_body_fit"):
        rgba_pred = self.forward(batch, symmetric_rgb_range=False)
        shaded_pred = self.predict_shaded_mesh(batch)

        N =batch["rgb"].shape[0]

        rgb_gt = batch["rgb"]
        rgb_gt = rgb_gt[:N] * 0.5 + 0.5
        rgb_pred = rgba_pred[:N, :3]
        shaded_pred = shaded_pred[:N, :3]

        images = torch.cat([rgb_gt, rgb_pred, shaded_pred], dim=0)
        log_img = torchvision.utils.make_grid(images, nrow=N)
        self.logger.experiment.add_image(title + "prediction", log_img, self.current_epoch)

    

    ### 주로 모델의 추론 결과를 제공하고 싶을 때
    def forward(self, batch, ignore_expr=False, ignore_pose=False, center_prediction=False, symmetric_rgb_range=True):
        """
        returns rgba tensor of shape N x 3+1 x H x W with rgb values ranging from -1 ... 1 and alpha value
        ranging from 0 to +1
        :param batch:
        :param symmetric_rgb_range: if True rgb values range from -1 ... 1 else 0 ... 1
        :return:
        """
        K = batch["cam_intrinsic"]
        R = batch["camera_rotation"]
        T = batch["cam_extrinsic"]
        H, W = batch["rgb"].shape[-2:]


        smpl_params = self._create_smpl_param_batch(batch, ignore_offsets=True)
        vertices = self._forward_smpl(smpl_params)
        smpl_body_params_offsets = self._create_smpl_param_batch(batch, vertices)
        offsets_verts = self._forward_smpl(smpl_body_params_offsets)
        cam_parmas = batch["smpl_trans"]
        
        
        offsets_verts = offsets_verts*cam_parmas[:,0:1].view(offsets_verts.shape[0],1,1)
        offsets_verts[:,:,0:1]=offsets_verts[:,:,0:1]+cam_parmas[:,1:2].view(offsets_verts.shape[0],1,1)*cam_parmas[:,0:1].view(offsets_verts.shape[0],1,1)
        offsets_verts[:,:,1:2]=offsets_verts[:,:,1:2]+cam_parmas[:,2:3].view(offsets_verts.shape[0],1,1)*cam_parmas[:,0:1].view(offsets_verts.shape[0],1,1)

        # offsets_verts=offsets_verts[:,:,:].view(-1,3)
        offsets_verts[:,:,2:3]=offsets_verts[:,:,2:3]*(-1)
        offsets_verts[:,:,1:2]=offsets_verts[:,:,1:2]*(-1)
    

        # rgba prediction

        pose = batch["pose"]
        
        rgba_pred, _ = self._render_rgba(offsets_verts, K, R, T, H, W,  pose=pose, return_rasterizer_results=True)
        
        # if not symmetric_rgb_range:
        #     rgba_pred[:, :3] = torch.clip(rgba_pred[:, :3] * 0.5 + 0.5, min=0.0, max=1.0)
        rgba_pred = rgba_pred[:, :3]
        rgba_pred = (rgba_pred+1)/2
        
        return rgba_pred
    

    @torch.no_grad()
    def predict_reenaction(self, batch, driving_model, base_target_params, base_driving_params, return_alpha=False):

        K = batch["cam_intrinsic"]
        R= batch["cam_extrinsic"]
        T = batch["cam_extrinsic"]
        N, C, H, W = batch["rgb"].shape

        # OBTAINING smpl_body PARAMS
        smpl_body_params_offsets = self._create_smpl_param_batch(batch)

        # insert correct shape parameters
        smpl_body_params_offsets["shape"] = base_target_params["shape"].expand(N, -1)

        # adopt driving frame parameters
        smpl_body_params_driving = driving_model._create_smpl_param_batch(batch)
        for key in ["expr", "translation", "rotation", "neck", "jaw", "eyes"]:
            residual_param = smpl_body_params_driving[key] - base_driving_params[key].expand(N, -1)
            smpl_body_params_offsets[key] = base_target_params[key].expand(N, -1) + residual_param

        offsets_verts = self._forward_smpl(smpl_body_params_offsets, return_mouth_conditioning=True)

        # rgba prediction
        expr = smpl_body_params_offsets["expr"]
        pose = torch.cat((smpl_body_params_offsets["rotation"], smpl_body_params_offsets["neck"], smpl_body_params_offsets["jaw"],
                          smpl_body_params_offsets["eyes"]), dim=1)

        rgba_pred = self._render_rgba(offsets_verts, K, R, T, H, W, expr=expr, pose=pose)

        rgb_pred, seg_pred = rgba_pred[:, :3], rgba_pred[:, 3:]

        # rgb pred logging
        # rgb_pred.permute(0, 2, 3, 1)[seg_pred[:, 0] < .5] = 1  # change background to white  NOT NECESSARY
        rgb_pred = torch.clip(rgb_pred * 0.5 + 0.5, min=0, max=1)
        if return_alpha:
            return rgb_pred, seg_pred
        else:
            return rgb_pred

    @torch.no_grad()
    def predict_shaded_mesh(self, batch, tex_color=np.array((188, 204, 245)) / 255, light_colors=(0.4, 0.6, 0.3)):
        K = batch["cam_intrinsic"]
        R = batch["cam_extrinsic"]
        T = batch["cam_extrinsic"]
        H, W = batch["rgb"].shape[-2:]

        smpl_body_params_offsets = self._create_smpl_param_batch(batch)
        offsets_verts = self._forward_smpl(smpl_body_params_offsets)

        vertex_colors = torch.ones_like(offsets_verts) * torch.tensor(tex_color, device=self.device).float().view(1, 3)
        # define meshes and textures
        tex = TexturesVertex(vertex_colors)
        mesh = Meshes(
            verts=offsets_verts,
            faces=self._smpl._faces[None].expand(len(offsets_verts), -1, -1),
            textures=tex,
        )

        return render_shaded_mesh(mesh, K, R, T, (H, W), self.device, light_colors)

    def get_current_lrs_n_lossweights(self):
        epoch = self.current_epoch

        if epoch < self.hparams["epochs_offset"]:
            i = 0

        elif epoch < self.hparams["epochs_offset"] + self.hparams["epochs_texture"]:
            i = 1

        else:
            i = 2

        return dict(
            w_rgb=self.hparams["w_rgb"][i],
            w_perc=self.hparams["w_perc"][i],
            w_norm=self.hparams["w_norm"][i],
            w_edge=self.hparams["w_edge"][i],
            w_silh=self._decays["silh"].get(epoch),
            w_lap=self._decays["lap"].get(epoch),
            w_surface_reg=self.hparams["w_surface_reg"][i],
            w_shape_reg=self.hparams["w_shape_reg"][i],
            w_expr_reg=self.hparams["w_expr_reg"][i],
            w_pose_reg=self.hparams["w_pose_reg"][i],
            texture_weight_decay=self.hparams["texture_weight_decay"][i],
            smpl_lr=self.hparams["smpl_lr"][i],
            offset_lr=self.hparams["offset_lr"][i],
            tex_lr=self.hparams["tex_lr"][i],
            trans_lr=self._trans_lr[i],
        )


    def configure_optimizers(self):

        # smpl_body
        # smpl_params = [self._shape, self._pose]

        lrs = self.get_current_lrs_n_lossweights()

        # # translation gets smaller learning rate
        # params = [
        #     {"params": smpl_params},
        #     {"params": [self._translation], "lr": lrs["trans_lr"]},
        # ]
        # smpl_body_optim = torch.optim.SGD(params, lr=lrs["smpl_lr"])

        # OFFSETS
        params = [{"params": self._vert_feats}]
        params.append({"params": self._offset_mlp.parameters()})
        offset_optim = torch.optim.Adam(params, lr=lrs["offset_lr"])

        # TEXTURE optimizer
        params = [
            {"params": list(self._texture.parameters()) + list(self._normal_encoder.parameters())},
            {"params": self._explFeatures.parameters()},
        ]
        tex_optim = torch.optim.Adam(params, lr=lrs["tex_lr"], weight_decay=lrs["texture_weight_decay"])

        # # JOINT smpl_body
        # joint_smpl_body_params = smpl_params
        # # translation gets smaller lr
        # params = [
        #     {"params": joint_smpl_body_params},
        #     {"params": [self._translation], "lr": lrs["trans_lr"]},
        # ]
        # joint_smpl_body_optim = torch.optim.SGD(params, lr=lrs["smpl_lr"])



        return [
            # {"optimizer": smpl_body_optim},
            {"optimizer": offset_optim},
            {"optimizer": tex_optim},
            # {"optimizer": joint_smpl_body_optim},
        ]