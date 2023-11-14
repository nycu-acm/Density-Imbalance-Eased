import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from model.model_blocks import Mapping2Dto3D, Identity
from model.template import get_template


class Atlasnet(nn.Module):

    def __init__(self, opt):
        """
        Core Atlasnet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and returns a pointcloud or a mesh
        Author : Thibault Groueix 01.11.2019
        :param opt: 
        """
        super(Atlasnet, self).__init__()
        self.opt = opt
        self.device = opt.device

        # Define number of points per primitives
        self.nb_pts_in_primitive = self.opt.number_points
        self.nb_pts_in_primitive_eval = self.opt.number_points

        if opt.remove_all_batchNorms:
            torch.nn.BatchNorm1d = Identity
            print("Replacing all batchnorms by identities.")

        # Initialize templates
        self.template = [get_template(opt.template_type, device=opt.device) for i in range(0, opt.nb_primitives)]

        # Intialize deformation networks
        self.decoder = nn.ModuleList([Mapping2Dto3D(opt) for i in range(0, opt.nb_primitives)])

    def forward(self, latent_vector, pts, train=True):
        """
        Deform points from self.template using the embedding latent_vector
        :param latent_vector: an opt.bottleneck size vector encoding a 3D shape or an image. size : batch, bottleneck
        :return: A deformed pointcloud os size : batch, nb_prim, num_point, 3
        """
        # Sample points in the patches
        # input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive,
        #                                                     device=latent_vector.device)
        #                 for i in range(self.opt.nb_primitives)]
        # print(latent_vector.shape)   # B 16384, 1024
        Batch = latent_vector.size(0)
        # print(Batch)
        latent_vector = latent_vector.reshape(-1,1024)
        if train:
            input_points = [self.template[i].get_random_points(
                torch.Size((1, self.template[i].dim, self.nb_pts_in_primitive)),
                latent_vector.device) for i in range(self.opt.nb_primitives)]
        else:
            # print(latent_vector.device)
            input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive_eval,
                                                                device=latent_vector.device)
                            for i in range(self.opt.nb_primitives)]
        # input_points = [self.template[i].get_random_points(
        # torch.Size((1, self.template[i].dim, self.nb_pts_in_primitive)),
        # latent_vector.device) for i in range(self.opt.nb_primitives)]

        # Deform each patch
        output_points = [self.decoder[i](input_points[i], latent_vector.unsqueeze(2)) for i in
                                   range(0, self.opt.nb_primitives)]
        # print(output_points[0].shape)   # B*16384 3 4  -->  B, 16384*4, 3
        
        # final_points = output_points[0].permute(0,2,1).reshape(Batch,-1,3)
        final_points = output_points[0].permute(0,2,1)    # B*16384 4 3
        # final_points = final_points.reshape(Batch,-1,3)    # B, 16384*4, 3
        # final_points = torch.tanh(final_points)
        
        # add_pts = pts.repeat(1,1,self.nb_pts_in_primitive).reshape(-1,self.nb_pts_in_primitive,3) 
        # print(pts)
        # final_points = final_points + add_pts     # B*16384 4 3
        # final_points = final_points      # B*16384 4 3
        # final_points = final_points.reshape(Batch, -1,4,3).transpose(1,0)
        
        
        # print(final_points.shape)

        # Return the deformed pointcloud
        return final_points.contiguous(), pts  # B, 16384, 3

    def generate_mesh(self, latent_vector):
        assert latent_vector.size(0)==1, "input should have batch size 1!"
        import pymesh
        input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive, latent_vector.device)
                        for i in range(self.opt.nb_primitives)]
        input_points = [input_points[i] for i in range(self.opt.nb_primitives)]

        # Deform each patch
        output_points = [self.decoder[i](input_points[i], latent_vector.unsqueeze(2)).squeeze() for i in
                         range(0, self.opt.nb_primitives)]

        output_meshes = [pymesh.form_mesh(vertices=output_points[i].transpose(1, 0).contiguous().cpu().numpy(),
                                          faces=self.template[i].mesh.faces)
                         for i in range(self.opt.nb_primitives)]

        # Deform return the deformed pointcloud
        mesh = pymesh.merge_meshes(output_meshes)

        return mesh
