import torch
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from scipy.interpolate import griddata as interp_grid
from scipy.ndimage import minimum_filter, maximum_filter
import gradio as gr
from torchvision.transforms import ToTensor
import os

class CameraParams:
    def __init__(self, H: int = 512, W: int = 512):
        self.H = H
        self.W = W
        self.focal = (5.8269e+02, 5.8269e+02)
        self.fov = (2*np.arctan(self.W / (2*self.focal[0])), 2*np.arctan(self.H / (2*self.focal[1])))
        self.K = np.array([
            [self.focal[0], 0., self.W/2],
            [0., self.focal[1], self.H/2],
            [0.,            0.,       1.],
        ]).astype(np.float32)

def tilt_pan(deg, num_frames, mode):
    degsum = deg
    # 512x512
    pre_angle = 60
    pre_num = 4
    if mode in ['left','right']:
        if mode == 'left':
            pre = np.linspace(0, -pre_angle, pre_num)
            thlist = np.concatenate((pre, np.linspace(0, -degsum, num_frames)))
        elif mode == 'right':
            pre = np.linspace(0, pre_angle, pre_num)
            thlist = np.concatenate((pre, np.linspace(0, degsum, num_frames)))
        philist = np.zeros_like(thlist)
    elif mode in ['up','down']:
        if mode == 'up':
            pre = np.linspace(0, -pre_angle, pre_num)
            philist = np.concatenate((pre, np.linspace(0, -degsum, num_frames)))
        elif mode == 'down':
            pre = np.linspace(0, pre_angle, pre_num)
            philist = np.concatenate((pre, np.linspace(0, degsum, num_frames)))
        thlist = np.zeros_like(philist)
    assert len(thlist) == len(philist)
    zero_index = len(pre)
    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.zeros((3,1))
    return render_poses, zero_index

def roll(deg, num_frames, mode):
    degsum = deg
    if mode == 'anticlockwise':
        galist = np.linspace(0, degsum, num_frames)
    elif mode == 'clockwise':
        galist = np.linspace(0, -degsum, num_frames)
    render_poses = np.zeros((len(galist), 3, 4))
    for i in range(len(galist)):
        ga = galist[i]
        render_poses[i,:3,:3] = np.array([[np.cos(ga/180*np.pi), -np.sin(ga/180*np.pi), 0], [np.sin(ga/180*np.pi), np.cos(ga/180*np.pi), 0], [0, 0, 1]])
        render_poses[i,:3,3:4] = np.zeros((3,1))
    zero_index = 0
    return render_poses, zero_index


def pedestal_truck(dis, num_frames, mode):
    pre_dis = 2
    pre_num = 5
    if mode in ['right','down']:
        pre = np.linspace(0, -pre_dis, pre_num)
        movement = np.concatenate((pre, np.linspace(0, -dis, num_frames)))
    elif mode in ['left','up']:
        pre = np.linspace(0, pre_dis, pre_num)
        movement = np.concatenate((pre, np.linspace(0, dis, num_frames)))
    render_poses = np.zeros((len(movement), 3, 4))
    for i in range(len(movement)):
        render_poses[i,:3,:3] = np.eye(3)
        if mode in ['right','left']:
            render_poses[i,:3,3:4] = np.array([[movement[i]], [0], [0]])
        elif mode in ['up','down']:
            render_poses[i,:3,3:4] = np.array([[0], [movement[i]], [0]])
    zero_index = len(pre)
    return render_poses, zero_index

def zoom(dis, num_frames, mode):
    if mode == 'out':
        pre_dis =  1
        pre_num = 2
    elif mode == 'in':
        pre_dis = 1
        pre_num = 2
    if mode == 'out':
        pre = np.linspace(0, pre_dis, pre_num)
        movement = np.concatenate((pre, np.linspace(0, dis, num_frames)))
    elif mode == 'in':
        pre = np.linspace(0, -pre_dis, pre_num)
        movement = np.concatenate((pre, np.linspace(0, -dis, num_frames)))
    render_poses = np.zeros((len(movement), 3, 4))
    for i in range(len(movement)):
        render_poses[i,:3,:3] = np.eye(3)
        render_poses[i,:3,3:4] = np.array([[0], [0], [movement[i]]])
    zero_index = len(pre)
    return render_poses, zero_index

def hybrid_out_left_up_down(dis, num_frames, mode):
    zoom_mode = 'out'
    zoom_dis = 2
    zoom_num_frames = num_frames
    zoom_pre_dis = 1
    zoom_pre_num = 2
    move_dis = 30
    move_num_frames = num_frames
    move_pre_dis = 30
    move_pre_num = 2
    if zoom_mode == 'out':
        zoom_pre = np.linspace(0, zoom_pre_dis, zoom_pre_num)
        zoom_movement = np.concatenate((zoom_pre, np.linspace(0, zoom_dis, zoom_num_frames)))
    elif zoom_mode == 'in':
        zoom_pre = np.linspace(0, -zoom_pre_dis, zoom_pre_num)
        zoom_movement = np.concatenate((zoom_pre, np.linspace(0, -zoom_dis, zoom_num_frames)))

    move_pre = np.linspace(0, move_pre_dis, move_pre_num)
    thlist = np.concatenate((move_pre, np.linspace(0, move_dis, move_num_frames)))

    move_pre = np.linspace(0, move_pre_dis, move_pre_num)
    philist = np.concatenate((move_pre, np.linspace(0, move_dis, move_num_frames)))
    render_poses = np.zeros((len(philist), 3, 4))


    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.array([[zoom_movement[i]], [zoom_movement[i]], [zoom_movement[i]]])
    assert len(zoom_pre) == len(move_pre)
    zero_index = len(zoom_pre)
    return render_poses, zero_index

def hybrid_in_then_up(dis, num_frames, mode):
    zoom_dis = 2
    zoom_num_frames = num_frames
    zoom_pre_dis = 1
    zoom_pre_num = 2
    zoom_pre = np.linspace(0, -zoom_pre_dis, zoom_pre_num)
    zoom_movement = np.concatenate((zoom_pre, np.linspace(0, -zoom_dis, zoom_num_frames)))
    render_poses = np.zeros((len(zoom_movement), 3, 4))

    for i in range(len(zoom_movement)):
        render_poses[i,:3,:3] = np.eye(3)
        if i < len(zoom_pre):
            render_poses[i,:3,3:4] = np.array([[0], [-zoom_movement[i]], [0]])
        else:
            mem = (len(zoom_pre) + zoom_num_frames//2)
            if i <  mem:
                render_poses[i,:3,3:4] = np.array([[0], [0], [zoom_movement[i]]])
            else:
                fix=zoom_movement[mem-1]
                render_poses[i,:3,3:4] = np.array([[0], [-zoom_movement[i-mem+2]], [fix]])
    zero_index = len(zoom_pre)
    return render_poses, zero_index

def rotate(deg, num_frames, mode, center_depth):
    degsum = deg
    if mode == 'clockwise':
        thlist = np.linspace(0, degsum, num_frames)
    elif mode == 'anticlockwise':
        thlist = np.linspace(0, -degsum, num_frames)
    phi = 0
    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        # d = 4.3 # manual central point for arc / you can change this value
        d = center_depth
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.array([d*np.sin(th/180*np.pi), 0, d-d*np.cos(th/180*np.pi)]).reshape(3,1) + np.array([0, d*np.sin(phi/180*np.pi), d-d*np.cos(phi/180*np.pi)]).reshape(3,1)# Transition vector
    return render_poses, 0

def get_pcdGenPoses(pcdgenpath_list, center_depth):

    pcdgenpath = pcdgenpath_list[0]
    if pcdgenpath_list[1] == 'default':
        deg = 1
    else:
        deg = int(pcdgenpath_list[1])
    frame = int(pcdgenpath_list[2])
    zero_index = 0
    if pcdgenpath == 'zoom':
        render_poses, zero_index = zoom(deg,frame,pcdgenpath_list[3])
    elif pcdgenpath in ['tilt','pan']:
        render_poses, zero_index = tilt_pan(deg,frame,pcdgenpath_list[3])
    elif pcdgenpath in ['pedestal','truck']:
        render_poses, zero_index = pedestal_truck(deg,frame,pcdgenpath_list[3])
    elif pcdgenpath == 'roll':
        render_poses, zero_index = roll(deg,frame,pcdgenpath_list[3])
    elif pcdgenpath == 'rotate':
        render_poses, zero_index = rotate(deg,frame,pcdgenpath_list[3], center_depth)
    elif pcdgenpath == 'hybrid':
        if pcdgenpath_list[3] == 'in_then_up':
            render_poses, zero_index = hybrid_in_then_up(deg,frame,pcdgenpath_list[3])
        elif pcdgenpath_list[3] == 'out_left_up_down':
            render_poses, zero_index = hybrid_out_left_up_down(deg,frame,pcdgenpath_list[3])
    elif pcdgenpath == 'complex':
        render_poses = np.zeros((frame, 3, 4))
        if pcdgenpath_list[3] == 'mode_4':
            trajectories = torch.load('assets/trajectory/complex_4.pth').reshape([16, 3, 4])
            render_poses[zero_index:] = trajectories[:frame]
        else:
            trajectories = torch.load(pcdgenpath_list[3]).reshape([14, 3, 4])
            render_poses[zero_index:] = trajectories[:frame]
    else:
        raise("Invalid pcdgenpath")

    return render_poses, zero_index


class Warper:
    def __init__(self, H, W):
        self.H = H
        self.W = W
        self.cam = CameraParams(self.H, self.W)
        # stable diffusion
        self.rgb_model =  StableDiffusionInpaintPipeline.from_pretrained(
            'runwayml/stable-diffusion-inpainting', revision='fp16', safety_checker=None, torch_dtype=torch.float16).to('cuda')
        self.d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')

    def d(self, im):
        return self.d_model.infer_pil(im)

    def rgb(self, prompt, image, negative_prompt='', generator=None, num_inference_steps=50, mask_image=None):
        image_pil = Image.fromarray(np.round(image * 255.).astype(np.uint8))
        mask_pil = Image.fromarray(np.round((1 - mask_image) * 255.).astype(np.uint8))
        return self.rgb_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            image=image_pil,
            mask_image=mask_pil,
            height = self.H,
            width = self.W,
            # eta = 0.5,
        ).images[0]

    def generate_pcd(self, rgb_cond, prompt, negative_prompt, pcdgen, seed, diff_steps, save_dir=None, save_warps=False, progress=gr.Progress()):
        generator=torch.Generator(device='cuda').manual_seed(-1)
        warped_images = torch.tensor([]).to('cuda')
        image = ToTensor()(rgb_cond)
        image = image * 2.0 - 1.0
        image = image.unsqueeze(0).to('cuda')
        warped_images = torch.concat([warped_images, image])

        w_in, h_in = rgb_cond.size
        image_curr = rgb_cond
        depth_curr = self.d(image_curr)
        center_depth = np.mean(depth_curr[h_in//2-10:h_in//2+10, w_in//2-10:w_in//2+10])
        render_poses, zero_index = get_pcdGenPoses(pcdgen, center_depth)

        H, W, K = self.cam.H, self.cam.W, self.cam.K
        x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels
        edgeN = 2
        edgemask = np.ones((H-2*edgeN, W-2*edgeN))
        edgemask = np.pad(edgemask, ((edgeN,edgeN),(edgeN,edgeN)))

        R0, T0 = render_poses[0,:3,:3], render_poses[0,:3,3:4]
        pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))
        new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32) ## new_pts_coord_world2
        new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)

        pts_coord_world, pts_colors = new_pts_coord_world2.copy(), new_pts_colors2.copy()

        iterable_dream = range(1, len(render_poses))
        for i in iterable_dream:
            R, T = render_poses[i,:3,:3], render_poses[i,:3,3:4]

            pts_coord_cam2 = R.dot(pts_coord_world) + T
            pixel_coord_cam2 = np.matmul(K, pts_coord_cam2)

            valid_idx = np.where(np.logical_and.reduce((pixel_coord_cam2[2]>0, 
                                                        pixel_coord_cam2[0]/pixel_coord_cam2[2]>=0, 
                                                        pixel_coord_cam2[0]/pixel_coord_cam2[2]<=W-1, 
                                                        pixel_coord_cam2[1]/pixel_coord_cam2[2]>=0, 
                                                        pixel_coord_cam2[1]/pixel_coord_cam2[2]<=H-1)))[0]
            pixel_coord_cam2 = pixel_coord_cam2[:2, valid_idx]/pixel_coord_cam2[-1:, valid_idx]
            round_coord_cam2 = np.round(pixel_coord_cam2).astype(np.int32)

            x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
            grid = np.stack((x,y), axis=-1).reshape(-1,2)
            image2 = interp_grid(pixel_coord_cam2.transpose(1,0), pts_colors[valid_idx], grid, method='linear', fill_value=0).reshape(H,W,3)
            image2 = edgemask[...,None]*image2 + (1-edgemask[...,None])*np.pad(image2[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')

            round_mask2 = np.zeros((H,W), dtype=np.float32)
            round_mask2[round_coord_cam2[1], round_coord_cam2[0]] = 1

            round_mask2 = maximum_filter(round_mask2, size=(9,9), axes=(0,1))
            image2 = round_mask2[...,None]*image2 + (1-round_mask2[...,None])*(-1)

            mask2 = minimum_filter((image2.sum(-1)!=-3)*1, size=(11,11), axes=(0,1))
            image2 = mask2[...,None]*image2 + (1-mask2[...,None])*0

            mask_hf = np.abs(mask2[:H-1, :W-1] - mask2[1:, :W-1]) + np.abs(mask2[:H-1, :W-1] - mask2[:H-1, 1:])
            mask_hf = np.pad(mask_hf, ((0,1), (0,1)), 'edge')
            mask_hf = np.where(mask_hf < 0.3, 0, 1)
            border_valid_idx = np.where(mask_hf[round_coord_cam2[1], round_coord_cam2[0]] == 1)[0]

            image_curr = self.rgb(
                prompt=prompt, image=image2,
                negative_prompt=negative_prompt[0], generator=generator, num_inference_steps=diff_steps,
                mask_image=mask2,
            )
            depth_curr = self.d(image_curr)

            if i-zero_index >= 1:
                image = ToTensor()(image_curr)
                image = image * 2.0 - 1.0
                image = image.unsqueeze(0).to('cuda')
                warped_images = torch.concat([warped_images, image])
                if save_warps == True:
                    os.makedirs(save_dir, exist_ok=True)
                    image_curr.save(f'{save_dir}/{i-zero_index}_concat.png')
            else:
                if save_warps == True:
                    os.makedirs(save_dir, exist_ok=True)
                    image_curr.save(f'{save_dir}/{i-zero_index}.png')


            with torch.enable_grad():
                t_z2 = torch.tensor(depth_curr)
                sc = torch.ones(1).float().requires_grad_(True)
                optimizer = torch.optim.Adam(params=[sc], lr=0.001)

                for idx in range(100):
                    trans3d = torch.tensor([[sc,0,0,0], [0,sc,0,0], [0,0,sc,0], [0,0,0,1]]).requires_grad_(True)
                    coord_cam2 = torch.matmul(torch.tensor(np.linalg.inv(K)), torch.stack((torch.tensor(x)*t_z2, torch.tensor(y)*t_z2, 1*t_z2), axis=0)[:,round_coord_cam2[1], round_coord_cam2[0]].reshape(3,-1))
                    coord_world2 = (torch.tensor(np.linalg.inv(R)).float().matmul(coord_cam2) - torch.tensor(np.linalg.inv(R)).float().matmul(torch.tensor(T).float()))
                    coord_world2_warp = torch.cat((coord_world2, torch.ones((1,valid_idx.shape[0]))), dim=0)
                    coord_world2_trans = torch.matmul(trans3d, coord_world2_warp)
                    coord_world2_trans = coord_world2_trans[:3] / coord_world2_trans[-1]
                    loss = torch.mean((torch.tensor(pts_coord_world[:,valid_idx]).float() - coord_world2_trans)**2)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()    

            with torch.no_grad():
                coord_cam2 = torch.matmul(torch.tensor(np.linalg.inv(K)), torch.stack((torch.tensor(x)*t_z2, torch.tensor(y)*t_z2, 1*t_z2), axis=0)[:,round_coord_cam2[1, border_valid_idx], round_coord_cam2[0, border_valid_idx]].reshape(3,-1))
                coord_world2 = (torch.tensor(np.linalg.inv(R)).float().matmul(coord_cam2) - torch.tensor(np.linalg.inv(R)).float().matmul(torch.tensor(T).float()))
                coord_world2_warp = torch.cat((coord_world2, torch.ones((1, border_valid_idx.shape[0]))), dim=0)
                coord_world2_trans = torch.matmul(trans3d, coord_world2_warp)
                coord_world2_trans = coord_world2_trans[:3] / coord_world2_trans[-1]
            trans3d = trans3d.detach().numpy()


            pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]]
            camera_origin_coord_world2 = - np.linalg.inv(R).dot(T).astype(np.float32) # 3, 1
            new_pts_coord_world2 = (np.linalg.inv(R).dot(pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)
            new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
            new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
            new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
            new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]

            vector_camorigin_to_campixels = coord_world2_trans.detach().numpy() - camera_origin_coord_world2
            vector_camorigin_to_pcdpixels = pts_coord_world[:,valid_idx[border_valid_idx]] - camera_origin_coord_world2

            compensate_depth_coeff = np.sum(vector_camorigin_to_pcdpixels * vector_camorigin_to_campixels, axis=0) / np.sum(vector_camorigin_to_campixels * vector_camorigin_to_campixels, axis=0) # N_correspond
            compensate_pts_coord_world2_correspond = camera_origin_coord_world2 + vector_camorigin_to_campixels * compensate_depth_coeff.reshape(1,-1)

            compensate_coord_cam2_correspond = R.dot(compensate_pts_coord_world2_correspond) + T
            homography_coord_cam2_correspond = R.dot(coord_world2_trans.detach().numpy()) + T

            compensate_depth_correspond = compensate_coord_cam2_correspond[-1] - homography_coord_cam2_correspond[-1] # N_correspond
            compensate_depth_zero = np.zeros(4)
            compensate_depth = np.concatenate((compensate_depth_correspond, compensate_depth_zero), axis=0)  # N_correspond+4

            pixel_cam2_correspond = pixel_coord_cam2[:, border_valid_idx] # 2, N_correspond (xy)
            pixel_cam2_zero = np.array([[0,0,W-1,W-1],[0,H-1,0,H-1]])
            pixel_cam2 = np.concatenate((pixel_cam2_correspond, pixel_cam2_zero), axis=1).transpose(1,0) # N+H, 2

            masked_pixels_xy = np.stack(np.where(1-mask2), axis=1)[:, [1,0]]
            new_depth_linear, new_depth_nearest = interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy), interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy, method='nearest')
            new_depth = np.where(np.isnan(new_depth_linear), new_depth_nearest, new_depth_linear)

            pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]]
            x_nonmask, y_nonmask = x.reshape(-1)[np.where(1-mask2.reshape(-1))[0]], y.reshape(-1)[np.where(1-mask2.reshape(-1))[0]]
            compensate_pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x_nonmask*new_depth, y_nonmask*new_depth, 1*new_depth), axis=0))
            new_warp_pts_coord_cam2 = pts_coord_cam2 + compensate_pts_coord_cam2

            new_pts_coord_world2 = (np.linalg.inv(R).dot(new_warp_pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)
            new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
            new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
            new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
            new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]

            pts_coord_world = np.concatenate((pts_coord_world, new_pts_coord_world2), axis=-1) ### Same with inv(c2w) * cam_coord (in homogeneous space)
            pts_colors = np.concatenate((pts_colors, new_pts_colors2), axis=0)

        return warped_images


    def generate_pcd_3d(self, rgb_cond, prompt, negative_prompt, pcdgen, seed, diff_steps, save_dir, save_warps, progress=gr.Progress()):
        generator=torch.Generator(device='cuda').manual_seed(-1)
        warped_images = torch.tensor([]).to('cuda')
        image = ToTensor()(rgb_cond.resize((256,256)))
        image = image * 2.0 - 1.0
        image = image.unsqueeze(0).to('cuda')
        warped_images = torch.concat([warped_images, image])

        w_in, h_in = rgb_cond.size
        image_curr = rgb_cond
        depth_curr = self.d(image_curr)
        center_depth = np.mean(depth_curr[h_in//2-10:h_in//2+10, w_in//2-10:w_in//2+10])
        render_poses, zero_index = get_pcdGenPoses(pcdgen, center_depth)

        H, W, K = self.cam.H, self.cam.W, self.cam.K
        x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels
        edgeN = 2
        edgemask = np.ones((H-2*edgeN, W-2*edgeN))
        edgemask = np.pad(edgemask, ((edgeN,edgeN),(edgeN,edgeN)))

        R0, T0 = render_poses[0,:3,:3], render_poses[0,:3,3:4]
        pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))
        new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32) ## new_pts_coord_world2
        new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.) ## new_pts_colors2

        pts_coord_world, pts_colors = new_pts_coord_world2.copy(), new_pts_colors2.copy()

        iterable_dream = range(1, len(render_poses))
        for i in iterable_dream:
            R, T = render_poses[i,:3,:3], render_poses[i,:3,3:4]
            pts_coord_cam2 = R.dot(pts_coord_world) + T
            pixel_coord_cam2 = np.matmul(K, pts_coord_cam2)
            valid_idx = np.where(np.logical_and.reduce((pixel_coord_cam2[2]>0, 
                                                        pixel_coord_cam2[0]/pixel_coord_cam2[2]>=0, 
                                                        pixel_coord_cam2[0]/pixel_coord_cam2[2]<=W-1, 
                                                        pixel_coord_cam2[1]/pixel_coord_cam2[2]>=0, 
                                                        pixel_coord_cam2[1]/pixel_coord_cam2[2]<=H-1,
                                                        # )))[0] # 2d scene
                                                        pts_colors.sum(-1) != 3)))[0]
            pixel_coord_cam2 = pixel_coord_cam2[:2, valid_idx]/pixel_coord_cam2[-1:, valid_idx] 
            round_coord_cam2 = np.round(pixel_coord_cam2).astype(np.int32)

            x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
            grid = np.stack((x,y), axis=-1).reshape(-1,2)
            image2 = interp_grid(pixel_coord_cam2.transpose(1,0), pts_colors[valid_idx], grid, method='linear', fill_value=1).reshape(H,W,3)
            get_mask = interp_grid(pixel_coord_cam2.transpose(1,0), pts_colors[valid_idx], grid, method='linear', fill_value=0).reshape(H,W,3)
            get_mask = np.array(get_mask.sum(-1)==0).astype(int)

            image2 = edgemask[...,None]*image2 + (1-edgemask[...,None])*np.pad(image2[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')

            round_mask2 = np.zeros((H,W), dtype=np.float32)
            round_mask2[round_coord_cam2[1], round_coord_cam2[0]] = 1

            round_mask2 = maximum_filter(round_mask2, size=(9,9), axes=(0,1))
            image2 = round_mask2[...,None]*image2 + (1-round_mask2[...,None])*(-1)
            mask2 = minimum_filter((image2.sum(-1)!=-3)*1, size=(11,11), axes=(0,1))
            image2 = mask2[...,None]*image2 + (1-mask2[...,None])*1
            mask2 = get_mask ^ mask2

            mask_hf = np.abs(mask2[:H-1, :W-1] - mask2[1:, :W-1]) + np.abs(mask2[:H-1, :W-1] - mask2[:H-1, 1:])
            mask_hf = np.pad(mask_hf, ((0,1), (0,1)), 'edge')
            mask_hf = np.where(mask_hf < 0.3, 0, 1)
            border_valid_idx = np.where(mask_hf[round_coord_cam2[1], round_coord_cam2[0]] == 1)[0]

            image_curr = self.rgb(
                prompt=prompt[0], image=image2,
                negative_prompt=negative_prompt[0], generator=generator, num_inference_steps=diff_steps,
                mask_image=mask2,
            )
            depth_curr = self.d(image_curr)

            if i-zero_index >= 1:
                image = ToTensor()(image_curr.resize((256, 256)))
                image = image * 2.0 - 1.0
                image = image.unsqueeze(0).to('cuda')
                warped_images = torch.concat([warped_images, image])
                if save_warps == True:
                    os.makedirs(save_dir, exist_ok=True)
                    image_curr.save(f'{save_dir}/{i-zero_index}_concat.png')
            else:
                if save_warps == True:
                    os.makedirs(save_dir, exist_ok=True)
                    image_curr.save(f'{save_dir}/{i-zero_index}.png')


            with torch.enable_grad():
                t_z2 = torch.tensor(depth_curr)
                sc = torch.ones(1).float().requires_grad_(True)
                optimizer = torch.optim.Adam(params=[sc], lr=0.001)

                for idx in range(100):
                    trans3d = torch.tensor([[sc,0,0,0], [0,sc,0,0], [0,0,sc,0], [0,0,0,1]]).requires_grad_(True)
                    coord_cam2 = torch.matmul(torch.tensor(np.linalg.inv(K)), torch.stack((torch.tensor(x)*t_z2, torch.tensor(y)*t_z2, 1*t_z2), axis=0)[:,round_coord_cam2[1], round_coord_cam2[0]].reshape(3,-1))
                    coord_world2 = (torch.tensor(np.linalg.inv(R)).float().matmul(coord_cam2) - torch.tensor(np.linalg.inv(R)).float().matmul(torch.tensor(T).float()))
                    coord_world2_warp = torch.cat((coord_world2, torch.ones((1,valid_idx.shape[0]))), dim=0)
                    coord_world2_trans = torch.matmul(trans3d, coord_world2_warp)
                    coord_world2_trans = coord_world2_trans[:3] / coord_world2_trans[-1]
                    loss = torch.mean((torch.tensor(pts_coord_world[:,valid_idx]).float() - coord_world2_trans)**2)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()    

            with torch.no_grad():
                coord_cam2 = torch.matmul(torch.tensor(np.linalg.inv(K)), torch.stack((torch.tensor(x)*t_z2, torch.tensor(y)*t_z2, 1*t_z2), axis=0)[:,round_coord_cam2[1, border_valid_idx], round_coord_cam2[0, border_valid_idx]].reshape(3,-1))
                coord_world2 = (torch.tensor(np.linalg.inv(R)).float().matmul(coord_cam2) - torch.tensor(np.linalg.inv(R)).float().matmul(torch.tensor(T).float()))
                coord_world2_warp = torch.cat((coord_world2, torch.ones((1, border_valid_idx.shape[0]))), dim=0)
                coord_world2_trans = torch.matmul(trans3d, coord_world2_warp)
                coord_world2_trans = coord_world2_trans[:3] / coord_world2_trans[-1]
            trans3d = trans3d.detach().numpy()


            pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]]
            camera_origin_coord_world2 = - np.linalg.inv(R).dot(T).astype(np.float32) # 3, 1
            new_pts_coord_world2 = (np.linalg.inv(R).dot(pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)
            new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
            new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
            new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
            new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]

            vector_camorigin_to_campixels = coord_world2_trans.detach().numpy() - camera_origin_coord_world2
            vector_camorigin_to_pcdpixels = pts_coord_world[:,valid_idx[border_valid_idx]] - camera_origin_coord_world2

            compensate_depth_coeff = np.sum(vector_camorigin_to_pcdpixels * vector_camorigin_to_campixels, axis=0) / np.sum(vector_camorigin_to_campixels * vector_camorigin_to_campixels, axis=0) # N_correspond
            compensate_pts_coord_world2_correspond = camera_origin_coord_world2 + vector_camorigin_to_campixels * compensate_depth_coeff.reshape(1,-1)

            compensate_coord_cam2_correspond = R.dot(compensate_pts_coord_world2_correspond) + T
            homography_coord_cam2_correspond = R.dot(coord_world2_trans.detach().numpy()) + T

            compensate_depth_correspond = compensate_coord_cam2_correspond[-1] - homography_coord_cam2_correspond[-1] # N_correspond
            compensate_depth_zero = np.zeros(4)
            compensate_depth = np.concatenate((compensate_depth_correspond, compensate_depth_zero), axis=0)  # N_correspond+4

            pixel_cam2_correspond = pixel_coord_cam2[:, border_valid_idx] # 2, N_correspond (xy)
            pixel_cam2_zero = np.array([[0,0,W-1,W-1],[0,H-1,0,H-1]])
            pixel_cam2 = np.concatenate((pixel_cam2_correspond, pixel_cam2_zero), axis=1).transpose(1,0) # N+H, 2

            masked_pixels_xy = np.stack(np.where(1-mask2), axis=1)[:, [1,0]]
            new_depth_linear, new_depth_nearest = interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy), interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy, method='nearest')
            new_depth = np.where(np.isnan(new_depth_linear), new_depth_nearest, new_depth_linear)

            pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]]
            x_nonmask, y_nonmask = x.reshape(-1)[np.where(1-mask2.reshape(-1))[0]], y.reshape(-1)[np.where(1-mask2.reshape(-1))[0]]
            compensate_pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x_nonmask*new_depth, y_nonmask*new_depth, 1*new_depth), axis=0))
            new_warp_pts_coord_cam2 = pts_coord_cam2 + compensate_pts_coord_cam2

            new_pts_coord_world2 = (np.linalg.inv(R).dot(new_warp_pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)
            new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
            new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
            new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
            new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]

            pts_coord_world = np.concatenate((pts_coord_world, new_pts_coord_world2), axis=-1)
            pts_colors = np.concatenate((pts_colors, new_pts_colors2), axis=0)

        return warped_images
