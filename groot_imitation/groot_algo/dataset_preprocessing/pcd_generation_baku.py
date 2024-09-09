import open3d as o3d
from groot_imitation.groot_algo.o3d_modules import convert_convention, O3DPointCloud
import pickle
from tqdm import tqdm
import numpy as np
from groot_imitation.groot_algo.misc_utils import get_first_frame_annotation, resize_image_to_same_shape, normalize_pcd
from groot_imitation.groot_algo.point_mae_modules import Group
import sys
import torch
from einops import rearrange
import cv2
import matplotlib.tri as mtri
import plotly.graph_objects as go
import plotly.io as pio

def save_plot(pcd, fig, colors=None):
    if len(pcd) > 1:
        pcd = np.concatenate(pcd, axis=0)
        colors = np.concatenate(colors, axis=0)
    else:
        pcd = pcd[0]
        colors = colors[0]
    ### DATA GENERATION
    # Make parameter spaces radii and angles.
    # Sample data for the point cloud
    x = pcd[:, 0]
    y = pcd[:, 1]
    z = pcd[:, 2]

    if colors is None:
        colors = z

    # Create a 3D scatter plot
    point_cloud = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=1,
            color=colors,  # Set color based on z value for variation
            # colorscale='Viridis',  # Choose a colorscale
            opacity=0.8
        )
    )

    fig.add_trace(point_cloud)

def save_plot_video(fig):
    # Make 10th trace visible
    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Slider switched to step: " + str(i)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )
    fig.write_html("/fs/cfar-projects/waypoint_rl/BAKU_original/BAKU/my-graphs/index.html", auto_open=False)

def full_pcd_fn(cfg, depth, rgb_img_input, mask_img_input, first_frame_annotation, intrinsic_matrix, extrinsic_matrix, fig, should_save_plot=False):
    if cfg.is_real_robot:
        rgb_img = cv2.cvtColor(rgb_img_input, cv2.COLOR_BGR2RGB)
    else:
        rgb_img = rgb_img_input

    depth_img = depth
    mask_img = resize_image_to_same_shape(mask_img_input, rgb_img)
    depth_img = resize_image_to_same_shape(depth_img, rgb_img)
    masked_depth_img = depth_img.copy().astype(np.float32)

    xyz_list = []
    colors_list = []

    for mask_idx in range(1, 2):
        #TODO: PUT IN REAL DEPTH IMAGE
        # masked_depth_img = depth_model.infer_image(rgb_img)
        o3d_pc = O3DPointCloud(max_points=50000)
        o3d_pc.create_from_rgbd(np.ascontiguousarray(rgb_img), masked_depth_img, intrinsic_matrix)
        #TODO: Comment back in when we have extrinsic matrix
        # o3d_pc.transform(extrinsic_matrix)
        o3d_pc.preprocess(use_rgb=True)
        points = o3d_pc.get_points()
        colors = o3d_pc.get_colors()
        xyz_list.append(points)
        colors_list.append(colors)

    if should_save_plot:
        plot = save_plot(xyz_list, fig, colors_list)
        return np.stack(xyz_list, axis=0), plot
    else:
        return np.stack(xyz_list, axis=0), None


def object_pcd_fn(cfg, depth, rgb_img_input, mask_img_input, first_frame_annotation, intrinsic_matrix, extrinsic_matrix, prev_xyz=None, fig=None, should_save_plot=False):
    if cfg.is_real_robot:
        rgb_img = cv2.cvtColor(rgb_img_input, cv2.COLOR_BGR2RGB)
    else:
        rgb_img = rgb_img_input

    depth_img = depth
    mask_img = resize_image_to_same_shape(mask_img_input, rgb_img)
    depth_img = resize_image_to_same_shape(depth_img, rgb_img)

    xyz_list = []
    colors_list = []

    for mask_idx in range(1, first_frame_annotation.max() + 1):
        #TODO: PUT IN REAL DEPTH IMAGE
        # masked_depth_img = depth_model.infer_image(rgb_img[:, :, ::-1])
        masked_depth_img = depth_img.copy()
        binary_mask = np.where(mask_img == mask_idx, 1, 0)
        masked_depth_img[binary_mask == 0] = -1
        masked_depth_img = masked_depth_img.astype(np.float32)

        total_mask = sum(sum(masked_depth_img != -1))

        o3d_pc = O3DPointCloud()
        # o3d_pc.create_from_depth(masked_depth_img, intrinsic_matrix)
        o3d_pc.create_from_rgbd(np.ascontiguousarray(rgb_img), masked_depth_img, intrinsic_matrix)
        #TODO: Comment back in when we have extrinsic matrix
        # o3d_pc.transform(extrinsic_matrix)
        if total_mask > 0:
            o3d_pc.preprocess(use_rgb=False)
            points = o3d_pc.get_points()
            xyz_list.append(points)
            colors = o3d_pc.get_colors()
            colors_list.append(colors)
        else:
            if prev_xyz is not None:
                xyz_list.append(prev_xyz[mask_idx - 1])
                colors_list.append(np.zeros_like(prev_xyz[mask_idx - 1]))
            else:
                xyz_list.append(np.zeros((512, 3)))
                colors_list.append(np.zeros((512, 3)))

    if should_save_plot:
        plot = save_plot(xyz_list, fig, colors_list)
        return np.stack(xyz_list, axis=0), plot
    else:
        return np.stack(xyz_list, axis=0), None

def object_pcd_generation(cfg, save_plots=False, show_color=False):
    # sys.path.append(cfg.depth_path)
    # from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
    # model_configs = {
    #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    # }
    # encoder = 'vitl'
    # dataset = 'hypersim'
    # max_depth = 20
    # depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    # depth_model.load_state_dict(torch.load(f'{cfg.depth_path}/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=cfg.device))
    # depth_model = depth_model.to(cfg.device).eval()


    with open(cfg.dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    with open(f"{cfg.annotation_folder}/dataset_masks.pkl", 'rb') as f:
        dataset_masks = pickle.load(f)

    first_frame, first_frame_annotation = get_first_frame_annotation(cfg.annotation_folder)

    # For Metaworld
    intrinsic_matrix = np.array([[443.40500674, 0.0, 255.5],
                                [0.0, 443.40500674, 255.5],
                                [0.0, 0.0, 1.0]])

    extrinsic_matrix = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 1.0]])

    #For Real Robot
    # intrinsic_matrix = np.array([[608.456,   0.0, 315.041],
    #                             [  0.0, 608.476, 252.704],
    #                             [  0.0,   0.0,   1.0]])

    # extrinsic_matrix = np.array([[ 0.86964333,  0.48952771, -0.0638991 , -0.17677556],
    #    [ 0.17188394, -0.42157098, -0.89035601,  0.24200464],
    #    [-0.46279194,  0.76330894, -0.45075835,  1.22712932],
    #    [ 0.0,  0.0,  0.0,  1.0]])
    
    points_path = f"{cfg.annotation_folder}/xyz.pkl"
    final_xyz = []
    for (idx, demo) in tqdm(enumerate(dataset['observations'])):
        # Create figure
        if save_plots or show_color:
            fig = go.Figure()
        else:
            fig = None  

        images = demo[cfg.pixel_key]
        depths = demo[cfg.depth_key]
        masks = dataset_masks[idx]

        episode_xyz = []
        frames = []
        count = 0
        for (image, mask, depth) in zip(images, masks, depths):
            if show_color:
                points, _ = full_pcd_fn(cfg, depth, image, mask, first_frame_annotation, intrinsic_matrix, extrinsic_matrix, fig, True)
                fig.show()
                breakpoint()
            prev = None if count == 0 else episode_xyz[-1]
            points, plot = object_pcd_fn(cfg, depth, image, mask, first_frame_annotation, intrinsic_matrix, extrinsic_matrix, prev, fig, count % 5 == 0 and save_plots)
            if count % 5 == 0 and save_plots:
                frames.append(plot)
            episode_xyz.append(points)
            count += 1

        episode_xyz = np.stack(episode_xyz, axis=0)
        final_xyz.append(episode_xyz)
        if save_plots:
            save_plot_video(fig)
            breakpoint()
    pickle.dump(final_xyz, open(points_path, 'wb'))

def object_pcd_grouping(cfg):
    group_divider = Group(num_group=cfg.num_group, group_size=cfg.group_size).cuda()
    with open(f"{cfg.annotation_folder}/xyz.pkl", 'rb') as f:
        dataset = pickle.load(f)

    all_centers = []
    all_neighborhoods = []
    for xyz_sequence in dataset:
        normalized_xyz_sequence = normalize_pcd(xyz_sequence, max_array=cfg.max_array, min_array=cfg.min_array)
        B, N, D = xyz_sequence.shape[:-1]

        xyz_tensor = torch.from_numpy(normalized_xyz_sequence).cuda().float()
        xyz_tensor = rearrange(xyz_tensor, "b n d t-> (b n) d t")
        neighborhood, centers = group_divider(xyz_tensor)
        centers = centers.unsqueeze(-2)
        neighborhood = rearrange(neighborhood, "(b n) g d t -> b (n g) d t", b=B, n=N)
        centers = rearrange(centers, "(b n) g d t -> b (n g) d t", b=B, n=N)

        all_centers.append(centers.cpu().numpy())
        all_neighborhoods.append(neighborhood.cpu().numpy())

    pickle.dump(all_centers, open(f"{cfg.annotation_folder}/centers.pkl", 'wb'))
    pickle.dump(all_neighborhoods, open(f"{cfg.annotation_folder}/neighborhoods.pkl", 'wb'))
