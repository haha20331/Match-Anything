import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CenterCropResizePad, CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.inout import load_json, save_json_bop23

def run_inference(segmentor_model, output_dir, template_img, cad_path, rgb_path, depth_path, cam_path, stability_score_thresh):
    template_dir = os.path.join(template_img)
    num_templates = len(glob.glob(f"{template_dir}/*.png")) + len(glob.glob(f"{template_dir}/*.jpg"))
    templates = []
    for idx in range(num_templates):
        image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        templates.append(image)
    templates = torch.stack(templates)  # shape: [N, H, W, 3]
    templates = templates.permute(0, 3, 1, 2)  # shape: [N, 3, H, W]
    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = CenterCropResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates)

    # logging.info("Initializing template")


    # template_dir = os.path.join(output_dir, 'templates')
    # num_templates = len(glob.glob(f"{template_dir}/*.npy"))
    # boxes, masks, templates = [], [], []
    # for idx in range(num_templates):
    #     image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
    #     mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
    #     boxes.append(mask.getbbox())

    #     image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
    #     mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
    #     image = image * mask[:, :, None]
    #     templates.append(image)
    #     masks.append(mask.unsqueeze(-1))
    # print(f"模板數量: {len(templates)}")
    # print(f"單張 template shape (before stack): {templates[0].shape}")  # 通常是 [H, W, 3]
    # templates = torch.stack(templates)  # shape: [N, H, W, 3]
    # print(f"template shape after stack: {templates.shape}")
    # templates = templates.permute(0, 3, 1, 2)  # shape: [N, 3, H, W]
    # print(f"template shape after permute: {templates.shape}")
    # masks = torch.stack(masks).permute(0, 3, 1, 2)
    # boxes = torch.tensor(np.array(boxes))
    
    # processing_config = OmegaConf.create(
    #     {
    #         "image_size": 224,
    #     }
    # )
    # proposal_processor = CropResizePad(processing_config.image_size)
    # templates = proposal_processor(images=templates, boxes=boxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='fastsam', help="The segmentor model in ISM")
    parser.add_argument("--output_dir", default='/workspace/SAM-6D/SAM-6D/Data/Example/outputs', nargs="?", help="Path to root directory of the output")
    parser.add_argument("--template_img", default='/workspace/SAM-6D/SAM-6D/Instance_Segmentation_Model/template_img', nargs="?", help="Path to root directory of the template image")
    parser.add_argument("--cad_path", default='/workspace/SAM-6D/SAM-6D/Data/Example/obj_000005.ply', nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--rgb_path", default='/workspace/SAM-6D/SAM-6D/Data/Example/rgb.png', nargs="?", help="Path to RGB image")
    parser.add_argument("--depth_path", default='/workspace/SAM-6D/SAM-6D/Data/Example/depth.png', nargs="?", help="Path to Depth image(mm)")
    parser.add_argument("--cam_path", default='/workspace/SAM-6D/SAM-6D/Data/Example/camera.json', nargs="?", help="Path to camera information")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()
    os.makedirs(f"{args.output_dir}/sam6d_results", exist_ok=True)
    run_inference(
        args.segmentor_model, args.output_dir, args.template_img, args.cad_path, args.rgb_path, args.depth_path, args.cam_path, 
        stability_score_thresh=args.stability_score_thresh,
    )