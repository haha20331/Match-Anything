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
from utils.bbox_utils import ResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.inout import load_json, save_json_bop23

inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )


def visualize(rgb, detections, save_path):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = 0.
    for mask_idx, det in enumerate(detections):
        if best_score < det['score']:
            best_score = det['score']
            best_det = detections[mask_idx]

    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det["category_id"]
    temp_id = obj_id - 1

    r = int(255*colors[temp_id][0])
    g = int(255*colors[temp_id][1])
    b = int(255*colors[temp_id][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
    img[edge, :] = 255
    
    # ===== 把原圖中物體區域抓出來 =====
    rgb_np = np.array(rgb).copy()
    object_only = rgb_np.copy()
    object_only[~mask] = 0 
    # ✂️ 取得 mask 的 bounding box
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # +1 是為了包含該點   
    object_only_cropped = object_only[y0:y1, x0:x1]
    object_only_img = Image.fromarray(object_only_cropped)
    object_only_img.save(save_path+"/object_only.png")


    img = Image.fromarray(np.uint8(img))
    img.save(save_path+"/vis_ism.png")
    prediction = Image.open(save_path+"/vis_ism.png")
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def run_inference(segmentor_model, output_dir, template_img, rgb_path, stability_score_thresh, exp_name):
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")
        
    #改完輸入多張圖片，沒額外mask來crop，只做resize and padding
    logging.info("Initializing template")

    # transform 應該根據圖片再多加設計
    gallery_transform = T.Compose([
    T.CenterCrop((512, 512)),     
    ])

    extensions = ['png', 'jpg', 'jpeg']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(template_img, f"*.{ext}")))
    image_paths.sort()  
    templates = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image = gallery_transform(image)
        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        templates.append(image)
    templates = torch.stack(templates)  # shape: [N, H, W, 3]
    templates = templates.permute(0, 3, 1, 2)  # shape: [N, 3, H, W]
    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = ResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_patch_feature(
                    templates).unsqueeze(0).data

    
    # run inference
    rgb = Image.open(rgb_path).convert("RGB")
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    # descriptor_model = DINOv2
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)

    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # compute the appearance score
    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    final_score = (semantic_score + appe_scores) / (2)

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))   
         
    detections.to_numpy()
    save_path = f"{output_dir}/sam6d_results/detection_ism"
    detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path+".json", detections)
    vis_img = visualize(rgb, detections, f"{output_dir}/sam6d_results/{exp_name}")
    vis_img.save(f"{output_dir}/sam6d_results/{exp_name}/vis_match_anything.png")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='fastsam', help="The segmentor model in ISM")
    parser.add_argument("--output_dir", default='/workspace/SAM-6D/SAM-6D/Instance_Segmentation_Model/Example/outputs', nargs="?", help="Path to root directory of the output")
    parser.add_argument("--template_img", default='/workspace/SAM-6D/SAM-6D/Instance_Segmentation_Model/Example/inputs/template/ele_bow', nargs="?", help="Path to root directory of the template image")
    parser.add_argument("--rgb_path", default='/workspace/SAM-6D/SAM-6D/Instance_Segmentation_Model/Example/inputs/target_img/usb.jpg', nargs="?", help="Path to RGB image")
    parser.add_argument("--exp_name", default="no_target_case")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()
    os.makedirs(f"{args.output_dir}/sam6d_results/{args.exp_name}", exist_ok=True)
    run_inference(
        args.segmentor_model, args.output_dir, args.template_img, args.rgb_path, 
        stability_score_thresh=args.stability_score_thresh,exp_name=args.exp_name,
    )