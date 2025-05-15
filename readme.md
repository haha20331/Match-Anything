# Clone from https://github.com/JiehongLin/SAM-6D
# HackMD
* https://hackmd.io/YLbxjY-2RsWmLFwpILgSrA
# Requirements
* GPU: NVIDIA GeForce RTX 2080 (8192 MB)
* SAM-6D/readme 內有docker image
* 進入docker和conda環境後，跟隨SAM-6D/prepare.sh安裝 ISM pretrained model即可(可能需另外裝wget套件)，另外兩個用不到

# Inference
* Inference code: SAM-6D/Instance_Segmentation_Model/match_anything.py
    ```
    cd SAM-6D/Instance_Segmentation_Model
    python match_anything.py --config MA_config.yaml
    ```
* Inference config: SAM-6D/MA_config.yaml，裡面有註解全部的超參數

# Input and output
## Input
* 格式: RGB image (png, jpg)
* target_img 為目標場景的rgb圖，放在SAM-6D/Instance_Segmentation_Model/Example/inputs/target_img
* template 為matching object 的template，放在SAM-6D/Instance_Segmentation_Model/Example/inputs/template
    * 底下子資料夾為object名稱，內部可放多張RGB image作為template使用

## Output
* 格式: .json, .png，放在SAM-6D/Instance_Segmentation_Model/Example/outputs
* detection_ism.json: 紀錄每個SAM proposal和他的matching score
* vis_ism.png: 最高分的matching對象segment結果
* object_mask: 最高分的matching對象segment結果的mask，受config內fat_mask影響，希望能透過擷取多的背景幫助後續夾取和碰撞檢測
* vis_match_anything.png: vis_ism.png與原圖得比較
* object_only.png: 最高分的matching對象單獨segment

# Inference time and GPU memory
* Template使用9張
    | 階段         | 所需時間 |
    |--------------|----------|
    | 模型初始化   | 10.69s   |
    | matching    | 2.67s    |
    | 視覺化(可不做)| 4.35s    |
    | GPU使用量     | 2GB   |