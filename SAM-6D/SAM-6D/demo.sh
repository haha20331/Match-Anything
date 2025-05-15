start_time=$(date +%s)
export CAD_PATH=/workspace/SAM-6D/SAM-6D/Data/Example/obj_000005.ply    
export RGB_PATH=/workspace/SAM-6D/SAM-6D/Data/Example/rgb.png           
export DEPTH_PATH=/workspace/SAM-6D/SAM-6D/Data/Example/depth.png       
export CAMERA_PATH=/workspace/SAM-6D/SAM-6D/Data/Example/camera.json    
export OUTPUT_DIR=/workspace/SAM-6D/SAM-6D/Data/Example/outputs  
# Render CAD templates
cd Render
# blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH #--colorize True 


# Run instance segmentation model
export SEGMENTOR_MODEL=fastsam

cd ../Instance_Segmentation_Model
python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH


# Run pose estimation model
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

cd ../Pose_Estimation_Model
#python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH
python run_inference_custom_HighScoreType.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH


end_time=$(date +%s)
duration=$((end_time - start_time))
echo "總共花了 $duration 秒！"