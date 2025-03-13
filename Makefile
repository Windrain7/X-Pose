dev = $(shell nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print NR-1,$$1}' | sort -k2 -n | head -1 | awk '{print $$1}' )
debug := 0

ifeq ($(debug), 1)
	cmd = python -m debugpy --listen 0.0.0.0:5678 --wait-for-client
else
	cmd = python
endif

setup-data:
	-ln -s /comp_robot/cv_public_dataset/Hand_Data/FreiHand/ data/freihand
	-ln -s /comp_robot/cv_public_dataset/Hand_Data/coco/ data/coco
	-ln -s /comp_robot/cv_public_dataset/Hand_Data/HInt/ data/
	-ln -s /comp_robot/cv_public_dataset/Human_Data/AI_Challenger data/
	-ln -s /comp_robot/zhangyuhong1/mmpose/data/UBody/ data/
	-ln -s /comp_robot/cv_public_dataset/Human_Data/unipose_data/keypoint-5 data/
	-ln -s /comp_robot/cv_public_dataset/Human_Data/unipose_data/carfusion/CarFusion/raw/CarFusion/carfusion data/
	-ln -s /comp_robot/cv_public_dataset/Human_Data/MPII data/
	-ln -s /comp_robot/cv_public_dataset/Human_Data/crowdpose data/
	-ln -s /comp_robot/cv_public_dataset/Human_Data/HumanArt data/
	-ln -s /comp_robot/cv_public_dataset/Human_Data/unipose_data/UniKPT data/ 
	# -ln -s /comp_robot/shock/mirror_home/workspace/mmpose/log

infer:
	CUDA_VISIBLE_DEVICES=${dev} ${cmd} inference_on_a_image.py \
		-c config_model/UniPose_SwinT.py \
		-p weights/unipose_swint.pth \
		-i inputs/car-person \
		-o outputs/car-person \
		-t person \
		--box_threshold 0.3 \
		--iou_threshold 0.6

test:
	# CUDA_VISIBLE_DEVICES=${dev} ${cmd} test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/Keypoint-5/table_test.json \
	# 	-d data/UniKPT/Keypoint-5 \
	# 	-t table \
	# 	-o outputs/table_test.json \

	# CUDA_VISIBLE_DEVICES=${dev} ${cmd} test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/Keypoint-5/bed_test.json \
	# 	-d data/UniKPT/Keypoint-5 \
	# 	-t bed \
	# 	-o outputs/bed_test.json \

	# CUDA_VISIBLE_DEVICES=${dev} ${cmd} test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/Keypoint-5/chair_test.json \
	# 	-d data/UniKPT/Keypoint-5 \
	# 	-t chair \
	# 	-o outputs/chair_test.json \
		
	# CUDA_VISIBLE_DEVICES=${dev} ${cmd} test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/Keypoint-5/sofa_test.json \
	# 	-d data/UniKPT/Keypoint-5 \
	# 	-t sofa \
	# 	-o outputs/sofa_test.json \

	# CUDA_VISIBLE_DEVICES=${dev} ${cmd} test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/Keypoint-5/swivelchair_test.json \
	# 	-d data/UniKPT/Keypoint-5 \
	# 	-t swivelchair \
	# 	-o outputs/swivelchair_test.json \

	# CUDA_VISIBLE_DEVICES=${dev} ${cmd} test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/AP-10K/ap10k-test-split1_animal.json \
	# 	-d data/UniKPT/AP-10K \
	# 	-t animal_in_AP10K \
	# 	-o outputs/ap10k-test-split1_animal.json \

	# CUDA_VISIBLE_DEVICES=${dev} ${cmd} test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/APT-36K/apt36k_annotations_test.json \
	# 	-d data/UniKPT/AP-36K \
	# 	-t animal_36K \
	# 	-o outputs/apt36k_annotations_test.json \

	# CUDA_VISIBLE_DEVICES=${dev} ${cmd} test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/CarFusion/car_keypoints_test.json \
	# 	-d data/UniKPT/CarFusion \
	# 	-t car \
	# 	-o outputs/car_keypoints_test.json \

	# CUDA_VISIBLE_DEVICES=${dev} ${cmd} test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/CUB-200-2011/CUB-200-2011_val_bird.json \
	# 	-d data/UniKPT/CUB-200-2011 \
	# 	-t bird \
	# 	-o outputs/CUB-200-2011_val_bird.json \

	CUDA_VISIBLE_DEVICES=${dev} ${cmd} test.py \
		-c config_model/UniPose_SwinT.py \
		-p weights/unipose_swint.pth \
		-i data/UniKPT/COCO/person_keypoints_val2017_hasgt2.json \
		-d data/UniKPT/COCO \
		-t person \
		-o outputs/person_keypoints_val2017_hasgt2.json \
		--box_threshold 0.0 \
		# --draw

	# CUDA_VISIBLE_DEVICES=${dev} ${cmd} test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/OneHand10K/onehand10k_test.json \
	# 	-d data/UniKPT/OneHand10K \
	# 	-t hand \
	# 	-o outputs/onehand10k_test.json \

	

	
