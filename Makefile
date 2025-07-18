dev = $(shell nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print NR-1,$$1}' | sort -k2 -n | head -1 | awk '{print $$1}' )

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
	CUDA_VISIBLE_DEVICES=${dev} python inference_on_a_image.py \
		-c config_model/UniPose_SwinT.py \
		-p weights/unipose_swint.pth \
		-i inputs/Cat-and-Dog-People-1.jpg \
		-o outputs \
		-t 'person,cat,dog,face,hand' \
		-k 'person,AP10K,AP10K,face,hand' \
		--box_threshold 0.05 \
		--iou_threshold 0.3
		
	CUDA_VISIBLE_DEVICES=${dev} python inference_on_a_image.py \
		-c config_model/UniPose_SwinT.py \
		-p weights/unipose_swint.pth \
		-i inputs/Cat-and-Dog-People-1.jpg \
		-o outputs \
		-t 'cat,dog' \
		-k 'AP10K,AP10K' \
		--box_threshold 0.0 \
		--iou_threshold 0.3

	# CUDA_VISIBLE_DEVICES=${dev} python inference_on_a_image.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i inputs/k5-others \
	# 	-o outputs/k5-others \
	# 	-t "person,table,bed,chair,sofa,swivelchair" \
	# 	-k "person,table,bed,chair,sofa,swivelchair" \
	# 	--box_threshold 0.3 \
	# 	--iou_threshold 0.3

	# CUDA_VISIBLE_DEVICES=${dev} python inference_on_a_image.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i inputs/k5-others \
	# 	-o outputs/k5-others \
	# 	-t "table,bed,chair,sofa,swivelchair" \
	# 	-k "table,bed,chair,sofa,swivelchair" \
	# 	--box_threshold 0.3 \
	# 	--iou_threshold 0.3


	# CUDA_VISIBLE_DEVICES=${dev} python inference_on_a_image.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i inputs/k5-others \
	# 	-o outputs/k5-others \
	# 	-t "chair" \
	# 	-k "chair" \
	# 	--box_threshold 0.3 \
	# 	--iou_threshold 0.3


test:
	CUDA_VISIBLE_DEVICES=${dev} python test.py \
		-c config_model/UniPose_SwinT.py \
		-p weights/unipose_swint.pth \
		-i input.json \
		-d inputs/visualization \
		-t person,cat,dog,face,hand \
		-k person,AP10K,AP10K,face,hand \
		-o outputs/Cat-and-Dog-People-1_{person,cat,dog,face,hand}.json \
		--iou_threshold 0.3 \
	
	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/k5-test/copypaste_k4v1.json \
	# 	-d data/k5-test/imgs \
	# 	-t table,bed,chair,sofa \
	# 	-k table,bed,chair,sofa \
	# 	-o outputs/k5-test/copypaste_k4v1_{table_bed_chair_sofa}.json \
	
	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/Keypoint-5/table_test.json \
	# 	-d data/UniKPT/Keypoint-5 \
	# 	-t table \
	# 	-k table \
	# 	-o outputs/table_test.json \

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/keypoint-5/ann5/bed.json \
	# 	-d data/keypoint-5 \
	# 	-t bed \
	# 	-k bed \
	# 	-o outputs/bed_test.json \

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/Keypoint-5/chair_test.json \
	# 	-d data/UniKPT/Keypoint-5 \
	# 	-t chair \
	# 	-o outputs/chair_test.json \
		
	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/Keypoint-5/sofa_test.json \
	# 	-d data/UniKPT/Keypoint-5 \
	# 	-t sofa \
	# 	-o outputs/sofa_test.json \

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/Keypoint-5/swivelchair_test.json \
	# 	-d data/UniKPT/Keypoint-5 \
	# 	-t swivelchair \
	# 	-o outputs/swivelchair_test.json \

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i /comp_robot/cv_public_dataset/Human_Data/unipose_data/ap-10k/annotations/ap10k-val-split1.json \
	# 	-d data/UniKPT/AP-10K/imgs \
	# 	-t "antelope,argali sheep,bison,buffalo,cow,sheep,arctic fox,dog,fox,wolf,beaver,alouatta,monkey,noisy night monkey,spider monkey,uakari,deer,moose,hamster,elephant,horse,zebra,bobcat,cat,cheetah,jaguar,king cheetah,leopard,lion,panther,snow leopard,tiger,giraffe,hippo,chimpanzee,gorilla,orangutan,rabbit,skunk,mouse,rat,otter,weasel,raccoon,rhino,marmot,squirrel,pig,mole,black bear,brown bear,panda,polar bear,bat" \
	# 	-k animal_in_AP10K \
	# 	-o outputs/0pct_0pct/ap10k-val-split1.json \
	# 	--box_threshold 0.0 \
	# 	--iou_threshold 0.0

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i /comp_robot/cv_public_dataset/Human_Data/unipose_data/ap-10k/annotations/ap10k-test-split1.json \
	# 	-d data/UniKPT/AP-10K/imgs \
	# 	-t "antelope,argali sheep,bison,buffalo,cow,sheep,arctic fox,dog,fox,wolf,beaver,alouatta,monkey,noisy night monkey,spider monkey,uakari,deer,moose,hamster,elephant,horse,zebra,bobcat,cat,cheetah,jaguar,king cheetah,leopard,lion,panther,snow leopard,tiger,giraffe,hippo,chimpanzee,gorilla,orangutan,rabbit,skunk,mouse,rat,otter,weasel,raccoon,rhino,marmot,squirrel,pig,mole,black bear,brown bear,panda,polar bear,bat" \
	# 	-k animal_in_AP10K \
	# 	-o outputs/0pct_0pct/ap10k-test-split1.json \
	# 	--box_threshold 0.0 \
	# 	--iou_threshold 0.0

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/APT-36K/apt36k_annotations_animal_test.json \
	# 	-d data/UniKPT/APT-36K \
	# 	-t animal_36K \
	# 	-o outputs/apt36k_annotations_test.json \

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/CarFusion/car_keypoints_test.json \
	# 	-d data/UniKPT/CarFusion \
	# 	-t car,person,hand \
	# 	-k car,person,hand \
	# 	-o outputs/car_keypoints_test.json \

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/CUB-200-2011/CUB-200-2011_val_bird.json \
	# 	-d data/UniKPT/CUB-200-2011 \
	# 	-t bird \
	# 	-o outputs/CUB-200-2011_val_bird.json \

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/COCO/person_keypoints_val2017_hasgt2.json \
	# 	-d data/UniKPT/COCO \
	# 	-t person \
	# 	-k person \
	# 	-o outputs/person_keypoints_val2017_hasgt2_{person}.json \
	# 	# --draw

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/OneHand10K/onehand10k_test.json \
	# 	-d data/UniKPT/OneHand10K \
	# 	-t hand,person,car \
	# 	-k hand,person,car \
	# 	-o outputs/onehand10k_test.json \
	# 	# --draw

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/300W/face_landmarks_300w_test.json \
	# 	-d data/UniKPT/300W \
	# 	-t face \
	# 	-k face \
	# 	-o outputs/face_landmarks_300w_test.json \
	# 	# --draw

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/HumanArt/test_humanart.json \
	# 	-d data/UniKPT \
	# 	-t person \
	# 	-o outputs/humanart_test.json \
	# 	--box_threshold 0.0 \
	# 	--iou_threshold 0.0
	# 	# --draw

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/HumanArt/validation_humanart_re.json \
	# 	-d data/UniKPT \
	# 	-t person \
	# 	-k person \
	# 	-o outputs/humanart_validation_re_{person}.json \
	# 	--box_threshold 0.0 \
	# 	--iou_threshold 0.0
	# 	# --draw

	# CUDA_VISIBLE_DEVICES=${dev} python test.py \
	# 	-c config_model/UniPose_SwinT.py \
	# 	-p weights/unipose_swint.pth \
	# 	-i data/UniKPT/HumanArt/validation_humanart.json \
	# 	-d data/UniKPT \
	# 	-t person \
	# 	-o outputs/humanart_validation.json \
	# 	--box_threshold 0.0 \
	# 	--iou_threshold 0.0
		# --draw


	

	
