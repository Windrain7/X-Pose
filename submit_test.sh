#!/bin/bash
#SBATCH --job-name=x-pose-test
#SBATCH --output=x-pose-test_%j.log
#SBATCH --error=x-pose-test_%j.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:hgx:1
#SBATCH --partition=cvr


# export CUDA_HOME=/comp_robot/shock/share/pkgs/cuda-12.1

# cd models/UniPose/ops/
# rm -rf build *.so *.cpp
# python setup.py clean
# python setup.py build install
# cd ../../../

CUDA_VISIBLE_DEVICES=0 python test.py -c config_model/UniPose_SwinT.py -p weights/unipose_swint.pth -i data/UniKPT/HumanArt/validation_humanart.json -d data/UniKPT/HumanArt/.. -t "person,face,hand,antelope,argali sheep,bison,buffalo,cow,sheep,arctic fox,dog,fox,wolf,beaver,alouatta,monkey,noisy night monkey,spider monkey,uakari,deer,moose,hamster,elephant,horse,zebra,bobcat,cat,cheetah,jaguar,king cheetah,leopard,lion,panther,snow leopard,tiger,giraffe,hippo,chimpanzee,gorilla,orangutan,rabbit,skunk,mouse,rat,otter,weasel,raccoon,rhino,marmot,squirrel,pig,mole,black bear,brown bear,panda,polar bear,bat,table,bed,chair,sofa,car,face" -k "person,face,hand,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,AP10K,table,bed,chair,sofa,car,face" -o outputs/UniKPT/HumanArt_allinonev1.json
