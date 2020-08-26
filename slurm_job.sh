#!/bin/bash
#SBATCH --job-name=yolo_o    ## job name
#SBATCH --nodes=1                ## 索取 1 節點
#SBATCH --cpus-per-task=4       ## 該 task 用 32 CPUs
#SBATCH --gres=gpu:1             ## 每個節點索取 8 GPUs
#SBATCH --account=MST108466    ## iService_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gp1d        ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、p4d(最長跑4天)

module purge
module load miniconda3
conda activate torch_a

# 大部分使用 conda 用戶，程式並沒有透過 MPI 溝通，僅用 1 task
# 不需要再添加 srun/mpirun，直接加上你要運行的指令即可

#python -u yoloatt_train.py --epochs=1 --num_workers=8 --batch_size=8 --log_period=10 --save_period=1 --loss_weight=0.001 --log_path=epoch_yolo_test_2 --weight=./weights/yoloatt_v3_w.pth
#python -u dect_obj_3_save.py --out_path=./com_yolo_output --image_folder=./data/common_obj.txt --weights_path=./weights/yoloatt_v3_w.pth --batch_size=8
#python -u yoloatt_train.py --epochs=5 --num_workers=8 --batch_size=8 --log_period=125 --save_period=1 --loss_weight=0.0001 --log_path=epoch_yoloatt_2 --weight=./epoch_yoloatt_1/yoloatt/model/weight_5
python -u yoloatt_train.py --epochs=8 --num_workers=8 --batch_size=8 --log_period=125 --save_period=1 --loss_weight=0.0001 --log_path=epoch_yolo_test_1 --weight=../weights/yoloatt_v3_2_w.pth
