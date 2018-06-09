python train.py --model_dir ./model/train \
	             --data_dir  ./data/temp/dataset \
				 --batch_size 3 \
				 --train_epochs 100 \
	             --pre_trained_model ./model/pretrain/resnet_v2_101.ckpt \
                 --gpu False
