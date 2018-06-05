python3 inference.py --data_dir ./data/data_dataset_voc/JPEGImages \
	                --output_dir ./result \
					--infer_data_list ./data/dataset/sample_images_list.txt \
					--model_dir ./model/train \
					--base_architecture resnet_v2_101 \
					--output_stride 16
                     
