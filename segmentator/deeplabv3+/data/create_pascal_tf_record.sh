python create_pascal_tf_record.py  --data_dir ./temp/data_dataset_voc \
	                               --output_path ./temp/dataset \
	                               --train_data_list  ./temp/dataset/train.txt \
								   --valid_data_list ./temp/dataset/val.txt \
	                               --image_data_dir  JPEGImages \
                                   --label_data_dir  SegmentationClass \
