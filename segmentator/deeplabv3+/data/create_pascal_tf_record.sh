python3 create_pascal_tf_record.py --data_dir ./data_dataset_voc \
	                              --output_path ./dataset \
	                              --train_data_list  ./dataset/train.txt \
								  --valid_data_list ./dataset/val.txt \
	                              --image_data_dir  JPEGImages \
                                  --label_data_dir  SegmentationClass \
