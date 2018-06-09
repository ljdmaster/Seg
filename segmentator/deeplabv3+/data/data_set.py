import os
import sys

label_dir = "./temp/data_dataset_voc/SegmentationClass"
output_dir = "./temp/dataset"

file_list = os.listdir(label_dir)
num = len(file_list)
ratio = 0.65

with open(output_dir+"/train.txt", "w") as f:
    for label_file in file_list[:int(num*ratio)]:
        f.write(label_file.split('.')[0]+'\n')
f.close()

with open(output_dir+"/val.txt", "w") as f:
    for label_file in file_list[int(num*ratio):int(num*(ratio*0.5+0.5))]:
        f.write(label_file.split('.')[0]+'\n')
f.close()


with open(output_dir+"/test.txt", "w") as f:
    for label_file in file_list[int(num*(0.5+ratio*0.5)):]:
        f.write(label_file.split('.')[0]+'\n')
f.close()


with open(output_dir+"/sample_images_list.txt", "w") as f:
    for label_file in file_list[int(num*(0.5+ratio*0.5)):]:
        f.write(label_file.split('.')[0]+'.jpg\n')
f.close()

