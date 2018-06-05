import os
import sys

label_dir = "./data_dataset_voc/SegmentationClass"
output_dir = "./dataset"

file_list = os.listdir(label_dir)
num = len(file_list)

with open(output_dir+"/train.txt", "w") as f:
    for label_file in file_list[:int(num*0.8)]:
        f.write(label_file.split('.')[0]+'\n')
f.close()

with open(output_dir+"/val.txt", "w") as f:
    for label_file in file_list[int(num*0.8):int(num*0.9)]:
        f.write(label_file.split('.')[0]+'\n')
f.close()


with open(output_dir+"/test.txt", "w") as f:
    for label_file in file_list[int(num*0.9):]:
        f.write(label_file.split('.')[0]+'\n')
f.close()


with open(output+"/sample_images_list.txt", "w") as f:
    for label_file in file_list[int(num*0.99):]:
        f.write(label_file.split('.')[0]+'.jpg\n')
f.close()

