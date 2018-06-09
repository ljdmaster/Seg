1.准备数据
  a. 进入data文件在labels.txt内保存要分割的物体类别
  b. 用labelme软件打标签, 原始图片和标注数据保存在./path/data_annotated文件
  c. 运行 python3 labelme2voc.py labels.txt data_annotated data_dataset_voc
     得到voc数据集格式数据
  d. 运行 data_set.py 得到训练，验证，测试数据信息txt文件 
  e. 运行 create_pascal_tf_record.sh 生成训练集和验证集的tfrecord文件 
  Note：以上操作需确认路径正确

2.训练
  a. 修改train.py 内超参数设置
  b. 修改train.sh 内参数 
  c. 运行train.py 开始训练
  d. tensorboard  --logdir  model_path/ 检测训练结果

3.测试
  a. 修改inference.py内参数 
  b. 运行inference.py调用训练好的网络 
