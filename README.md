# template_tensorflow_train
a template for writing model using tensroflow

一个适用于普通监督学习的tensorflow_keras模型的代码模板（就是最常见的输入到输出的形式，包括图像或者其他的预测）,方便快速实现模型。 
## 
先运行preprocess_tfrecord.py生成tfrecords文件，然后运行fea_try_nn.py  开始训练。
两种控制流，基于图的和eager的  分别定义于nn_modules文件夹下的graph_act.py和eager_act.py中,通过修改fea_try_nn.py中代码切换。
