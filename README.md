# VlogTool
## 软件功能

- [x] 使用MTCNN完成人脸检测，保存人脸大小为160x160到本地
- [x] 使用facenet提取人脸的特征，特征维度为512维，保存到本地
- [x] 使用贪心算法对人脸进行聚类
- [x] 完成基本的界面，支持显示文件中图像、显示一个id的相关图像
- [x] 代码支持Linux和Widows平台，支持cpu和gpu环境
- [ ] 对人脸聚类算法进行改进
- [ ] 支持自定义文件夹人脸检测，提取，聚类
- [ ] 支持数据库，实现更高效的增删查改
- [ ] 支持原图中截取人物脸部后进行相关人物的图像查询
- [ ] 支持图像风格转换的功能
- [ ] 对界面进行美化



## 软件相关算法

1. 人脸检测使用MTCNN[1]
2. 人脸识别使用Facenet[2]
3. 人脸特征使用InceptionResnetV1网络[3]



## 界面逻辑

界面分为左侧，中间和右侧三部分，左侧用于显示列表的图像（包括文件中图像列表和人物的图像列表），包括一个文件夹中所有图像和一个id相关的所有图像。中间部分用于显示大尺寸的图像（后续打算在此部分加入对图像的操作）。右侧用于显示人物id的视图。

界面初始化逻辑如下：

```flow
flow
st=>start: 界面初始化
op0=>operation: 左侧显示文件夹中图像
op1=>operation: 选中左侧列表中第一张图像，显示在中间
op2=>operation: 右侧显示人物id的脸部图像
e=>end: 初始化结束
st->op0->op1->op2->e
```

界面操作的逻辑如下：

```flow
flow
op0=>operation: 左侧可以滑动浏览图像的缩略图像，点击左侧图像
op1=>operation: 对左侧图像的选中状态重置，将点击图像变为选中状态
op2=>operation: 在中间显示左侧选中的图像
op0->op1->op2
```

```flow
flow
op0=>operation: 右侧可以滑动浏览id的缩略图像，点击id人脸
op1=>operation: 将左侧切换到人像，将人像中原有显示的图像清空，显示id相关的图像
op2=>operation: 选中人像列表中图像时，在中间显示选中的图像
op0->op1->op2
```



## 参考文献和相关资料

[1] Joint face detection and alignment using multitask cascaded convolutional networks

[2] FaceNet: A Unified Embedding for Face Recognition and Clustering

[3] Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning



