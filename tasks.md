# Laplace项目复现任务分配

## 环境配置与基础设置
- [ ] 创建conda环境
- [ ] 安装所有依赖包
- [ ] 配置CUDA环境
- [ ] 准备数据集下载脚本

## 表1：标准图像任务上的OOD检测

### 成员A：MNIST相关实验
- [ ] 数据准备
  - [ ] 下载MNIST主数据集
  - [ ] 下载EMNIST、Fashion-MNIST、KMNIST作为OOD数据集
- [ ] 模型训练
  - [ ] 实现LeNet-5模型
  - [ ] 训练MAP模型（50轮）
  - [ ] 保存检查点
- [ ] Laplace实现
  - [ ] 实现最后一层Laplace近似
  - [ ] 优化先验精度
- [ ] 评估
  - [ ] 计算AUROC
  - [ ] 计算平均置信度
  - [ ] 运行3次实验取平均

### 成员B：CIFAR-10相关实验
- [ ] 数据准备
  - [ ] 下载CIFAR-10主数据集
  - [ ] 下载SVHN、LSUN、CIFAR-100作为OOD数据集
- [ ] 模型训练
  - [ ] 实现WideResNet-16-4模型
  - [ ] 训练MAP模型（200轮）
  - [ ] 实现数据增强（crop + flip）
  - [ ] 保存检查点
- [ ] Laplace实现
  - [ ] 实现最后一层Laplace近似
  - [ ] 优化先验精度
- [ ] 评估
  - [ ] 计算AUROC
  - [ ] 计算平均置信度
  - [ ] 运行3次实验取平均

### 成员C：基线方法实现
- [ ] Deep Ensemble
  - [ ] 实现5个独立网络训练
  - [ ] 集成预测
- [ ] VB (Flipout)
  - [ ] 实现变分贝叶斯方法
  - [ ] 训练和评估
- [ ] CSGHMC
  - [ ] 实现12条链采样
  - [ ] 训练和评估
- [ ] SWAG
  - [ ] 实现40个快照
  - [ ] 训练和评估

## 图6：WILDS基准实验

### 成员A：图像数据集
- [ ] Camelyon17
  - [ ] 数据准备
  - [ ] DenseNet-121模型实现
  - [ ] Laplace近似
  - [ ] 评估
- [ ] FMoW
  - [ ] 数据准备
  - [ ] DenseNet-121模型实现
  - [ ] Laplace近似
  - [ ] 评估

### 成员B：文本数据集
- [ ] CivilComments
  - [ ] 数据准备
  - [ ] DistilBERT-base实现
  - [ ] Laplace近似
  - [ ] 评估
- [ ] Amazon
  - [ ] 数据准备
  - [ ] DistilBERT-base实现
  - [ ] Laplace近似
  - [ ] 评估

### 成员C：回归任务
- [ ] PovertyMap
  - [ ] 数据准备
  - [ ] ResNet-18实现
  - [ ] Laplace近似
  - [ ] 评估
- [ ] 温度缩放实现
  - [ ] 在所有数据集上实现
  - [ ] 超参数调优
  - [ ] 评估

## 结果整合与验证
- [ ] 汇总所有实验结果
- [ ] 验证数值对齐
- [ ] 生成对比图表
- [ ] 编写实验报告

## 时间节点
- 第一周：环境配置和基础实验
- 第二周：完成表1的实验
- 第三周：完成图6的实验
- 第四周：结果整合和报告撰写

## 注意事项
1. 所有实验都需要使用相同的随机种子(0,1,2)
2. 及时保存实验检查点和结果
3. 定期同步代码和实验结果
4. 记录实验过程中的问题和解决方案 