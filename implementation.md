# 实现细节

## 环境配置

### 基础环境要求
- Python ≥ 3.9
- CUDA 11.x

### 安装步骤
```bash
# 创建conda环境
conda create -n laplace-redux python=3.9 pytorch torchvision cudatoolkit=11.8 -c pytorch

# 安装核心依赖
pip install laplace-torch backpack-for-pytorch wilds tqdm pandas seaborn

# 可选依赖（用于Figure 6）
pip install scikit-learn einops timm transformers
```

## 表1：标准图像任务上的OOD检测

### 任务描述
评估模型在分布外数据上给出低置信度的能力：
- CIFAR-10的OOD数据集：SVHN、LSUN、CIFAR-100
- MNIST的OOD数据集：EMNIST、Fashion-MNIST、KMNIST
- 评估指标：AUROC↑（越高越好）和平均置信度↓（越低越好）

### 模型与推断方法
1. MAP（标准训练）
2. Deep Ensemble（5个独立网络）
3. VB（Flipout变分贝叶斯）
4. CSGHMC（12条链采样）
5. SWAG（40个快照）
6. LA / LA*（最后一层+KFAC；LA*使用完整经验Fisher替代Hessian）

### 基础网络
- MNIST：LeNet-5
- CIFAR-10：WideResNet-16-4
  - 训练细节见附录C.2.1

### 训练方式
- 所有方法首先完整训练MAP权重（70-120轮，遵循标准文献）
- 然后对最后一层应用post-hoc Laplace
- VB/CSGHMC/SWAG/Ensemble要么复用基线权重，要么按照各自论文推荐重新训练

### 指标计算
- 置信度 = max softmax logit（越低越好）
- AUROC使用置信度来区分ID和OOD图像
- 结果在3个随机种子上取平均，并给出±标准误

### 主要发现
LA和LA*显著降低置信度并提高AUROC，同时几乎不增加训练/推理开销。

### 复现步骤

#### 1. 数据准备
```python
# 使用torchvision下载数据集
from torchvision import datasets
# MNIST/CIFAR10主数据集
# 同时下载OOD数据集：SVHN、LSUN、CIFAR100、EMNIST、Fashion-MNIST、KMNIST
```

#### 2. 训练MAP模型
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
num_epochs = 100

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
num_epochs = 200     

```

复现 “+DA” 列（Figure 8(a)(c))，启用 RandomCrop(32,4) + RandomHorizontalFlip()。

复现 “−DA” 列（Figure 8(b)(d))，需禁用所有数据增强

#### 3. 构造Laplace近似
```python
from laplace import Laplace

la = Laplace(model, 'classification',
             subset_of_weights='last_layer',
             hessian_structure='kron')

# 估计Hessian
la.fit(train_loader)

# 通过EB优化先验精度
la.optimize_prior_precision()
```

#### 4. 预测
使用`probit=True`获取封闭式预测，收集softmax和置信度。

#### 5. 计算指标
```python
from sklearn.metrics import roc_auc_score
# 计算每个OOD集合的AUROC并取平均
# 计算所有OOD样本的平均置信度
```

#### 6. 其他基线方法
- Deep Ensemble：5个独立训练
- VB：使用`torch.nn.functional.flipout`或原论文脚本
- CSGHMC & SWAG：按照原仓库参数设置

### 复现提示
- 使用随机种子0,1,2，对三次结果取均值±标准误
- 如果显存不足，可以将CIFAR-10的KFAC改为对角Hessian（AUROC差异约0.01）

## 图6：WILDS基准上的真实分布漂移

### 数据集
来自WILDS基准，涵盖不同领域和模态：
- Camelyon17
- FMoW
- CivilComments
- Amazon
- PovertyMap
每个数据集都包含ID和OOD划分（医院、区域、人口统计、国家等）

### 预训练模型
- Camelyon17 / FMoW：DenseNet-121
- CivilComments / Amazon：DistilBERT-base
- PovertyMap：ResNet-18

### 方法对比
- MAP
- Deep Ensemble（使用官方仓库提供的5个权重）
- Temperature Scaling（温度缩放）
- Last-layer Laplace（根据数据集大小选择KFAC或完整版本）

### 评估指标
对每个数据集，两列（ID和OOD）：
- 上排：NLL（回归任务用MSE）
- 下排：ECE（回归任务用RCE），带1σ误差条

### 超参数调优
在各自的ID验证集上：
- 温度缩放：温度T
- Laplace：先验精度γ，PovertyMap的观测噪声σ

### 主要观察
Laplace的标定误差（NLL/ECE）显著低于MAP，在OOD条件下与温度缩放或Deep Ensemble相当，同时计算和内存开销更小。

### 复现步骤

#### 1. 安装WILDS
```bash
pip install wilds
wilds get_dataset  # 下载五个数据集
```

#### 2. 下载预训练检查点
从官方codalab worksheet下载预训练模型（约3-6GB）

#### 3. 最后一层Laplace
- DenseNet/ResNet：根据最后一层维度选择KFAC或full
- DistilBERT：最后的线性分类器，使用full Hessian（d≈768×K）

#### 4. 调优超参数
在ID验证集上使用`optimize_prior_precision(method='marglik')`优化γ和σ

#### 5. 预测和评估
```python
# NLL计算
torch.nn.NLLLoss(reduction='mean')

# ECE计算（15-bin）
# 使用eval_utils.py中的实现
```

#### 6. 其他方法实现
- Deep Ensemble：同时forward 5个checkpoint并平均logits
- 温度缩放：在验证集上网格搜索T并重新计算softmax

### 复现注意事项
- CPU内存可能成为瓶颈，建议一次只加载一个数据集
- DistilBERT + full Hessian约需600MB内存
- 如果GPU内存不足，可以使用对角Hessian

## 结果验证清单

### 数值对齐标准
- AUROC误差在±0.3范围内
- ECE误差在±0.01范围内
- 使用相同的随机种子(0-4)确保置信区间一致
- 图表使用相同的指标名称和方向（↑/↓）

## 常见问题与解决方案

| 问题 | 解决方案 |
|------|----------|
| Hessian奇异或计算慢 | - 减小batch size（≤128）<br>- 使用`ggn`替代`ef`<br>- 切换到对角结构 |
| γ无法收敛 | 1. 先用固定γ=1计算Hessian<br>2. 再使用`optimize_prior_precision(method='marglik')` |
| WILDS数据下载慢 | 使用代理或预先在服务器下载（总数据约55GB） |
