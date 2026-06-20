# 可解释的谣言检测（Explainable Rumor Detection）

基于 DistilBERT 与多策略解释的谣言检测系统，对推文进行二分类并输出自然语言判断依据。

## 模型架构

本系统采用复合模型设计：

1. **分类模块**：微调 **DistilBERT**（`distilbert-base-uncased`）进行二分类
2. **解释生成模块**：结合三种互补策略：
   - **梯度归因**：通过梯度反向传播识别对预测贡献最大的关键词语
   - **语言学模式分析**：检测谣言特征（突发新闻用语、情绪化表达、未经证实说法、大写强调等）与可信度特征（信源引用、官方确认、数据支撑等）
   - **模板化自然语言生成**：将分析结果组合为通顺的判断依据文本

## 项目结构

```
├── train.csv                  # 训练数据集
├── val.csv                    # 验证数据集
├── src/
│   ├── train.py               # 模型训练脚本
│   ├── explainer.py           # 解释生成模块
│   ├── predict.py             # 统一预测接口
│   ├── evaluate.py            # 验证集评估脚本
│   └── generate_report.py     # 报告生成脚本
├── checkpoints/               # 训练后生成的模型与分词器
├── requirements.txt           # Python 依赖
├── readme.md                  # 本文件
└── report.pdf                 # 大作业报告
```

## 环境配置

### 依赖要求

- Python >= 3.8
- PyTorch >= 2.0.0
- transformers >= 4.40.0
- pandas, scikit-learn, numpy, tqdm

### 安装步骤

```bash
# 1.（可选）创建 conda 虚拟环境
conda create -n rumor-det python=3.11
conda activate rumor-det

# 2. 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
cd src
python train.py
```

脚本将自动完成以下操作：
- 加载 `train.csv` 作为训练集，`val.csv` 作为验证集
- 在谣言检测任务上微调 DistilBERT
- 将最佳模型保存至 `checkpoints/best_model.pt`
- 将分词器保存至 `checkpoints/` 目录

**预计训练时间**：CPU 约 60 分钟（10 轮），GPU 约 10 分钟。

### 2. 调用模型进行预测

```python
from src.predict import RumorDetector

# 初始化（自动加载已训练模型）
detector = RumorDetector()

# 单条预测
result = detector.predict("BREAKING: The government is hiding the truth!")
print(f"预测结果: {result['label']}")           # Rumor / Non-Rumor
print(f"置信度: {result['confidence']:.2%}")     # 模型置信度
print(f"判断依据: {result['explanation']}")      # 解释文本

# 批量预测
texts = ["第一条推文...", "第二条推文..."]
results = detector.batch_predict(texts)
```

### 3. 输出格式说明

每条预测返回一个字典，包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `prediction` | int | 0 = 非谣言，1 = 谣言 |
| `confidence` | float | 模型置信度（0~1） |
| `explanation` | str | 自然语言判断依据 |
| `label` | str | 可读标签（"Rumor" 或 "Non-Rumor"） |

## 解释示例

**谣言示例：**
> 输入文本："BREAKING: The police are covering up evidence in the shooting case! Wake up people!"
> - 预测结果：**Rumor**（置信度：94.2%）
> - 判断依据：模型将该推文分类为谣言（置信度：94.2%）。模型重点关注的关键信号词：breaking, police, covering, evidence, shooting。语言学分析发现以下谣言特征：使用了突发新闻用语（BREAKING）；指控隐瞒行为（"covering up"）；使用了号召性用语（"Wake up"）。这些语言模式是谣言传播的典型特征。

**非谣言示例：**
> 输入文本："According to the official statement released by the White House, the bill was signed today."
> - 预测结果：**Non-Rumor**（置信度：91.7%）
> - 判断依据：模型将该推文分类为非谣言（置信度：91.7%）。文本显示出可信度特征：包含明确信源引用；引用了官方确认信息（"official statement"）。信源引用和事实性报道语言表明这更可能是真实信息而非谣言。

## 数据集说明

数据集包含标注为谣言（1）或非谣言（0）的英文推文：
- **训练集**：2,840 条推文，涵盖 7 个不同事件类别
- **验证集**：401 条推文

## 模型性能

| 指标 | 数值 |
|------|------|
| 验证集准确率 | 85.79% |
| 加权 F1 值 | 0.86 |
| 谣言类 F1 值 | 0.83 |
| 非谣言类 F1 值 | 0.88 |
| 单条推理时间 | < 1 秒（CPU） |
