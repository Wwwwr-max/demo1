# demo1：中文新闻文本分类

基于 `bert-base-chinese` 的中文新闻文本分类示例。训练输入为新闻标题和关键词，模型判断文本所属的新闻分类。

## 项目结构

```text
demo1/
├── data/
│   ├── train_3k.txt   # 训练数据
│   ├── dev_1k.txt     # 验证数据
│   └── test_1k.txt    # 测试数据
├── train.py           # 数据处理、训练与验证
├── test.py            # 加载 best_model.pth 并测试
├── require.py         # 依赖与环境说明
└── README.md
```

## 数据格式

每行数据使用 `_!_` 分隔，共 5 个字段：

```text
id_!_code_!_name_!_title_!_key
```

示例：

```text
6552436691485852168_!_107_!_news_car_!_拉力风格的偏时点火系统调教之Fuel Cut篇_!_REV,Fuel,排气系统,ECU,Cut
```

字段含义：

- `id`：样本 ID
- `code`：分类编码，训练时统一减 100，使标签从 0 开始
- `name`：分类名称，用于人工阅读
- `title`：新闻标题
- `key`：逗号分隔的关键词，可能为空

文本拼接规则：

- 有关键词时使用 `关键词,标题`
- 没有关键词时只使用标题

## 环境依赖

```bash
pip install torch transformers tensorboard
```

首次运行需要联网下载 `bert-base-chinese` 预训练模型和分词器。

## 运行

训练：

```bash
python train.py
```

脚本会在每个 epoch 结束后运行一次验证，并在验证准确率提高时把模型权重保存到：

```text
best_model.pth
```

测试：

```bash
python test.py
```

测试脚本会加载 `best_model.pth`，在 `data/test_1k.txt` 上输出测试准确率。

查看训练曲线：

```bash
tensorboard --logdir log
```

## 模型说明

- 预训练模型：`bert-base-chinese`
- 任务模型：`AutoModelForSequenceClassification`
- 分类头输出数：`num_labels=17`
- 优化器：AdamW
- 学习率：`1e-5`
- 默认训练轮数：3
- 批大小：训练和验证均为 12

## 后续优化方向

- 按实际出现的 15 个类别重新映射标签，避免两个永远没有样本的输出类别
- 调整关键词拼接方式，让标题和关键词都有更明确的边界
- 增加训练轮数并配合 warmup、学习率衰减和早停
- 尝试 `hfl/chinese-roberta-wwm-ext` 等更强的中文预训练模型
- 在验证集上做错误分析，再针对容易混淆的类别补数据
