# demo1：基于 bert-base-chinese 的中文新闻文本分类

本项目使用本地 `bert-base-chinese` 预训练模型，对中文新闻标题和关键词进行分类。训练输入由新闻标题和关键词拼接而成，模型判断文本所属的新闻分类。

## 框架组成

```text
demo1
├── demo_config.json        # 参数配置
├── config.py               # 读取 demo_config.json
├── dataset.py              # 数据读取与 Dataset,tokenizer
├── model_m.py              # 模型
├── train.py                # 训练入口
├── test.py                 # 测试入口
├── requirements.txt        # 第三方依赖
├── data/                   # 训练/验证/测试数据
├── pretrain/bert-base-chinese/  # 本地预训练模型
├── log/                    # TensorBoard 日志
├── picture/                # 结果截图
└── best_model.pth          # 验证集最优权重
```

## 各模块说明

- `demo_config.json`
  保存 `max_length`、数据路径、`batch_size`、学习率、epoch 数和本地预训练模型路径。

- `config.py`
  提供 `load_config()`，统一读取 `demo_config.json`。

- `dataset.py`
  `pro_data()` 解析原始数据，把分类码 `code - 100` 转成从 0 开始的标签。
  `Mydata` 继承 `torch.utils.data.Dataset`，保存文本和标签。
  `make_collate_fn(tokenizer, max_length)`：对一个 batch 的文本进行截断、填充和标签整理。
- `model_m.py`。
  - `Mymodel`：使用 `BertModel + Linear`，取 BERT 输出的 CLS 向量，再经过全连接层输出 logits。

- `train.py`
  加载配置和数据，完成训练和验证。`train_T()` 执行一轮训练，`verify()` 在验证集上计算准确率,精确率，召回率，F1，并在每个 epoch 使用 TensorBoard 记录指标；验证集准确率更高时保存 `best_model.pth`。

- `test.py`
  加载测试集和 `best_model.pth`，计算测试集准确率。

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
- `code`：分类编码
- `name`：分类名称
- `title`：新闻标题
- `key`：关键词

文本拼接规则：

- 有关键词时，使用 `key,title`
- 没有关键词时，只使用 `title`

## 模型与训练配置

- 预训练模型：本地 `./pretrain/bert-base-chinese`
- 任务模型：`BertModel` 后接 `Linear`
- 最大长度：`max_length=512`
- 优化器：AdamW
- 学习率：`1e-5`
- 批大小：`12`
- 训练轮数：`12`
- 损失函数：CrossEntropyLoss

## 运行

```bash
pip install -r requirements.txt
python train.py
python test.py
```

查看 TensorBoard 曲线：

```bash
tensorboard --logdir=./log
```

## 运行结果

在训练集 3000 条、验证集 1000 条、batch_size=12、学习率 1e-5 的条件下训练 12 轮：

| Epoch | train_loss | dev_acc | p_macro | r_macro | f1_macro |
| --- | ---: | ---: |------:|--------:|---------:|
| 1 | 1.0899 | 0.8190 | 0.8590|  0.8214 |0.8315|
| 2 | 0.4081 | 0.8610 |0.8664|  0.8270 |0.8392|
| 3 | 0.2263 | 0.8650 |0.8646|  0.8619 |0.8618|
| 4 | 0.1359 | 0.8490 |0.8345|  0.8204 |0.8196|
| 5 | 0.0681 | 0.8600 |0.8532|  0.8400 |0.8421|
| 6 | 0.0437 | 0.8540 |0.8532|  0.8400 |0.8421|
| 7 | 0.0298 | 0.8440 |0.8279|  0.8376 |  0.8232 |

- 最优`dev_acc`: **0.8670**，最优`macro‑F1`: **0.8618**，出现在 Epoch 3
- `best_model.pth` 保存第 3 轮模型权重
- 早停 patience=4，第 7 轮触发早停终止训练
