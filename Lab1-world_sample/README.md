# World 地理坐标采样数据集

本目录包含自主构造的地理问题、来自 GeoNames 的坐标标签，以及 Qwen3-4B-Base 的隐藏层特征。用于训练线性模型，从问题表征中预测地点经纬度。本数据集不包含句子真假标签。

## 数据规模与划分

共 **12,000 个独立地点实体**，覆盖 235 个 GeoNames 国家/地区代码。

| 划分 | 地点数 | 国家/地区代码数 | 用途 |
|---|---:|---:|---|
| `train` | 8,382 | 211 | 训练模型；拟合标准化等预处理参数 |
| `valid` | 1,050 | 163 | 新数据分布下的超参数选择与验证 |
| `test` | 1,050 | 163 | 评估训练见过国家内的新地点 |
| `test_unseen` | 1,518 | 24 | 评估训练未见国家/地区中的地点 |
| 合计 | 12,000 | 235（去重） | 各划分不共享地点实体 |

随机种子为 `20261004`。先按大洲分层留出约10%的国家/地区代码作为 `test_unseen`，再将其余国家中的地点按约80/10/10划分为 train/valid/test。少于3个地点的国家全部进入train，以保证普通valid/test中的国家在train中出现过。

`test_unseen` 的国家/地区不出现在新增train中。“未见”仅针对线性探测模型，不代表 Qwen 预训练时没有见过这些国家。它不是从全部地点中随机抽出的普通测试集。

这里的代码数量按 GeoNames 字段统计，不是对政治实体数量的认定。原课程的500条训练和130条验证数据不在本目录中，也未并入本目录的train。

## 数据产生方式

### 1. 选取地点并构造问题

地点来自 GeoNames `cities15000.zip` 快照，以城市、居民点及各级行政中心为主。名称虽为cities15000，该来源也包含部分人口较少的首都，并非每个地点人口都超过15,000。

只接受 `PPL/PPLA/PPLA2/PPLA3/PPLA4/PPLC` 等居民点类型，排除历史、废弃居民点和城市内部子区等其它类型。采样按国家/地区轮流选择，每个代码最多600条；国家内部兼顾10°经纬度网格和人口档位（<10万、10万至100万、≥100万），桶内优先较高人口地点。

该策略增加全球地点覆盖，但不等于按地表面积均匀采样。新增地点中9,954条位于北半球、2,046条位于南半球。与原课程包含湖泊、山峰、建筑等的world数据相比，本数据更偏向城市，存在分布差异。

使用地点名、行政区和国家构造英文问题：

```text
What's the longitude and latitude of {place}, {admin1}, {country}?
```

行政区缺失或与地点/国家同名时省略。行政区用于消歧，国家保留在问号前最后位置。每个问题对应一个地点实体；剔除重复或含糊提示词，并按原课程地点名及GeoNames别名作保守排除，但不能保证识别所有未知拼写差异。

### 2. 提取模型表征

使用 Hugging Face 的 **`Qwen/Qwen3-4B-Base` 非量化模型**，固定revision：

```text
906bfd4b4dc7f14ee4320094d8b41684abff8539
```

将问题句子输入模型，仅进行前向传播，不生成答案、不使用聊天模板，标签坐标不进入输入。

- 提取 **0-based decoder block 11、23、35** 的输出，每个输出为2560维。
- 对每句话取国家名最后一个重叠token的隐藏状态。
- 使用真实block输出；第35层也取最终RMSNorm之前的输出。
- 模型权重及计算使用BF16；保存的特征为float16。特征存为float16不等于模型权重量化。
- 云端采集使用RTX5090；模型处于eval模式，关闭梯度和KV缓存。

## 标签意义与来源

**`latitude`、`longitude` 原样取自GeoNames的WGS84地名库坐标，不是Qwen回答的坐标。** 因此小模型是否能够正确回答地理问题，不影响本数据集标签的构造。

| 标签 | 含义 | 单位与范围 |
|---|---|---|
| `latitude` | 纬度；北纬为正，南纬为负 | 十进制度，[-90, 90] |
| `longitude` | 经度；东经为正，西经为负 | 十进制度，[-180, 180] |

虽然问题先问longitude，**CSV及模型目标列顺序始终为 `latitude, longitude`（纬度、经度）**，与课程world训练格式保持一致。

城市坐标是地名库的代表点，不是城市边界、所有建筑的位置，也不一定是严格的行政或人口中心。GeoNames汇集多种来源并允许社区编辑，仍可能有误；这里的ground truth指可追溯的知识库标签，不代表逐条人工测量或精度保证。小数位数也不能直接解释为定位精度。

原课程标签没有提供足够的来源信息，不能据此断言它们来自模型生成；本目录明确采用独立地名库标签。

## CSV 字段

| 字段 | 含义 |
|---|---|
| `sentence` | 实际输入模型的英文问题 |
| `country` | 国家/地区名称；也用于定位输入中的国家token |
| `latitude`、`longitude` | 地点的监督坐标标签 |
| `entity_id` | 稳定地点标识，如 `geonames:3862583` |
| `place` | 地点名称 |
| `admin1` | 一级行政区名称，用于消歧 |
| `country_code` | GeoNames国家/地区代码 |
| `continent` | 大洲代码 |
| `population` | GeoNames记录的人口数，不保证为当前统计值 |
| `feature_code` | GeoNames地物类型代码 |
| `source_url` | 对应GeoNames实体页面 |
| `source_modified` | GeoNames记录修改日期，不是采样或测量日期 |

## 文件与读取方式

每个划分包含：

```text
train.csv
train.layer11.npy
train.layer23.npy
train.layer35.npy
train.extraction.json
```

valid/test/test_unseen文件结构相同。每个`.npy`的形状为`(对应CSV行数, 2560)`，第i行特征与对应CSV的第i条数据严格对齐；读取CSV时表头不算数据行。`*.extraction.json`记录模型、采集配置、完成状态和CSV/特征文件SHA256。

```python
from pathlib import Path
import numpy as np
import pandas as pd

folder = Path('LAB1/sampled_data/world')  # 从项目根目录运行
x = np.load(folder / 'train.layer11.npy', allow_pickle=False).astype(np.float64)
df = pd.read_csv(folder / 'train.csv')
y = df[['latitude', 'longitude']].to_numpy(dtype=np.float64)
assert x.shape == (8382, 2560)
assert y.shape == (8382, 2)
```

使用单隐藏层训练时只加载所选层；标准化统计量只能从训练数据拟合。测试集不应参与超参数选择。本项目后续有按原课程valid选型的实验，在那些实验中本目录valid仅作为额外评估集；具体协议应以相应实验报告为准。

## 来源、许可与复现

- [GeoNames数据下载及字段说明](https://download.geonames.org/export/dump/)，标签数据许可为 **CC BY 4.0**，再分发时保留GeoNames署名和来源。
- [Qwen模型固定版本](https://huggingface.co/Qwen/Qwen3-4B-Base/tree/906bfd4b4dc7f14ee4320094d8b41684abff8539)，模型采用Apache 2.0许可。
- 来源快照与生成脚本在 `LAB1/world_sample/`；采样种子、来源哈希、划分统计见其中 `datasets/world/dataset_manifest.json`。
- 本目录是云端采集结果的本地副本；采集完成后通过了数据、模型权重及输出特征完整性验证。
