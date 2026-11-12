# 共享洪水情景下的交通网络连通风险

这是独立的气候资料Python项目。`data/`保存合成资料，`climate_data.io`负责严格读取、公历日期和来源字节摘要，`climate_data.domain`提供现有描述统计，`climate_data.cli`将摘要输出为JSON。所有观测值、站点、人群、情景与金额都是虚构。字段和研究约定见[资料约定](docs/data-contract.md)。

当前Python 3.11及以上可直接使用，没有第三方Python依赖。项目根目录运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q climate_data tests
python -m climate_data.cli inspect data/normal.json
```

`inspect`显示当前原始摘要，不覆盖资料。`invalid-`前缀文件为不合法资料，其他变体使用相同字段保留业务边界。来源字节SHA-256随摘要输出，后续结果应同时标明输入身份、量纲和适用范围。

## 相关灾害连通风险分析

`climate_data.risk`针对同一流域道路在同一场洪水中联动失效的问题，显式建模共享因子：因子相互独立、状态概率为精确分数，同一条道路的可通性是其允许状态约束的合取，同一因子可同时约束多条平行道路。`risk`子命令对基准与每个加固方案输出：

- 精确有理数连通/失败概率、状态合并下的访问状态数、记忆化状态数、已证明概率质量与终止状态；小空间附全枚举核对，预算耗尽时不给精确结论。
- 网络已失败条件下各因子状态的精确后验；零失败概率时标记为不可定义，不用先验替代。
- 固定种子、全部方案共享同一抽样矩阵的蒙特卡洛估计、配对改善量与配对样本标准误（不把两个边际标准误相加），以及仅供解释的失败代表情景与终端分区。

```sh
python -m climate_data.cli risk data/normal.json
python -m climate_data.cli risk data/large-factor-space.json -o report.json
```

所有情景与路网在一次分析内冻结，工具只做只读风险计算，不导航也不操作交通设施。语义细节见[资料约定](docs/data-contract.md)。

容器使用固定基础镜像，可运行`docker build -t climate-material .`，随后`docker run --rm climate-material`检查资料。`docker run --rm climate-material python -m unittest discover -s tests -v`执行已有检查。运行不依赖远端账号或实时服务。
