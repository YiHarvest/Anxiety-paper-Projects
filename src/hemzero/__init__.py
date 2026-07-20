"""HemoZero研究流程库。

本库提供医学数据分析研究的完整流程支持，
包括数据处理、模型训练、结果验证和报告生成。

主要模块:
    - data: 数据加载、特征工程和数据验证
    - models: 基线模型、TabPFN视图和模型融合
    - validation: 交叉验证、Bootstrap评估和稳定性分析
    - distillation: 知识蒸馏（EBM、PySR）
    - fusion: 多模型融合与不确定性量化
    - reporting: 结果报告、图表和文稿生成
    - orchestration: 任务编排与TaskWeaver适配
"""

__version__ = "0.2.0"
