"""Pi风格确定性研究监督器。

本模块实现了基于策略驱动的确定性研究流程编排系统，
用于自动化医学研究实验的阶段调度与状态管理。

核心组件:
    - PolicyEngine: 基于依赖关系的阶段调度策略引擎
    - StateManager: 实验状态的持久化管理器
    - ToolRegistry: 工具注册与执行器
    - PiRunner: 实验流程编排主运行器
    - EvidenceReader: 实验证据读取器
"""
