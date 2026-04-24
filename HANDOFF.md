# HANDOFF.md
> 本文档仅供本地开发助手参考，不上传 GitHub。
> 每位助手在完成任务后必须追加自己的批注。
---
## 项目背景与用户意图
- 这是一个 AstrBot 插件，基于 LLM 对聊天回复做后处理（清洗、分段、审查打回）。
- 用户核心诉求：高自由定制——所有提示词均可通过配置界面编辑，不硬编码行为。
- 运行环境：Linux Docker 容器，从 GitHub 拉取代码运行。本地 `C:\Users\ASUS\.astrbot\data\plugins\` 仅用于调试。
- 插件有两套分段机制：LLM 驱动分段（主路径）和本地 fallback 分段（备用路径）。
---
## 助手1 — 2026-04-23 18:50
### 本次修改
- 新增 `strip_segment_trailing_period` bool 配置项（默认 true），控制分段末尾句号去除行为
- 将"分段末尾去句号"从清洗阶段迁移到分段阶段：清洗阶段保留所有句号，分段阶段负责去除末尾单个句号
- 在 `_compose_segment_rule_block()` 和 `_compose_step_d_block()` 中动态注入约束（仅当开关开启时）
- 新增 `_strip_single_trailing_period()` 代码兜底方法，保护省略号变式（。。。/。。）不被误删
- 更新 `DEFAULT_JUDGE_PROMPT` 和 `judge_prompt_template` 配置默认值中的示例2/3
- 从 `clean_prompt_template` 默认值中移除"清洗掉分段末尾的句号"规则
### 重要上下文
- 用户明确要求：不能硬编码，必须通过配置项控制，保持高自由定制
- 省略号变式（。。。/。。）必须保留，只去除末尾单个句号
- 三管齐下保障：提示词动态注入 + 示例展示 + 代码兜底
- retry 路径复用 `_compose_segment_rule_block()`，自动获得约束，无需额外修改
### 待办/注意事项
- CHANGELOG.md 和版本号需要更新（1.4.0 → 1.4.1）
- 需要同步到本地插件目录并推送到 GitHub
---
## 助手1（续） — 2026-04-23 19:20
### 本次修改
- 新增 `process_all_replies` bool 配置项（默认 false），开启后会将插件工具调用等非 LLM 回复也纳入分段与清洗范围
- 修改 `on_decorating_result()` 中的判断逻辑：当 `process_all_replies` 开启时跳过 `_is_model_generated_reply` 检查
- 版本号 1.4.1 → 1.4.2
### 重要上下文
- 灵感来源于 spliter 插件的 `split_scope` 配置（llm_only / all），但用户偏好用 bool 值而非下拉选项
- 默认关闭，不影响现有用户行为
- 用户要求 hint 写得比 spliter 更好
---
## 助手1（续2） — 2026-04-23 19:50
### 本次修改
- 精简重复文本：Step C 和清洗规则框架中的空行压缩说明改为引用"严格禁止"区域
- 精简清洗规则框架文本，降低信息密度
- 新增3个 few-shot 示例（短文本、占位符、换行）
- 优化 reason 字段格式：给出具体格式示例
- 严格禁止区域新增 segments 拼接一致性约束
- 同步更新 _conf_schema.json 中的 judge_prompt_template 默认值
- 版本号 1.4.2 → 1.4.3
### 重要上下文
- 步骤顺序不能硬性要求串行（A→B→C→D），因为用户可能只开部分功能
- 用户放弃"内容清洗"功能（如清洗括号及其中内容），因为风险太大——LLM 分不清用户追加的规则和原有硬约束
- 插件定位是轻量可定制化的分段器，使用中小参量模型，提示词质量至关重要
---
## 助手2 — 2026-04-24 16:30
### 本次修改
- Step A 提示词从模糊的"不适合主题"改为明确列举四类拒绝范围：露骨色情、政治敏感、涉及儿童、自残/暴力/违禁品
- 新增"轻微擦边、暧昧暗示、日常调侃必须放行"的明确约束
- 拒绝时 reason 固定为"内容涉及不适合主题，拒绝处理"，与 `_should_force_local_fallback` 逻辑对齐
- 修复单段输出末尾句号未清理：在 `on_decorating_result()` 中新增最终兜底，对 segments_text 执行 `_strip_single_trailing_period()`
- 新增字数限制功能：`enable_max_process_length`（默认开启）+ `max_process_length`（默认 500），超长文本跳过后处理原样发送
- 版本号 1.4.3 → 1.4.4
### 重要上下文
- 用户反馈白丝相关内容被 Step A 过度拒绝——模型把轻微擦边也当成了"不适合主题"
- Step A 是永远开启的，与 Step B（判别与打回）独立，用户纠正了助手对此的误解
- 单段句号 bug 的根因未完全定位（从代码静态分析看 `_apply_segment_limits` 应该已经处理），但在 `on_decorating_result` 中加了最终兜底确保万无一失
- 字数限制统计的是纯文本字数（排除 `[[RP_COMP_x]]` 占位符），使用 `_strip_placeholders()` 计算
- 超长文本跳过时直接 return，不做任何处理，原样发送
### 待办/注意事项
- 需要观察 Step A 新提示词在实际使用中的效果，可能需要进一步微调拒绝范围
---
## 助手2（续） — 2026-04-24 21:20
### 本次修改
- 修复 `_strip_single_trailing_period()` 在分段末尾含占位符时句号清理失效的问题
- 根因：占位符被补回到文末后，句号在占位符前面，方法只检查字符串末尾，无法匹配
- 修复：先检测末尾是否为占位符，如果是则检查占位符前的文本是否以单个句号结尾
- 同时修复了之前 v1.4.4 中"单段句号兜底"的根因——`_apply_segment_limits` 调用 `_strip_single_trailing_period` 时同样存在此问题
- 版本号 1.4.4 → 1.4.5
### 重要上下文
- 用户提供的日志样本：`那小奈多吃点甜食好了。[[RP_COMP_1]]`，句号在占位符前未被清理
- `_strip_single_trailing_period` 从 `@staticmethod` 改为实例方法，因为需要访问 `self.TRAILING_PLACEHOLDERS_PATTERN`
- 之前的"兜底句号清理"（v1.4.4 在 on_decorating_result 中加的）也调用了此方法，所以两处路径都受益于本次修复
