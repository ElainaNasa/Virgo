"""协作模式的静态定义。

内容来自项目需求中的“协作拓扑”章节。定义作为 Jev choice 的判据随请求发送，
不在运行时依赖 ``需求.md``。
"""

from __future__ import annotations


MODE_CHOICE_CRITERIA: dict[str, str] = {
    "Single": (
        "单个智能体接收任务并生成最终结果，不发生智能体间通信。"
        "选定后只选择 1 个角色。适用于一个角色能够独立交付、无需并行验证、"
        "无需多步角色接力的任务。完成后才可从 Parallel、FullMesh、ModelEscalation、Stop 中选择后续操作。"
    ),
    "Chain": (
        "多个智能体按顺序组成执行链；每个智能体的完整输出传给下一个，"
        "后者据此前序结果继续处理，最后一个智能体生成最终结果。"
        "每次只选择 1 个角色；至少必须完成 2 个工作节点，第二个节点完成后角色路由器才可选出 ChainStop。"
        "适用于必须按实现、审查、修复等步骤依赖前序结果的任务。"
        "整条链结束后才可从 Parallel、FullMesh、Stop 中选择后续操作。"
    ),
    "Parallel": (
        "先选择 1 个共同角色，再为该角色选择 2 至 4 个不同模型。"
        "每个模型都独立处理并解决同一个完整任务，彼此不通信，也不拆成不同职责；"
        "随后由 1 个 Synthesizer 比较候选、核查分歧与证据并形成最终结果。"
        "适用于需要多模型独立求解、交叉核验或比较候选答案的任务。"
        "汇总完成后才可从 Single、FullMesh、Stop 中选择后续操作。"
    ),
    "Tree": (
        "层级结构：根 Coordinator 将整体任务拆为最多 3 个子任务，"
        "SubtaskCoordinator 管理分支，并可委派最多 3 个第三层叶节点。"
        "分支间不直接通信，结果经父子连接逐层向上汇报；叶节点仅可嵌入 Single、Chain、FullMesh，"
        "不得继续嵌套。适用于存在清晰分解和分支管理需求的任务。"
        "根节点汇总所有分支后才可从 Single、Parallel、FullMesh、Stop 中选择后续操作。"
    ),
    "DAG": (
        "SpecialPlanner 将整体任务拆成带有向无环依赖边的子任务，并为每个节点给出初始上下文。"
        "无依赖的节点可并发，依赖节点必须等待全部前驱输出；节点内只能嵌入 Single、Chain、Parallel，"
        "不得继续调用宏模式。所有节点终止后由专项规划专家汇总终端节点。"
        "适用于存在明确依赖关系且可并行执行的多子任务工作。"
    ),
    "FullMesh": (
        "多个智能体可通过双向连接直接通信，在限定轮数内交换发现和反馈，"
        "随后由指定最终汇总环节生成结果。角色路由器最多选择 5 人，直到 FullMeshStop。"
        "适用于需要相互质询、讨论、澄清冲突或迭代共识的任务。"
        "讨论和最终汇总完成后才可从 Single、Parallel、Stop 中选择后续操作。"
    ),
    "ModelEscalation": (
        "不新建多智能体拓扑；以当前任务和已有结果改用更高能力的候选模型重新执行 Single。"
        "适用于角色与流程足够明确、主要瓶颈是当前模型能力不足的情况。"
    ),
    "Stop": "当前模式已经生成可接受的最终结果，或继续协作无法在剩余预算内带来合理收益；终止整个任务。",
}


def mode_choice_criteria(candidates: list[str]) -> dict[str, str]:
    """仅为本次允许的候选模式生成定义，保留候选顺序。"""
    return {
        candidate: MODE_CHOICE_CRITERIA.get(candidate, f"协作模式：{candidate}。仅在其满足当前任务结构时选择。")
        for candidate in candidates
    }
