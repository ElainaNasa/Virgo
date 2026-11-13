from virgo.types import ModeResult
from .langgraph_runtime import END, START, LangGraphTopologyExecutor, new_graph


class SingleExecutor(LangGraphTopologyExecutor):
    mode = "Single"

    def run(self, task_id: str, mode_instance_id: str, roles: list[str]) -> ModeResult:
        role = roles[0]
        graph = new_graph()

        def execute(state):
            updates, _, _, status = self.agent_state(state, role, "独立完成原始任务并生成最终结果。")
            updates["final_output"] = updates["outputs"][-1]
            updates["status"] = status
            return updates

        graph.add_node("execute", execute)
        graph.add_edge(START, "execute")
        graph.add_edge("execute", END)
        return self.invoke(self.compile(graph), task_id, mode_instance_id, roles)
