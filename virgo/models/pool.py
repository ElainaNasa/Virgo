"""用户提供的统一模型性能与价格目录；价格单位为人民币／百万 Token。"""
from dataclasses import dataclass, replace
import threading

PRICE_CURRENCY = "CNY"
# 任务预算上限与模型分级无关，保留现有任务接口。
TIER_MAX_COST = {"xlow": 1.0, "low": 3.0, "medium": 6.0, "high": 12.0, "xhigh": 25.0}

@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    provider: str
    input_price: float | None
    output_price: float | None
    capabilities: dict[str, float | None]
    cached_input_price: float | None = None
    evaluation_version: str = "用户提供的新模型表"
    pricing_note: str = "用户提供，人民币／百万 Token"
    alternate_prices: tuple[float, float, float] | None = None
    local: bool = False

MODEL_SPECS = (
    ModelSpec('minimax-M3.1-Flash', 'minimax-M3.1-Flash', 'minimax', 1, 4, {'AA_Composite': 29, 'FrontierCode_2_0': 21, 'Terminal_Bench_4_0': 26, 'AutomationBench_AA': 73, 'HLE_Diamond': 28, 'AA_Omniscience_v1_1': -6, 'DeepSWE_v1_1': 73.7, 'Toolathlon_Verified': None, 'HLE_No_Tools': None}, 0.1, evaluation_version='用户提供的新模型表', alternate_prices=None),
    ModelSpec('glm-5.3-flash', 'glm-5.3-flash', 'zhipu', 0.8, 2.8, {'AA_Composite': 28, 'FrontierCode_2_0': 18, 'Terminal_Bench_4_0': 33, 'AutomationBench_AA': 60, 'HLE_Diamond': 18, 'AA_Omniscience_v1_1': 1, 'DeepSWE_v1_1': 63.4, 'Toolathlon_Verified': 78.4, 'HLE_No_Tools': None}, 0.23, evaluation_version='用户提供的新模型表', alternate_prices=None),
    ModelSpec('mimo-v2.6-pro', 'mimo-v2.6-pro', 'mimo', 3, 6, {'AA_Composite': 30, 'FrontierCode_2_0': 24, 'Terminal_Bench_4_0': 35, 'AutomationBench_AA': 59, 'HLE_Diamond': 15, 'AA_Omniscience_v1_1': -4, 'DeepSWE_v1_1': 71.9, 'Toolathlon_Verified': 76.9, 'HLE_No_Tools': None}, 0.025, evaluation_version='用户提供的新模型表', alternate_prices=None),
    ModelSpec('mimo-v2.6-flash', 'mimo-v2.6-flash', 'mimo', 1, 2, {'AA_Composite': 27, 'FrontierCode_2_0': 22, 'Terminal_Bench_4_0': 23, 'AutomationBench_AA': 64, 'HLE_Diamond': 14, 'AA_Omniscience_v1_1': -11, 'DeepSWE_v1_1': 67.9, 'Toolathlon_Verified': 73.6, 'HLE_No_Tools': None}, 0.02, evaluation_version='用户提供的新模型表', alternate_prices=None),
    ModelSpec('kimi-k3.1-pro', 'kimi-k3.1-pro', 'moonshot', 13, 68, {'AA_Composite': 46, 'FrontierCode_2_0': 25, 'Terminal_Bench_4_0': 67, 'AutomationBench_AA': 73, 'HLE_Diamond': 37, 'AA_Omniscience_v1_1': 30, 'DeepSWE_v1_1': 75.5, 'Toolathlon_Verified': None, 'HLE_No_Tools': 51.1}, 1.3, evaluation_version='用户提供的新模型表', alternate_prices=None),
    ModelSpec('kimi-k2.7-code', 'kimi-k2.7-code', 'moonshot', 6.5, 27, {'AA_Composite': 26, 'FrontierCode_2_0': None, 'Terminal_Bench_4_0': 1, 'AutomationBench_AA': 24, 'HLE_Diamond': 15, 'AA_Omniscience_v1_1': None, 'DeepSWE_v1_1': None, 'Toolathlon_Verified': None, 'HLE_No_Tools': None}, 1.3, evaluation_version='用户提供的新模型表；AA 综合分为旧版', alternate_prices=None),
    ModelSpec('deepseek-v4.1-flash', 'deepseek-v4.1-flash', 'deepseek', 1, 4, {'AA_Composite': 33, 'FrontierCode_2_0': 28, 'Terminal_Bench_4_0': 34, 'AutomationBench_AA': 67, 'HLE_Diamond': 25, 'AA_Omniscience_v1_1': 1, 'DeepSWE_v1_1': 77, 'Toolathlon_Verified': None, 'HLE_No_Tools': 42.6}, 0.02, evaluation_version='用户提供的新模型表', alternate_prices=(2, 0.04, 8)),
    ModelSpec('deepseek-v4-pro', 'deepseek-v4-pro', 'deepseek', 4.5, 13.5, {'AA_Composite': 23, 'FrontierCode_2_0': 17, 'Terminal_Bench_4_0': 14, 'AutomationBench_AA': 57, 'HLE_Diamond': 20, 'AA_Omniscience_v1_1': -6, 'DeepSWE_v1_1': None, 'Toolathlon_Verified': 74.1, 'HLE_No_Tools': 42.7}, 0.15, evaluation_version='用户提供的新模型表', alternate_prices=(9, 0.3, 27)),
    ModelSpec('qwen3.8-flash', 'qwen3.8-flash', 'qwen', 0.8, 2.7, {'AA_Composite': 39.8, 'FrontierCode_2_0': None, 'Terminal_Bench_4_0': None, 'AutomationBench_AA': None, 'HLE_Diamond': None, 'AA_Omniscience_v1_1': None, 'DeepSWE_v1_1': 58.7, 'Toolathlon_Verified': 73.5, 'HLE_No_Tools': 35.9}, 0.1, evaluation_version='用户提供的新模型表；AA 综合分为旧版代理分数，不可直接跨版本比较', alternate_prices=None),
    ModelSpec('qwen3.7-plus', 'qwen3.7-plus', 'qwen', 2, 8, {'AA_Composite': 25, 'FrontierCode_2_0': None, 'Terminal_Bench_4_0': 1, 'AutomationBench_AA': 17, 'HLE_Diamond': 14, 'AA_Omniscience_v1_1': None, 'DeepSWE_v1_1': 16.5, 'Toolathlon_Verified': 50.6, 'HLE_No_Tools': 34.7}, 0.4, evaluation_version='用户提供的新模型表；AA 综合分为旧版', alternate_prices=(6, 1.2, 24)),
 )

CAPABILITY_DIMENSIONS = {key: {"meaning": key, "scale": "指数，可为负" if key in {"AA_Composite", "AA_Omniscience_v1_1"} else "百分比，存储为0至100", "source": "用户提供；旧版及代理分数见各模型 evaluation_version"} for key in MODEL_SPECS[0].capabilities}

class ModelPool:
    def __init__(self, specs=MODEL_SPECS, *, deepseek_v4_flash_pricing="nonpeak"):
        mode = {"off_peak": "nonpeak", "offpeak": "nonpeak"}.get(deepseek_v4_flash_pricing, deepseek_v4_flash_pricing)
        if mode not in {"nonpeak", "peak"}:
            raise ValueError(f"未知价格档：{mode}")
        self.deepseek_v4_flash_pricing = mode
        self._failed_channels = {}
        self._no_progress = {}
        self._excluded_models = {}
        self._failure_lock = threading.Lock()
        self._models = {}
        for model in specs:
            # DeepSeek 高峰档沿用现有配置；Qwen 第二档触发条件未提供，不擅自推断。
            if mode == "peak" and model.provider == "deepseek" and model.alternate_prices:
                i, c, o = model.alternate_prices
                model = replace(model, input_price=i, cached_input_price=c, output_price=o)
            self._models[model.model_id] = model

    def block_channel(self, task_id, model_id, reason):
        if model_id not in self._models:
            return
        with self._failure_lock:
            self._failed_channels.setdefault(task_id, {})[self.get(model_id).provider] = reason

    def channel_failures(self, task_id):
        with self._failure_lock:
            return dict(self._failed_channels.get(task_id, {}))

    def excluded_models(self, task_id):
        """返回任务级无进展排除列表，供所有模式路由解释选择。"""
        with self._failure_lock:
            return sorted(self._excluded_models.get(task_id, set()))

    def record_progress(self, task_id, model_id, no_progress):
        """连续两次无进展只排除该模型，不影响同渠道的其他模型。"""
        with self._failure_lock:
            key = (task_id, model_id)
            count = self._no_progress.get(key, 0) + 1 if no_progress else 0
            self._no_progress[key] = count
            if count >= 2:
                self._excluded_models.setdefault(task_id, set()).add(model_id)
                return True
            return False

    def channel_available(self, task_id, model_id):
        with self._failure_lock:
            excluded = model_id in self._excluded_models.get(task_id, set())
        return not excluded and (model_id not in self._models or self.get(model_id).provider not in self.channel_failures(task_id))

    def all(self):
        return list(self._models.values())

    def get(self, model_id):
        return self._models[model_id]

    def candidates(self, budget_tier=None, task_type="Other"):
        """兼容旧预算参数，但不按预算档过滤模型。"""
        return sorted(self.all(), key=lambda m: self._candidate_key(m, task_type))

    @staticmethod
    def _candidate_key(model, task_type):
        metric = task_type if task_type in model.capabilities else "AA_Composite"
        score = model.capabilities.get(metric)
        return (score is None, -score if score is not None else 0,
                model.input_price is None, model.input_price or 0)

    def escalation_candidates(self, previous_model_id, task_type="Other"):
        models = self.candidates(task_type=task_type)
        if previous_model_id not in self._models:
            return [m.model_id for m in models]
        previous = self.get(previous_model_id)
        metric = task_type if task_type in previous.capabilities else "AA_Composite"
        score = previous.capabilities.get(metric)
        better = [m for m in models if m.model_id != previous_model_id and score is not None and m.capabilities.get(metric) is not None and m.capabilities[metric] > score]
        return [m.model_id for m in (better or [m for m in models if m.model_id != previous_model_id])]

    def estimate_cost(self, model_id, input_tokens, output_tokens):
        model = self.get(model_id)
        if model.input_price is None or model.output_price is None:
            raise ValueError(f"模型 {model_id} 缺少计价数据")
        return (input_tokens * model.input_price + output_tokens * model.output_price) / 1_000_000

    def as_tables(self):
        return {"capability_dimensions": CAPABILITY_DIMENSIONS, "models": [
            {"model_id": m.model_id, "model": m.display_name, "provider": m.provider,
             **m.capabilities, "currency": PRICE_CURRENCY,
             "input_cache_miss_cny_per_m": m.input_price,
             "input_cache_hit_cny_per_m": m.cached_input_price,
             "output_cny_per_m": m.output_price,
             "alternate_prices": m.alternate_prices,
             "evaluation_version": m.evaluation_version, "pricing_note": m.pricing_note,
             "missing_metrics": [k for k,v in m.capabilities.items() if v is None]}
            for m in self.all()]}
