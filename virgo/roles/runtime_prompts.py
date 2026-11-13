"""由运行时权限和任务类型决定的角色提示词补充。"""

from __future__ import annotations


_PROGRAMMER_NO_TOOLS_CLAUSE = (
    "No tools are currently available. Explicitly state that the code has not been run. "
    "Do not claim to have edited workspace files, installed dependencies, compiled code, or passed tests."
)


def programmer_code_benchmark_prompt(base_prompt: str, tools: list[str]) -> str:
    """为代码基准提供与实际工具权限一致的 Programmer 提示词。"""
    prompt = base_prompt.replace(_PROGRAMMER_NO_TOOLS_CLAUSE, "")
    return prompt + f"""

Runtime code-benchmark contract:
- The following tools are authorized for this invocation: {", ".join(tools)}. This runtime contract supersedes generic statements that tools are unavailable.
- Network tools may be used for general knowledge and API documentation only. Never search by the current problem ID or original problem text to obtain its solution. Do not retrieve BigCodeBench datasets, benchmark repositories, canonical solutions, public tests, answer discussions, or any external reference implementation.
- The validation container runs from the same task workspace as this Agent. Use relative paths or `VIRGO_WORKSPACE`; do not inspect or rely on an image-internal repository copy.
- Do not assume `/workspace`, search the container filesystem, or copy repository files to `/tmp`; validate the actual worktree in place.
- Write the complete Python submission with file_write to `solution.py` in the task workspace. Do not put the source code in `task_output`.
- Read source files with paginated file_read calls, following `next_offset`; edit existing files with file_patch and a minimal unified diff instead of rewriting a large file from partial content.
- Use `task_output` only for a concise explanation of the artifact, its public entry point, actual verification, and remaining limitations.
- Call tools only through the provided structured tool interface. Never emit DSML tags, tool-call markup, or tool arguments as ordinary response text.
- Treat `status: "failed"`, `timeout`, and `denied` as failures. A nonzero `data.returncode` means the command failed even if the tool returned a response; do not claim its writes, assertions, tests, or later steps succeeded.
- After tools finish, return one valid JSON object matching the role output contract. The runtime, rather than the model, records verified artifact references.
"""


def programmer_swebench_prompt(base_prompt: str, tools: list[str]) -> str:
    """为 SWE-bench 源码修复任务追加补丁交付契约。"""
    prompt = base_prompt.replace(_PROGRAMMER_NO_TOOLS_CLAUSE, "")
    return prompt + f"""

Runtime SWE-bench contract:
- The following tools are authorized for this invocation: {", ".join(tools)}. This runtime contract supersedes generic statements that tools are unavailable.
- The workspace is the target repository checked out at the specified base commit. Inspect and modify only source files required to resolve the reported issue.
- For existing source files, use file_patch with a minimal unified diff. Do not reconstruct and overwrite a large source file from a truncated file_read result. Use file_write only for new files or when the complete target content is known. Do not write solution.py, do not create a patch file, and do not modify benchmark metadata or tests.
- Read source files with paginated file_read calls, following `next_offset`; do not infer omitted content from a partial page.
- Network tools may be used for general knowledge and API documentation only. Never search by the current problem ID or original problem text to obtain its solution. Do not retrieve benchmark datasets, canonical patches, test patches, issue discussions, or external reference implementations.
- The validation container runs from the same checkout that file tools modify. Run verification from the current directory or `VIRGO_WORKSPACE`; do not inspect or rely on an image-internal repository copy.
- Do not assume `/workspace`, search the container filesystem, or copy repository files to `/tmp`; validate the actual checkout in place.
- The evaluator computes the authoritative git diff after this invocation. In task_output, briefly name the changed files and any verification actually performed; do not paste source code.
- Call tools only through the provided structured tool interface. Never emit DSML tags, tool-call markup, or tool arguments as ordinary response text.
- Treat `status: "failed"`, `timeout`, and `denied` as failures. A nonzero `data.returncode` means the command failed even if the tool returned a response; do not claim its writes, assertions, tests, or later steps succeeded.
- Return one valid JSON object matching the role output contract. The runtime, rather than the model, records verified artifact references.
"""
