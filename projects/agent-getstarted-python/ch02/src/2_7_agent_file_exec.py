# 【例2-7】
# agent_file_exec.py

import traceback
from agent_runtime import BaseTool, LocalAgent as Agent, llm_config, safe_workspace_path

# 工具：读取文件内容
class ReadFileTool(BaseTool):
    def run(self, params: dict) -> str:
        path = safe_workspace_path(params.get("filepath", ""))
        if not path.exists():
            return f"文件不存在：{path}"
        try:
            with path.open("r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"读取失败：{str(e)}"

    @property
    def description(self):
        return "读取指定文件内容"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "文件路径"}
            },
            "required": ["filepath"]
        }

# 工具：写入文件内容
class WriteFileTool(BaseTool):
    def run(self, params: dict) -> str:
        path = safe_workspace_path(params.get("filepath", ""))
        content = params.get("content", "")
        try:
            with path.open("w", encoding="utf-8") as f:
                f.write(content)
            return f"内容已成功写入文件：{path}"
        except Exception as e:
            return f"写入失败：{str(e)}"

    @property
    def description(self):
        return "将文本写入指定文件"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "写入内容"}
            },
            "required": ["filepath", "content"]
        }

# 工具：执行Python代码
class ExecCodeTool(BaseTool):
    def run(self, params: dict) -> str:
        code = params.get("code", "")
        try:
            local_vars = {}
            exec(code, {}, local_vars)
            return "执行成功，输出变量：" + str(local_vars)
        except Exception:
            return "执行出错：" + traceback.format_exc(limit=2)

    @property
    def description(self):
        return "执行用户提供的Python代码（仅限受控环境）"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "待执行Python代码"}
            },
            "required": ["code"]
        }

# 构建Agent
def build_agent():
    agent = Agent(
        name="FileExecAgent",
        llm=llm_config(temperature=0.2),
        tools=[ReadFileTool(), WriteFileTool(), ExecCodeTool()],
        system_message="你是一位具备文件系统与代码执行能力的助手，负责文本文件处理与Python代码运行"
    )
    return agent

# 测试流程
def run_demo():
    agent = build_agent()

    print(agent.chat("请将以下内容写入文件test.txt：你好，这是智能体写入的内容"))
    print(agent.chat("读取文件test.txt中的内容"))
    print(agent.chat("请运行如下代码：a = 5\nb = 3\nc = a * b"))

if __name__ == "__main__":
    run_demo()
