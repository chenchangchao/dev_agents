# 【例2-8】
# agent_ui_api.py

from fastapi import FastAPI
from pydantic import BaseModel
from agent_runtime import  LocalAgent as Agent,  Tool, llm_config
from fastapi.middleware.cors import CORSMiddleware

## UI API 请求结构
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str

## Agent 定义（模拟记账机器人）
class AddRecordTool(Tool):
    def run(self, params: dict) -> str:
        category = params.get("category")
        amount = params.get("amount")
        return f"已记录消费：{category} - {amount}元"

    @property
    def description(self):
        return "记录一笔消费信息"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "消费类别"},
                "amount": {"type": "string", "description": "金额（单位：元）"}
            },
            "required": ["category", "amount"]
        }

def build_agent() -> Agent:
    return Agent(
        name="FinanceBot",
        llm=llm_config(temperature=0.3),
        tools=[AddRecordTool()],
        system_message="你是一个家庭记账助手，接收用户输入并记录消费类型与金额"
    )

## FastAPI 服务构建
app = FastAPI()
agent_instance = build_agent()

# CORS 允许跨域（用于网页调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 根接口
@app.get("/")
def index():
    return {"message": "Agent API 运行中"}

# 聊天API：接收用户输入→Agent处理→返回输出
@app.post("/chat", response_model=ChatResponse)
def chat_with_agent(req: ChatRequest):
    reply = agent_instance.chat(req.message)
    return ChatResponse(session_id=req.session_id, reply=reply)






# 使用如下命令启动脚本：
# uvicorn --version
# uvicorn agent_ui_api:app --reload --port 8080
# 使用如下方式请求API：
# curl -X POST http://localhost:8080/chat \
#   -H "Content-Type: application/json" \
#   -d '{"session_id": "user123", "message": "刚刚买菜花了20元"}'
