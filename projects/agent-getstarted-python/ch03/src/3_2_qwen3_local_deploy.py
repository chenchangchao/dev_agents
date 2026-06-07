# 【例3-2】
# qwen3_local_deploy.py

# 第一步：安装依赖
# pip install torch torchvision transformers accelerate
# pip install -U qwen==1.0.3  # 确保qwen库为官方发布版本

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 第二步：加载Qwen-235B模型
print(">>> 正在加载Tokenizer与模型...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-235B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-235B",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()
print(">>> 模型加载完毕")

# 第三步：构造输入
def build_prompt(user_input: str) -> str:
    return f"请解释以下中文成语的含义与出处：{user_input}"

# 第四步：执行推理
def run_inference(prompt: str, max_tokens: int = 128, temperature: float = 0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# 第五步：批量测试
idioms = ["画蛇添足", "纸上谈兵", "掩耳盗铃", "望梅止渴", "指鹿为马"]
print(">>> 开始批量推理任务")

for idiom in idioms:
    prompt = build_prompt(idiom)
    result = run_inference(prompt)
    print(f"\n【{idiom}】\n{result.strip()}\n")