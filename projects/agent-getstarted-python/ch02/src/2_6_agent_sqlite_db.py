# 【例2-6】
# agent_sqlite_db.py

import sqlite3
from agent_runtime import BaseTool, LocalAgent as Agent, data_path, llm_config

DB_PATH = data_path("employee.db")

# 初始化数据库（首次创建）
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT,
            title TEXT
        )
    ''')
    cursor.execute("UPDATE employees SET name = TRIM(name)")
    cursor.execute("""
        DELETE FROM employees
        WHERE id NOT IN (
            SELECT MAX(id) FROM employees GROUP BY name
        )
    """)
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_employees_name ON employees(name)")
    conn.commit()
    conn.close()

# 工具：添加员工
class AddEmployeeTool(BaseTool):
    def run(self, params: dict) -> str:
        name = (params.get("name") or "").strip()
        department = (params.get("department") or "").strip()
        title = (params.get("title") or "").strip()
        if not name:
            return "添加失败：员工姓名不能为空"
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO employees (name, department, title)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    department = excluded.department,
                    title = excluded.title
                """,
                (name, department, title),
            )
            conn.commit()
            return f"员工{name}信息已保存"
        except Exception as e:
            return f"添加失败：{str(e)}"
        finally:
            conn.close()

    @property
    def description(self):
        return "添加一名员工信息"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "员工姓名"},
                "department": {"type": "string", "description": "所属部门"},
                "title": {"type": "string", "description": "职务"}
            },
            "required": ["name"]
        }

# 工具：查询员工
class QueryEmployeeTool(BaseTool):
    def run(self, params: dict) -> str:
        name = (params.get("name") or "").strip()
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            if name:
                cursor.execute("SELECT name, department, title FROM employees WHERE name = ?", (name,))
            else:
                cursor.execute("SELECT name, department, title FROM employees ORDER BY id")
            rows = cursor.fetchall()
            if not rows:
                return "未找到匹配员工"
            return "\n".join([f"{n} | {d} | {t}" for n, d, t in rows])
        except Exception as e:
            return f"查询失败：{str(e)}"
        finally:
            conn.close()

    @property
    def description(self):
        return "查询员工信息"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "员工姓名（可选）"}
            }
        }

# 构建Agent
def build_db_agent():
    agent = Agent(
        name="EmployeeDBAgent",
        llm=llm_config(temperature=0.2),
        tools=[AddEmployeeTool(), QueryEmployeeTool()],
        system_message="你是一位人事数据库助手，能添加和查询员工信息，信息包含姓名、部门与职务"
    )
    return agent

# 模拟用户对话流程
def run_demo():
    init_db()
    agent = build_db_agent()

    print(agent.chat("请添加一名员工，姓名是王小明，部门是技术部，职位是高级工程师"))
    print(agent.chat("请查询王小明的详细信息"))
    print(agent.chat("我想看看所有员工列表"))

if __name__ == "__main__":
    run_demo()
