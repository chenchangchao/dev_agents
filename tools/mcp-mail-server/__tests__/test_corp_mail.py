'''
企业邮箱测试脚本，适用于EWS协议（即Exchange原生协议），无需SMTP，
仅在公司内部使用，不支持外网测试
'''

from exchangelib import Credentials, Account, Message, HTMLBody, Configuration
import traceback


# ===================== 【仅需修改这里的密码！】 =====================
# 你的完整邮箱（和OWA登录完全一致）
SENDER_EMAIL = "your_corp_mail@corp.mail"
# 你的域账号/邮箱登录密码（和OWA网页登录密码完全相同）
SENDER_PASSWORD = "your_password"
# 同事的收件邮箱
RECEIVER_EMAIL = ["receiver1@corp.mail", "receiver2@corp.mail"]
# =================================================================

# Exchange EWS配置（完全适配你的OWA入口，无需修改）
EWS_SERVER = "your_ews_server"  # 例如 "mail.corp.qihoo.net"
EWS_URL = f"https://{EWS_SERVER}/EWS/Exchange.asmx"
def main():
    try:
        # 1. 配置企业AD域认证（360内部Exchange强制用NTLM认证）
        credentials = Credentials(SENDER_EMAIL, SENDER_PASSWORD)
        config = Configuration(
            server=EWS_SERVER,
            credentials=credentials,
            auth_type="NTLM"
        )

        # 2. 登录Exchange账户（手动配置，不走自动发现，避免内网适配问题）
        account = Account(
            primary_smtp_address=SENDER_EMAIL,
            config=config,
            autodiscover=False,
            access_type="delegate"  # OWA能登录就默认拥有该权限
        )

        # 3. 构建邮件内容（支持HTML/纯文本两种格式）
        # 👉 纯文本版本：直接把HTMLBody换成普通字符串，如 body="这是纯文本测试邮件"
        html_content = """
        <h3>Python EWS测试邮件</h3>
        <p>这是通过Exchange原生EWS协议发送的测试邮件，无需SMTP，完全适配企业邮箱</p>
        <p>发送时间：2026年3月26日</p>
        """

        # 4. 创建邮件对象
        msg = Message(
            account=account,
            subject="Python EWS测试邮件 - 企业Exchange",
            body=HTMLBody(html_content),
            to_recipients=RECEIVER_EMAIL  # 支持群发，格式为 [A@360.cn, B@360.cn]
        )

        # 5. 发送邮件！
        msg.send()
        print("✅ 邮件发送成功！请通知同事查收~")

    except Exception as e:
        print(f"❌ 发送失败，错误原因：{str(e)}")
        print("\n完整错误栈（方便排查）：")
        traceback.print_exc()

if __name__ == "__main__":
    main()