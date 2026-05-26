import * as nodemailer from "nodemailer";
import dotenv from "dotenv";

// 加载环境变量
dotenv.config();

// --------------------------
// 类型定义
// --------------------------
interface EmailConfig {
  host: string;
  port: number;
  secure: boolean;
  auth: {
    user: string;
    pass: string;
  };
  from: string;
}

interface TemplateData {
  [key: string]: string;
}

interface SmtpError {
  code?: string;
  command?: string;
  address?: string;
  port?: number;
  response?: string;
  responseCode?: number;
}

// --------------------------
// 环境变量验证与配置
// --------------------------
const EMAIL_HOST = process.env.GMAIL_HOST || "smtp.qq.com";
const EMAIL_PORT = parseInt(process.env.GMAIL_PORT || "465");
const EMAIL_USER = process.env.GMAIL_USER;
const EMAIL_PASS = process.env.GMAIL_PASS;
const EMAIL_FROM = process.env.GMAIL_FROM;
const EMAIL_GOTO = process.env.GMAIL_GOTO;
const EMAIL_SUBJECT_TPL = process.env.GMAIL_MAIL_SUBJECT;
const EMAIL_CONTENT_TPL = process.env.GMAIL_MAIL_CONTENT;

// 严格验证所有必要环境变量
if (!EMAIL_USER || !EMAIL_PASS || !EMAIL_FROM || !EMAIL_GOTO || !EMAIL_SUBJECT_TPL || !EMAIL_CONTENT_TPL) {
  console.error("❌ 缺少必要的邮件环境变量，请检查.env文件");
  console.error("需要配置：GMAIL_USER, GMAIL_PASS, GMAIL_FROM, GMAIL_GOTO, GMAIL_SUBJECT, GMAIL_CONTENT");
  process.exit(1);
}

const emailConfig: EmailConfig = {
  host: EMAIL_HOST,
  port: EMAIL_PORT,
  secure: EMAIL_PORT === 465, // 自动根据端口判断是否使用SSL
  auth: {
    user: EMAIL_USER,
    pass: EMAIL_PASS,
  },
  from: EMAIL_FROM,
};

function getSmtpErrorHint(err: unknown): string | null {
  const smtpError = err as SmtpError;

  if (
    smtpError.code === "ESOCKET" &&
    smtpError.command === "CONN" &&
    smtpError.port === 465
  ) {
    return [
      "提示：当前连接 smtp.gmail.com:465 被拒绝。",
      "可以尝试改用 Gmail STARTTLS 端口：GMAIL_PORT=587（secure 会自动变为 false）。",
      "如果 587 也失败，通常是当前网络/代理/防火墙阻止了 SMTP 出站连接。",
    ].join("\n");
  }

  if (
    smtpError.responseCode === 534 &&
    smtpError.response?.includes("Application-specific password required")
  ) {
    return [
      "提示：Gmail 要求使用应用专用密码。",
      "请把 GMAIL_PASS 设置为 Google 账号生成的 16 位 App Password，不要使用普通登录密码。",
    ].join("\n");
  }

  return null;
}

function logSmtpError(message: string, err: unknown): void {
  console.error(message, err);

  const hint = getSmtpErrorHint(err);
  if (hint) {
    console.error(hint);
  }
}

// --------------------------
// 邮件服务类
// --------------------------
class EmailService {
  private transporter: nodemailer.Transporter;
  private config: EmailConfig;

  constructor(config: EmailConfig) {
    this.config = config;
    this.transporter = nodemailer.createTransport(config);
  }

  /**
   * 验证SMTP服务器连接是否正常
   */
  private async verifyConnection(): Promise<void> {
    try {
      await this.transporter.verify();
      console.log("✅ SMTP服务器连接成功");
    } catch (err) {
      logSmtpError("❌ SMTP服务器连接失败:", err);
      throw err;
    }
  }

  /**
   * 生成6位安全验证码（去掉易混淆字符0、O、1、I）
   */
  generateCheckCode(length: number = 6): string {
    const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    let code = "";
    for (let i = 0; i < length; i++) {
      code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return code;
  }

  /**
   * 通用模板渲染函数，支持{{变量名}}语法
   */
  renderTemplate(template: string, data: TemplateData): string {
    return template.replace(/{{(\w+)}}/g, (match, key) => {
      return data[key] || match;
    });
  }

  /**
   * 统一的HTML邮件模板
   */
  private renderHtmlTemplate(content: string): string {
    return `
      <!DOCTYPE html>
      <html lang="zh-CN">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>系统邮件</title>
          <style>
            * {
              margin: 0;
              padding: 0;
              box-sizing: border-box;
            }
            body {
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
              background-color: #f5f7fa;
              padding: 20px;
            }
            .email-container {
              max-width: 600px;
              margin: 0 auto;
              background-color: #ffffff;
              border-radius: 8px;
              overflow: hidden;
              box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
            }
            .email-header {
              background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
              color: #ffffff;
              padding: 24px;
              text-align: center;
            }
            .email-header h1 {
              font-size: 24px;
              font-weight: 600;
            }
            .email-body {
              padding: 32px 24px;
              line-height: 1.8;
              color: #333333;
              font-size: 16px;
            }
            .email-body p {
              margin-bottom: 16px;
            }
            .code-box {
              background-color: #f0f7ff;
              border: 1px solid #d0e8ff;
              border-radius: 6px;
              padding: 16px;
              text-align: center;
              margin: 24px 0;
            }
            .code {
              font-size: 28px;
              font-weight: 700;
              color: #1890ff;
              letter-spacing: 8px;
            }
            .tip {
              color: #666666;
              font-size: 14px;
              margin-top: 24px;
            }
            .email-footer {
              background-color: #f8f9fa;
              color: #8c8c8c;
              padding: 16px 24px;
              text-align: center;
              font-size: 14px;
              border-top: 1px solid #e8e8e8;
            }
          </style>
        </head>
        <body>
          <div class="email-container">
            <div class="email-header">
              <h1>MCP 系统通知</h1>
            </div>
            <div class="email-body">
              ${content}
            </div>
            <div class="email-footer">
              <p>此邮件由系统自动发送，请勿回复</p>
              <p>© 2026 MCP 平台 版权所有</p>
            </div>
          </div>
        </body>
      </html>
    `;
  }

  /**
   * 发送验证码邮件
   * @param to 收件人邮箱
   * @param code 验证码
   * @param expireMinutes 有效期（分钟），默认5分钟
   */
  async sendVerificationEmail(to: string, code: string, expireMinutes: number = 5): Promise<boolean> {
    try {
      // 渲染主题和内容模板
      const subject = this.renderTemplate(EMAIL_SUBJECT_TPL!, { CODE: code });
      const textContent = this.renderTemplate(EMAIL_CONTENT_TPL!, { 
        CODE: code, 
        EXPIRE_TIME: `${expireMinutes}分钟` 
      });

      // 渲染HTML内容
      const htmlContent = this.renderHtmlTemplate(`
        <p>您好！</p>
        <p>您正在进行MCP平台的身份验证操作，验证码如下：</p>
        <div class="code-box">
          <div class="code">${code}</div>
        </div>
        <p>验证码有效期为 <strong>${expireMinutes}分钟</strong>，请尽快完成验证。</p>
        <p>请勿将验证码泄露给任何人，包括平台工作人员。</p>
        <p class="tip">如非本人操作，请忽略此邮件并及时修改您的账户密码。</p>
      `);

      // 发送邮件
      const info = await this.transporter.sendMail({
        from: this.config.from,
        to: to,
        subject: subject,
        text: textContent,
        html: htmlContent,
      });

      console.log(`✅ 验证码邮件发送成功，收件人：${to}，MessageId：${info.messageId}`);
      return true;
    } catch (err) {
      logSmtpError(`❌ 验证码邮件发送失败，收件人：${to}`, err);
      return false;
    }
  }

  /**
   * 通用邮件发送方法（支持自定义主题和内容）
   */
  async sendEmail(to: string, subject: string, text: string, html?: string): Promise<boolean> {
    try {
      const info = await this.transporter.sendMail({
        from: this.config.from,
        to: to,
        subject: subject,
        text: text,
        html: html || this.renderHtmlTemplate(`<p>${text}</p>`),
      });

      console.log(`✅ 邮件发送成功，收件人：${to}，MessageId：${info.messageId}`);
      return true;
    } catch (err) {
      logSmtpError(`❌ 邮件发送失败，收件人：${to}`, err);
      return false;
    }
  }

  async ready(): Promise<void> {
    await this.verifyConnection();
  }
}

// --------------------------
// 创建并导出单例服务
// --------------------------
const emailService = new EmailService(emailConfig);
export default emailService;

// --------------------------
// 测试代码（仅在直接运行此文件时执行）
// --------------------------
if (require.main === module) {
  (async () => {
    console.log("🚀 开始测试邮件服务...");
    await emailService.ready();
    
    // 生成验证码
    const checkCode = emailService.generateCheckCode();
    console.log(`生成的验证码：${checkCode}`);
    
    // 发送测试邮件
    const success = await emailService.sendVerificationEmail(EMAIL_GOTO, checkCode);
    
    if (success) {
      console.log("🎉 邮件服务测试完成！");
    } else {
      console.log("❌ 邮件服务测试失败");
      process.exit(1);
    }
  })();
}
