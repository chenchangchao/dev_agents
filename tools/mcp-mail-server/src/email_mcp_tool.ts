import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import * as nodemailer from "nodemailer";
import { z } from "zod";

const server = new McpServer({
  name: "email-mcp-server",
  version: "0.0.1",
});

// 环境变量配置
const EMAIL_HOST = process.env.EMAIL_HOST || "smtp.qq.com";
const EMAIL_PORT = parseInt(process.env.EMAIL_PORT || "465", 10);
const EMAIL_USER = process.env.EMAIL_USER;
const EMAIL_PASS = process.env.EMAIL_PASS;
const EMAIL_FROM = process.env.EMAIL_FROM || EMAIL_USER;

if (!EMAIL_USER || !EMAIL_PASS) {
  console.error("EMAIL_USER or EMAIL_PASS environment variable is not set");
  process.exit(1);
}

// 创建邮件传输器
const transporter = nodemailer.createTransport({
  host: EMAIL_HOST,
  port: EMAIL_PORT,
  secure: EMAIL_PORT === 465,
  auth: {
    user: EMAIL_USER,
    pass: EMAIL_PASS,
  },
});

server.registerTool(
  "send-email",
  {
    title: "Send Email",
    description: "发送邮件，支持HTML内容、表格和附件",
    inputSchema: {
      to: z.string().describe("收件人邮箱，多个收件人用逗号分隔"),
      cc: z.string().optional().describe("抄送邮箱，多个抄送用逗号分隔"),
      subject: z.string().describe("邮件主题"),
      html: z.string().describe("邮件HTML内容，支持表格等HTML标签"),
      attachments: z
        .array(
          z.object({
            filename: z.string(),
            path: z.string(),
          })
        )
        .optional()
        .describe("附件列表"),
    },
  },
  async (args) => {
    try {
      const mailOptions: nodemailer.SendMailOptions = {
        from: EMAIL_FROM,
        to: args.to,
        subject: args.subject,
        html: args.html,
      };

      if (args.cc?.trim()) {
        mailOptions.cc = args.cc;
      }

      if (args.attachments?.length) {
        mailOptions.attachments = args.attachments;
      }

      console.error("Sending email with options:", {
        to: mailOptions.to,
        cc: mailOptions.cc,
        subject: mailOptions.subject,
      });

      const info = await transporter.sendMail(mailOptions);
      console.error("Message sent: %s", info.messageId);
      console.error("Server response: %s", info.response);

      const previewUrl = nodemailer.getTestMessageUrl(info);
      if (previewUrl) {
        console.error("Preview URL: %s", previewUrl);
      }

      return {
        content: [
          {
            type: "text" as const,
            text: `Email sent successfully. Message ID: ${info.messageId}`,
          },
        ],
      };
    } catch (error) {
      return {
        isError: true,
        content: [
          {
            type: "text" as const,
            text:
              error instanceof Error
                ? error.message
                : "Unknown error occurred",
          },
        ],
      };
    }
  }
);

// 启动服务
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Email MCP Server running on stdio");
}

// 异常捕获
main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
