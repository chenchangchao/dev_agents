import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import * as nodemailer from "nodemailer";
import { CronExpressionParser } from "cron-parser";

const server = new McpServer (
  {
    name: "email-mcp-server",
    version: "0.0.1",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// 环境变量配置
const EMAIL_HOST = process.env.EMAIL_HOST || "smtp.qq.com";
const EMAIL_PORT = parseInt(process.env.EMAIL_PORT || "465");
const EMAIL_USER = process.env.EMAIL_USER;
const EMAIL_PASS = process.env.EMAIL_PASS;

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

// Store for scheduled emails
interface ScheduledEmail {
  id: string;
  schedule: string; // cron expression
  emailData: {
    to: string;
    cc?: string;
    subject: string;
    html: string;
    attachments?: Array<{ filename: string; path: string }>;
  };
}

const scheduledEmails: Map<string, ScheduledEmail> = new Map();
const scheduledTimers: Map<string, NodeJS.Timeout> = new Map();

// Calculate next run time based on cron expression
function calculateNextRun(schedule: string): number {
  try {
    const interval = CronExpressionParser.parse(schedule);
    const next = interval.next().getTime();
    const now = new Date().getTime();
    return next - now;
  } catch (error) {
    throw new Error(`Invalid cron expression: ${schedule}`);
  }
}

// Schedule an email
function scheduleEmail(email: ScheduledEmail) {
  try {
    // Clear existing timer if any
    if (scheduledTimers.has(email.id)) {
      clearTimeout(scheduledTimers.get(email.id)!);
    }

    const delay = calculateNextRun(email.schedule);

    // Set up the timer
    const timer = setTimeout(async () => {
      try {
        console.log(`Sending scheduled email: ${email.id}`);
        await sendEmail(email.emailData);
        console.log(`Scheduled email sent successfully: ${email.id}`);

        // Reschedule for next occurrence if it still exists
        if (scheduledEmails.has(email.id)) {
          scheduleEmail(email);
        }
      } catch (error) {
        console.error(`Failed to send scheduled email ${email.id}:`, error);
      }
    }, delay);

    scheduledTimers.set(email.id, timer);
    console.log(
      `Email ${email.id} scheduled to run in ${delay}ms (${new Date(
        Date.now() + delay
      )})`
    );
  } catch (error) {
    console.error(`Failed to schedule email ${email.id}:`, error);
  }
}

// Send email helper function
async function sendEmail(emailData: {
  to: string;
  cc?: string;
  subject: string;
  html: string;
  attachments?: Array<{ filename: string; path: string }>;
}) {
  const mailOptions: any = {
    from: EMAIL_USER,
    to: emailData.to,
    subject: emailData.subject,
    html: emailData.html,
  };

  // Only add cc if it exists and is not empty
  if (emailData.cc && emailData.cc.trim() !== "") {
    mailOptions.cc = emailData.cc;
  }

  // Only add attachments if it exists and is not empty
  if (emailData.attachments && emailData.attachments.length > 0) {
    mailOptions.attachments = emailData.attachments;
  }

  console.log("Sending email with options:", {
    to: mailOptions.to,
    cc: mailOptions.cc,
    subject: mailOptions.subject,
  });

  const info = await transporter.sendMail(mailOptions);
  console.log("Message sent: %s", info.messageId);
  console.log("Server response: %s", info.response);

  return info;
}

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "send-email",
        description: "发送邮件，支持HTML内容、表格和附件",
        inputSchema: {
          type: "object",
          properties: {
            to: {
              type: "string",
              description: "收件人邮箱，多个收件人用逗号分隔",
            },
            cc: {
              type: "string",
              description: "抄送邮箱，多个抄送用逗号分隔",
            },
            subject: {
              type: "string",
              description: "邮件主题",
            },
            html: {
              type: "string",
              description: "邮件HTML内容，支持表格等HTML标签",
            },
            attachments: {
              type: "array",
              description: "附件列表",
              items: {
                type: "object",
                properties: {
                  filename: { type: "string" },
                  path: { type: "string" },
                },
              },
            },
          },
          required: ["to", "subject", "html"],
        },
      },
      {
        name: "schedule-email",
        description: "定时发送邮件",
        inputSchema: {
          type: "object",
          properties: {
            schedule: {
              type: "string",
              description:
                "定时表达式 (cron format, e.g., '45 11 * * *' for every day at 11:45 AM)",
            },
            to: {
              type: "string",
              description: "收件人邮箱，多个收件人用逗号分隔",
            },
            cc: {
              type: "string",
              description: "抄送邮箱，多个抄送用逗号分隔",
            },
            subject: {
              type: "string",
              description: "邮件主题",
            },
            html: {
              type: "string",
              description: "邮件HTML内容，支持表格等HTML标签",
            },
            attachments: {
              type: "array",
              description: "附件列表",
              items: {
                type: "object",
                properties: {
                  filename: { type: "string" },
                  path: { type: "string" },
                },
              },
            },
          },
          required: ["schedule", "to", "subject", "html"],
        },
      },
      {
        name: "list-scheduled-emails",
        description: "列出所有定时邮件任务",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "cancel-scheduled-email",
        description: "取消定时邮件任务",
        inputSchema: {
          type: "object",
          properties: {
            id: {
              type: "string",
              description: "定时邮件任务ID",
            },
          },
          required: ["id"],
        },
      },
    ],
  };
});

// 处理工具请求
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  try {
    if (!request.params.name || !request.params.arguments) {
      return {
        success: false,
        error: "Tool name and arguments are required",
      };
    }

    const args = request.params.arguments as any;

    switch (request.params.name) {
      case "send-email":
        if (!args.to || !args.subject || !args.html) {
          return {
            success: false,
            error:
              "Required fields missing: to, subject, and html are required",
          };
        }

        const info = await sendEmail(args);
        return {
          success: true,
          data: {
            messageId: info.messageId,
            response: info.response,
          },
        };

      case "schedule-email":
        if (!args.schedule || !args.to || !args.subject || !args.html) {
          return {
            success: false,
            error:
              "Required fields missing: schedule, to, subject, and html are required",
          };
        }

        const emailId = `email_${Date.now()}_${Math.random()
          .toString(36)
          .substr(2, 9)}`;
        const scheduledEmail: ScheduledEmail = {
          id: emailId,
          schedule: args.schedule,
          emailData: {
            to: args.to,
            cc: args.cc,
            subject: args.subject,
            html: args.html,
            attachments: args.attachments,
          },
        };

        scheduledEmails.set(emailId, scheduledEmail);
        scheduleEmail(scheduledEmail);

        return {
          success: true,
          data: {
            id: emailId,
            message: `Email scheduled successfully with ID: ${emailId}`,
          },
        };

      case "list-scheduled-emails":
        const emails = Array.from(scheduledEmails.values()).map((email) => ({
          id: email.id,
          schedule: email.schedule,
          to: email.emailData.to,
          subject: email.emailData.subject,
        }));

        return {
          success: true,
          data: {
            scheduledEmails: emails,
          },
        };

      case "cancel-scheduled-email":
        if (!args.id) {
          return {
            success: false,
            error: "ID is required",
          };
        }

        if (scheduledEmails.has(args.id)) {
          // Clear the timer
          if (scheduledTimers.has(args.id)) {
            clearTimeout(scheduledTimers.get(args.id)!);
            scheduledTimers.delete(args.id);
          }

          // Remove from storage
          scheduledEmails.delete(args.id);

          return {
            success: true,
            data: {
              message: `Scheduled email ${args.id} cancelled successfully`,
            },
          };
        } else {
          return {
            success: false,
            error: `Scheduled email with ID ${args.id} not found`,
          };
        }

      default:
        return {
          success: false,
          error: `Unknown tool: ${request.params.name}`,
        };
    }
  } catch (error) {
    console.error("Error processing tool request:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error occurred",
    };
  }
});

// 启动服务
async function main() {
  // Verify SMTP connection
  transporter.verify((error, success) => {
    if (error) {
      console.error("SMTP connection error:", error);
    } else {
      console.log("SMTP server is ready to take our messages");
    }
  });

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Email MCP Server running on stdio");
}

// Cleanup function to clear timers on exit
process.on("SIGINT", () => {
  console.log("Cleaning up scheduled emails...");
  for (const timer of scheduledTimers.values()) {
    clearTimeout(timer);
  }
  process.exit(0);
});

// 异常捕获
main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
