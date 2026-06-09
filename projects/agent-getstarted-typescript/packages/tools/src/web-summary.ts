export interface HtmlSummary {
  title: string;
  description: string;
  text: string;
}

export function summarizeHtml(html: string): HtmlSummary {
  const title = html.match(/<title[^>]*>(.*?)<\/title>/is)?.[1]?.trim() ?? "";
  const description =
    html.match(/<meta\s+name=["']description["']\s+content=["'](.*?)["']/is)?.[1]?.trim() ??
    html.match(/<meta\s+content=["'](.*?)["']\s+name=["']description["']/is)?.[1]?.trim() ??
    "";
  const text = html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return { title, description, text };
}
