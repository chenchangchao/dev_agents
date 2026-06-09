export function textEventStream(text: string, chunkSize = 12): Response {
  const encoder = new TextEncoder();
  let cursor = 0;

  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (cursor >= text.length) {
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
        return;
      }
      const chunk = text.slice(cursor, cursor + chunkSize);
      cursor += chunkSize;
      controller.enqueue(encoder.encode(`data: ${JSON.stringify({ delta: chunk })}\n\n`));
    }
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache"
    }
  });
}

export function openAiCompatibleEventStream(
  text: string,
  model: string,
  chunkSize = 12
): Response {
  const encoder = new TextEncoder();
  const id = `chatcmpl-${crypto.randomUUID()}`;
  let cursor = 0;

  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (cursor >= text.length) {
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              id,
              object: "chat.completion.chunk",
              model,
              choices: [{ index: 0, delta: {}, finish_reason: "stop" }]
            })}\n\n`
          )
        );
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
        return;
      }

      const chunk = text.slice(cursor, cursor + chunkSize);
      cursor += chunkSize;
      controller.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({
            id,
            object: "chat.completion.chunk",
            model,
            choices: [{ index: 0, delta: { content: chunk }, finish_reason: null }]
          })}\n\n`
        )
      );
    }
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache",
      Connection: "keep-alive"
    }
  });
}
