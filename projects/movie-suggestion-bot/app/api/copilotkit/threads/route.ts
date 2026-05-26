export const GET = async () => {
  return Response.json(
    {
      threads: [],
      nextCursor: null,
    },
    {
      headers: {
        "Cache-Control": "no-store",
      },
    }
  );
};
