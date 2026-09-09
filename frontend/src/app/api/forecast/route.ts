const BACKEND_URL =
  process.env.BACKEND_API_URL || "http://3.80.207.155:8000";

export async function POST(request: Request) {
  try {
    const body = await request.text();

    const response = await fetch(`${BACKEND_URL}/forecast`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body,
      cache: "no-store",
    });

    const responseBody = await response.text();

    return new Response(responseBody, {
      status: response.status,
      headers: {
        "Content-Type":
          response.headers.get("Content-Type") || "application/json",
      },
    });
  } catch (error) {
    console.error("Forecast proxy error:", error);

    return Response.json(
      { detail: "Unable to reach forecast backend" },
      { status: 502 }
    );
  }
}
