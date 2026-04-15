// Shared FORMA fetch wrapper. Sends the httpOnly JWT cookie with every
// request and parses JSON errors into useful Error messages.

export class ApiError extends Error {
  status: number;
  code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

export async function api<T = unknown>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const res = await fetch(path, {
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(init.headers || {}),
    },
    ...init,
  });

  if (res.status === 204) return undefined as T;

  const contentType = res.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  const body = isJson ? await res.json().catch(() => ({})) : await res.text();

  if (!res.ok) {
    const code =
      isJson && typeof body === "object" && body && "error" in body
        ? String((body as Record<string, unknown>).error)
        : undefined;
    const msg = code || (typeof body === "string" ? body : res.statusText);
    throw new ApiError(msg, res.status, code);
  }

  return body as T;
}
