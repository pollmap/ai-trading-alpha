import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const PUBLIC_PATHS = ["/login", "/api/auth"];

/** Static file extensions that do not require authentication. */
const STATIC_EXT = /\.(ico|png|jpg|jpeg|gif|svg|css|js|woff2?|ttf|eot|map)$/;

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Allow public paths
  if (PUBLIC_PATHS.some((p) => pathname.startsWith(p))) {
    return NextResponse.next();
  }

  // Allow static assets and Next.js internals (explicit extension allowlist)
  if (
    pathname.startsWith("/_next") ||
    pathname.startsWith("/favicon") ||
    STATIC_EXT.test(pathname)
  ) {
    return NextResponse.next();
  }

  // Check for auth cookie
  const token = request.cookies.get("atlas_token");
  if (!token) {
    const loginUrl = new URL("/login", request.url);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
