---
name: nextjs-app-router-patterns
description: "Master Next.js 14+ App Router with Server Components, streaming, parallel routes, and advanced data fetching. Use when building Next.js pages, implementing auth UI, or optimizing React Server Components."
---

# Next.js App Router Patterns

Comprehensive patterns for Next.js 14+ App Router architecture, Server Components, and modern full-stack React development.

## Use this skill when

- Building new Next.js pages or layouts with App Router
- Implementing authentication UI (login, signup, protected routes)
- Creating client-side state management (React Context)
- Setting up route middleware for auth guards
- Optimizing data fetching and caching
- Building full-stack features with Server Actions

## Do not use this skill when

- Working on Python backend code
- Working with Pages Router (legacy)

## Core Patterns

### Route Protection with Middleware
```typescript
// middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const token = request.cookies.get('atlas_token');
  const isPublic = ['/', '/login'].includes(request.nextUrl.pathname);

  if (!token && !isPublic) {
    return NextResponse.redirect(new URL('/login', request.url));
  }
  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
};
```

### Auth Context Provider
```typescript
'use client';
import { createContext, useContext, useEffect, useState } from 'react';

interface User { tenant_id: string; email: string; name: string; plan: string; }
interface AuthCtx { user: User | null; loading: boolean; login: (p: string) => void; logout: () => Promise<void>; }

const AuthContext = createContext<AuthCtx | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/auth/me', { credentials: 'include' })
      .then(r => r.ok ? r.json() : null)
      .then(setUser)
      .finally(() => setLoading(false));
  }, []);

  const login = (provider: string) => { window.location.href = `/api/auth/login/${provider}`; };
  const logout = async () => {
    await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' });
    setUser(null);
    window.location.href = '/';
  };

  return <AuthContext.Provider value={{ user, loading, login, logout }}>{children}</AuthContext.Provider>;
}

export const useAuth = () => { const c = useContext(AuthContext); if (!c) throw new Error('wrap in AuthProvider'); return c; };
```

### Server vs Client Components
- Default to Server Components (no 'use client')
- Add 'use client' only for: useState, useEffect, event handlers, browser APIs
- Keep data fetching in Server Components
- Pass data down as props to Client Components

### Layout with Auth
```typescript
// app/layout.tsx
import { AuthProvider } from '@/contexts/AuthContext';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko" className="dark">
      <body>
        <AuthProvider>
          <Sidebar />
          <main>{children}</main>
        </AuthProvider>
      </body>
    </html>
  );
}
```

### API Calls with Credentials
```typescript
// Always include credentials for cookie-based auth
const res = await fetch('/api/portfolios', { credentials: 'include' });
```

## Key Principles

1. **Server-first**: Default to Server Components, opt into client only when needed
2. **Streaming**: Use `loading.tsx` and Suspense for progressive rendering
3. **Colocation**: Keep related files together (`page.tsx`, `loading.tsx`, `error.tsx`)
4. **Type safety**: Full TypeScript, no `any`
5. **Responsive**: Mobile-first with Tailwind breakpoints

## References

- Source: [antigravity-awesome-skills/nextjs-app-router-patterns](https://github.com/sickn33/antigravity-awesome-skills/tree/main/skills/nextjs-app-router-patterns)
