---
name: e2e-testing
description: "Playwright E2E testing patterns — Page Object Model, multi-browser config, flaky test management, CI/CD integration. Use when writing or debugging end-to-end tests."
---

# E2E Testing (Playwright)

Comprehensive Playwright patterns for building stable, fast, maintainable E2E test suites.

## Use this skill when

- Writing end-to-end tests for web applications
- Setting up Playwright test infrastructure
- Debugging flaky tests
- Implementing Page Object Model (POM) patterns
- Configuring CI/CD test pipelines
- Testing OAuth login flows

## Test Organization

```
tests/e2e/
├── auth/
│   ├── login.spec.ts       # OAuth login flow
│   └── logout.spec.ts      # Session cleanup
├── features/
│   ├── simulation.spec.ts  # Create/run simulations
│   ├── strategy.spec.ts    # CRUD strategies
│   └── portfolio.spec.ts   # Portfolio views
├── pages/                  # Page Object Models
│   ├── LoginPage.ts
│   ├── DashboardPage.ts
│   └── SimulationPage.ts
├── fixtures/
│   ├── auth.fixture.ts     # Pre-authenticated state
│   └── data.fixture.ts     # Test data setup
└── playwright.config.ts
```

## Page Object Model

```typescript
export class LoginPage {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto('/login');
  }

  async loginWithGoogle() {
    await this.page.getByRole('button', { name: /google/i }).click();
    await this.page.waitForURL('/dashboard');
  }

  async loginWithGitHub() {
    await this.page.getByRole('button', { name: /github/i }).click();
    await this.page.waitForURL('/dashboard');
  }

  async expectLoggedIn(name: string) {
    await expect(this.page.getByText(name)).toBeVisible();
  }
}
```

## Configuration

```typescript
// playwright.config.ts
export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  retries: process.env.CI ? 2 : 0,
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'mobile', use: { ...devices['iPhone 14'] } },
  ],
  webServer: {
    command: 'npm run dev',
    port: 3000,
    reuseExistingServer: !process.env.CI,
  },
});
```

## Selector Strategy

Priority order:
1. `getByRole()` — accessibility-first
2. `getByText()` — visible text content
3. `getByTestId()` — `data-testid` attribute (last resort)

```typescript
// GOOD
await page.getByRole('button', { name: 'Create Simulation' }).click();

// ACCEPTABLE
await page.getByTestId('simulation-form').fill('...');

// BAD — fragile selectors
await page.locator('.btn-primary.submit-btn').click();
```

## Flaky Test Management

- Tag flaky tests: `test.fixme('reason')` or `test.skip()`
- Detect with: `npx playwright test --repeat-each=5 --retries=0`
- Common causes:
  - Race conditions → add `waitForResponse()` or `waitForSelector()`
  - Animation delays → use `animations: 'disabled'` in config
  - Network timing → mock API responses with `page.route()`

## OAuth Testing

```typescript
// Mock OAuth for testing — intercept the redirect
test('login with Google', async ({ page }) => {
  // Mock the OAuth callback
  await page.route('/api/auth/callback/google*', async (route) => {
    await route.fulfill({
      status: 302,
      headers: {
        'Location': '/dashboard',
        'Set-Cookie': 'atlas_token=mock_jwt; Path=/; HttpOnly',
      },
    });
  });

  await page.goto('/login');
  await page.getByRole('button', { name: /google/i }).click();
  await expect(page).toHaveURL('/dashboard');
});
```

## References

- Source: [affaan-m/everything-claude-code/e2e-testing](https://github.com/affaan-m/everything-claude-code/tree/main/skills/e2e-testing)
