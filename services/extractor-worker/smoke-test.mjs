import { spawn } from "node:child_process";
import process from "node:process";
import { setTimeout as sleep } from "node:timers/promises";

const port = 31991;
const baseUrl = `http://127.0.0.1:${port}`;
const child = spawn(process.execPath, ["server.mjs"], {
  env: {
    ...process.env,
    PORT: String(port),
    BROWSER_ENABLED: "false",
  },
  stdio: ["ignore", "pipe", "pipe"],
});

let output = "";
child.stdout.on("data", (chunk) => {
  output += chunk.toString();
});
child.stderr.on("data", (chunk) => {
  output += chunk.toString();
});

async function waitForHealth() {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (child.exitCode !== null) {
      throw new Error(`Worker exited early (${child.exitCode})\n${output}`);
    }
    try {
      const response = await fetch(`${baseUrl}/health`);
      if (response.ok) return;
    } catch {
      // Server is still starting.
    }
    await sleep(100);
  }
  throw new Error(`Worker health endpoint did not become ready\n${output}`);
}

const html = `<!doctype html>
<html>
  <head><title>SupraCrawl extraction fixture</title></head>
  <body>
    <nav>Home Products Pricing Login</nav>
    <article>
      <h1>SupraCrawl extraction fixture</h1>
      <p>SupraCrawl should preserve the main article while removing navigation and scripts.</p>
      <h2>Structured section</h2>
      <p>This paragraph exists to verify that Mozilla Readability returns useful prose.</p>
      <ul><li>first useful item</li><li>second useful item</li></ul>
      <pre><code>print("useful code")</code></pre>
    </article>
    <script>throw new Error("this script must never become model context")</script>
  </body>
</html>`;

try {
  await waitForHealth();
  const response = await fetch(`${baseUrl}/extract`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      url: "https://example.invalid/article",
      html,
    }),
  });
  if (!response.ok) {
    throw new Error(`Extract returned HTTP ${response.status}: ${await response.text()}`);
  }

  const body = await response.json();
  if (!body.title.includes("SupraCrawl extraction fixture")) {
    throw new Error(`Unexpected extracted title: ${JSON.stringify(body.title)}`);
  }
  if (!body.markdown.includes("Structured section")) {
    throw new Error(`Expected section missing from Markdown: ${body.markdown}`);
  }
  if (!body.markdown.includes("useful code")) {
    throw new Error(`Expected code missing from Markdown: ${body.markdown}`);
  }
  if (body.markdown.includes("this script must never become model context")) {
    throw new Error("Script content leaked into extracted Markdown");
  }
} finally {
  if (child.exitCode === null) child.kill("SIGTERM");
}
