import { spawn, spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
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

async function postExtract({ url, html, removeSelectors }) {
  const payload = { url, html };
  if (removeSelectors !== undefined) payload.remove_selectors = removeSelectors;
  const response = await fetch(`${baseUrl}/extract`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Extract returned HTTP ${response.status}: ${await response.text()}`);
  }
  return response.json();
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
  const body = await postExtract({
    url: "https://example.invalid/article",
    html,
  });
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

  const fixtures = JSON.parse(
    readFileSync(
      new URL("../../evaluation/phase4e_domain_rule_selection_fixtures.json", import.meta.url),
      "utf8",
    ),
  );
  const boilerplate = fixtures.fixtures.find((fixture) => fixture.id === "boilerplate_pollution");
  if (!boilerplate) throw new Error("Phase 4E boilerplate fixture missing");

  const baseline = await postExtract({
    url: boilerplate.url,
    html: boilerplate.static_html,
  });
  if (!baseline.markdown.replace(/\\_/g, "_").includes("PHASE4E_BOILER_FORBIDDEN")) {
    throw new Error("Phase 4E boilerplate baseline no longer demonstrates extraction noise");
  }

  const cleaned = await postExtract({
    url: boilerplate.url,
    html: boilerplate.static_html,
    removeSelectors: boilerplate.rules.remove_selectors,
  });
  const cleanedMarkers = cleaned.markdown.replace(/\\_/g, "_");
  if (cleanedMarkers.includes("PHASE4E_BOILER_FORBIDDEN")) {
    throw new Error("Configured remove_selectors did not remove boilerplate noise");
  }
  for (const marker of boilerplate.required_markers) {
    if (!cleanedMarkers.includes(marker)) {
      throw new Error(`Configured remove_selectors removed required marker ${marker}`);
    }
  }

  const missing = await postExtract({
    url: boilerplate.url,
    html: boilerplate.static_html,
    removeSelectors: [fixtures.fail_open_cases.missing_remove_selector],
  });
  if (JSON.stringify(missing) !== JSON.stringify(baseline)) {
    throw new Error("Missing remove selector did not preserve baseline extraction");
  }

  const atomicFailOpen = await postExtract({
    url: boilerplate.url,
    html: boilerplate.static_html,
    removeSelectors: [
      ...boilerplate.rules.remove_selectors,
      fixtures.fail_open_cases.invalid_remove_selector,
    ],
  });
  if (JSON.stringify(atomicFailOpen) !== JSON.stringify(baseline)) {
    throw new Error("Mixed valid/invalid remove selectors did not fail open atomically");
  }

  const renderedCleaned = await postExtract({
    url: boilerplate.url,
    html: boilerplate.rendered_html,
    removeSelectors: boilerplate.rules.remove_selectors,
  });
  if (renderedCleaned.markdown.replace(/\\_/g, "_").includes("PHASE4E_BOILER_FORBIDDEN")) {
    throw new Error("remove_selectors did not apply to the frozen rendered fixture HTML");
  }

  const phase4e = spawnSync(
    process.execPath,
    ["../../evaluation/phase4e_domain_rule_selection.mjs"],
    { stdio: "inherit" },
  );
  if (phase4e.status !== 0) {
    throw new Error(`Phase 4E evaluator failed with exit code ${phase4e.status}`);
  }

  const phase4g = spawnSync(
    process.execPath,
    ["../../evaluation/phase4g_domain_rule_admission.mjs"],
    { stdio: "inherit" },
  );
  if (phase4g.status !== 0) {
    throw new Error(`Phase 4G evaluator failed with exit code ${phase4g.status}`);
  }
} finally {
  if (child.exitCode === null) child.kill("SIGTERM");
}
