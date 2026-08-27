import dns from "node:dns/promises";
import process from "node:process";

import { Readability } from "@mozilla/readability";
import express from "express";
import ipaddr from "ipaddr.js";
import { JSDOM } from "jsdom";
import { chromium } from "playwright";
import TurndownService from "turndown";

const app = express();
const port = Number(process.env.PORT || 3000);
const browserEnabled = String(process.env.BROWSER_ENABLED || "false").toLowerCase() === "true";
const MAX_HTML_CHARS = 6_000_000;

app.use(express.json({ limit: "7mb" }));

let browserPromise = null;

function normalizeAddress(address) {
  let parsed = ipaddr.parse(address);
  if (parsed.kind() === "ipv6" && parsed.isIPv4MappedAddress()) {
    parsed = parsed.toIPv4Address();
  }
  return parsed;
}

function isForbiddenAddress(address) {
  const range = normalizeAddress(address).range();
  return new Set([
    "private",
    "loopback",
    "linkLocal",
    "uniqueLocal",
    "multicast",
    "unspecified",
    "reserved",
    "carrierGradeNat",
  ]).has(range);
}

async function validatePublicUrl(input) {
  let url;
  try {
    url = new URL(input);
  } catch {
    throw new Error("Invalid URL");
  }
  if (!new Set(["http:", "https:"]).has(url.protocol)) {
    throw new Error("Only HTTP(S) URLs are allowed");
  }
  if (url.username || url.password) {
    throw new Error("URL credentials are not allowed");
  }

  const host = url.hostname.replace(/\.$/, "").toLowerCase();
  if (host === "localhost" || host.endsWith(".localhost") || host.endsWith(".local") || host === "metadata.google.internal") {
    throw new Error("Local hostnames are not allowed");
  }

  if (ipaddr.isValid(host)) {
    if (isForbiddenAddress(host)) throw new Error("Private/non-routable address blocked");
    return url;
  }

  const answers = await dns.lookup(host, { all: true, verbatim: true });
  if (!answers.length || answers.some(({ address }) => isForbiddenAddress(address))) {
    throw new Error("Hostname resolves to a private/non-routable address");
  }
  return url;
}

function cleanMarkdown(markdown) {
  return markdown
    .replace(/\u00a0/g, " ")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{4,}/g, "\n\n\n")
    .trim();
}

function extractReadable(html, url) {
  if (typeof html !== "string" || html.length === 0 || html.length > MAX_HTML_CHARS) {
    throw new Error("HTML payload is empty or exceeds the worker limit");
  }

  const dom = new JSDOM(html, { url });
  const document = dom.window.document;
  for (const selector of ["script", "style", "noscript", "template", "svg", "canvas", "form", "iframe"]) {
    for (const element of document.querySelectorAll(selector)) element.remove();
  }

  const article = new Readability(document, { keepClasses: false }).parse();
  if (!article?.content) {
    return { title: document.title || "", markdown: "", text: "" };
  }

  const turndown = new TurndownService({
    headingStyle: "atx",
    bulletListMarker: "-",
    codeBlockStyle: "fenced",
    emDelimiter: "_",
  });
  const markdown = cleanMarkdown(turndown.turndown(article.content));
  return {
    title: String(article.title || document.title || "").trim(),
    markdown,
    text: String(article.textContent || "").trim(),
  };
}

async function getBrowser() {
  if (!browserEnabled) throw new Error("Browser rendering is disabled");
  browserPromise ??= chromium.launch({ headless: true });
  return browserPromise;
}

async function renderHtml(inputUrl) {
  const url = await validatePublicUrl(inputUrl);
  const browser = await getBrowser();
  const context = await browser.newContext({
    javaScriptEnabled: true,
    serviceWorkers: "block",
    acceptDownloads: false,
  });

  try {
    await context.route("**/*", async (route) => {
      const requestUrl = route.request().url();
      if (requestUrl.startsWith("data:") || requestUrl.startsWith("blob:") || requestUrl.startsWith("about:")) {
        return route.continue();
      }
      try {
        await validatePublicUrl(requestUrl);
        return route.continue();
      } catch {
        return route.abort("blockedbyclient");
      }
    });

    const page = await context.newPage();
    await page.goto(url.toString(), { waitUntil: "domcontentloaded", timeout: 10_000 });
    await page.waitForLoadState("load", { timeout: 2_500 }).catch(() => {});
    await page.waitForTimeout(400);
    return await page.content();
  } finally {
    await context.close();
  }
}

app.get("/health", (_req, res) => {
  res.json({ status: "ok", browser_enabled: browserEnabled });
});

app.post("/extract", async (req, res) => {
  try {
    const url = await validatePublicUrl(req.body?.url);
    const result = extractReadable(req.body?.html, url.toString());
    res.json(result);
  } catch (error) {
    res.status(422).json({ error: String(error?.message || error) });
  }
});

app.post("/render-extract", async (req, res) => {
  try {
    const url = await validatePublicUrl(req.body?.url);
    const html = await renderHtml(url.toString());
    const result = extractReadable(html, url.toString());
    res.json(result);
  } catch (error) {
    res.status(422).json({ error: String(error?.message || error) });
  }
});

const server = app.listen(port, "0.0.0.0", () => {
  console.log(`SupraCrawl extractor worker listening on :${port}`);
});

async function shutdown() {
  server.close();
  if (browserPromise) {
    const browser = await browserPromise.catch(() => null);
    await browser?.close();
  }
  process.exit(0);
}

process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
