import crypto from "node:crypto";
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const requireFromWorker = createRequire(new URL("../services/extractor-worker/package.json", import.meta.url));
const { Readability } = requireFromWorker("@mozilla/readability");
const { JSDOM } = requireFromWorker("jsdom");
const TurndownModule = requireFromWorker("turndown");
const TurndownService = TurndownModule.default || TurndownModule;

const fixturesPath = path.join(here, "phase4e_domain_rule_selection_fixtures.json");
const fixturesDoc = JSON.parse(fs.readFileSync(fixturesPath, "utf8"));

function cleanMarkdown(markdown) {
  return markdown
    .replace(/\u00a0/g, " ")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{4,}/g, "\n\n\n")
    .trim();
}

function extractReadable(html, url, mutateDocument = null) {
  const dom = new JSDOM(html, { url });
  const document = dom.window.document;
  if (mutateDocument) {
    const accepted = mutateDocument(document);
    if (accepted === false) return null;
  }
  for (const selector of ["script", "style", "noscript", "template", "svg", "canvas", "form", "iframe"]) {
    for (const element of document.querySelectorAll(selector)) element.remove();
  }
  const article = new Readability(document, { keepClasses: false }).parse();
  if (!article?.content) return { title: document.title || "", markdown: "", text: "" };
  const turndown = new TurndownService({
    headingStyle: "atx",
    bulletListMarker: "-",
    codeBlockStyle: "fenced",
    emDelimiter: "_",
  });
  return {
    title: String(article.title || document.title || "").trim(),
    markdown: cleanMarkdown(turndown.turndown(article.content)),
    text: String(article.textContent || "").trim(),
  };
}

function baseline(fixture) {
  return extractReadable(fixture.static_html, fixture.url);
}

function removeSelectors(fixture, selectors) {
  const base = baseline(fixture);
  const result = extractReadable(fixture.static_html, fixture.url, (document) => {
    try {
      for (const selector of selectors) {
        for (const element of document.querySelectorAll(selector)) element.remove();
      }
      return true;
    } catch {
      return false;
    }
  });
  return result === null ? base : result;
}

function contentSelector(fixture, selector) {
  const base = baseline(fixture);
  const result = extractReadable(fixture.static_html, fixture.url, (document) => {
    let matches;
    try {
      matches = document.querySelectorAll(selector);
    } catch {
      return false;
    }
    if (!matches.length) return false;
    const selected = matches[0].cloneNode(true);
    document.body.replaceChildren(selected);
    return true;
  });
  return result === null ? base : result;
}

function forceBrowser(fixture, enabled, browserEnabled) {
  if (!enabled || !browserEnabled) return baseline(fixture);
  return extractReadable(fixture.rendered_html, fixture.url);
}

function score(output, fixture) {
  const markdown = output?.markdown || "";
  const missingRequired = fixture.required_markers.filter((marker) => !markdown.includes(marker));
  const presentForbidden = fixture.forbidden_markers.filter((marker) => markdown.includes(marker));
  return {
    pass: missingRequired.length === 0 && presentForbidden.length === 0,
    missing_required: missingRequired,
    present_forbidden: presentForbidden,
  };
}

function sameOutput(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function sha256(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function subsets(values) {
  const result = [];
  for (let mask = 0; mask < (1 << values.length); mask += 1) {
    const subset = values.filter((_, index) => mask & (1 << index));
    result.push(subset);
  }
  return result;
}

const fixtureResults = {};
for (const fixture of fixturesDoc.fixtures) {
  const base = baseline(fixture);
  const candidates = {
    remove_selectors: removeSelectors(fixture, fixture.rules.remove_selectors),
    content_selector: contentSelector(fixture, fixture.rules.content_selector),
    force_browser: forceBrowser(fixture, fixture.rules.force_browser, true),
  };
  fixtureResults[fixture.id] = {
    class: fixture.class,
    fixture_hashes: {
      static_html_sha256: sha256(fixture.static_html),
      rendered_html_sha256: sha256(fixture.rendered_html),
    },
    baseline: { output: base, score: score(base, fixture) },
    candidates: Object.fromEntries(
      Object.entries(candidates).map(([name, output]) => [name, { output, score: score(output, fixture) }]),
    ),
  };
}

const control = fixturesDoc.fixtures.find((fixture) => fixture.class === "clean_control");
if (!control) throw new Error("clean_control fixture missing");
const controlBaseline = baseline(control);
const failOpen = {
  invalid_remove_selector: sameOutput(controlBaseline, removeSelectors(control, [fixturesDoc.fail_open_cases.invalid_remove_selector])),
  missing_remove_selector: sameOutput(controlBaseline, removeSelectors(control, [fixturesDoc.fail_open_cases.missing_remove_selector])),
  invalid_content_selector: sameOutput(controlBaseline, contentSelector(control, fixturesDoc.fail_open_cases.invalid_content_selector)),
  missing_content_selector: sameOutput(controlBaseline, contentSelector(control, fixturesDoc.fail_open_cases.missing_content_selector)),
  force_browser_disabled: sameOutput(controlBaseline, forceBrowser(control, true, false)),
};

const primitiveNames = ["remove_selectors", "content_selector", "force_browser"];
const targetFixtures = fixturesDoc.fixtures.filter((fixture) => fixture.class !== "clean_control");
const demonstratedNeed = targetFixtures
  .filter((fixture) => !fixtureResults[fixture.id].baseline.score.pass)
  .map((fixture) => fixture.class);
const coverage = Object.fromEntries(primitiveNames.map((name) => [name, []]));
for (const fixture of targetFixtures) {
  if (fixtureResults[fixture.id].baseline.score.pass) continue;
  for (const primitive of primitiveNames) {
    if (fixtureResults[fixture.id].candidates[primitive].score.pass) coverage[primitive].push(fixture.class);
  }
}

const eligible = primitiveNames.filter((primitive) => coverage[primitive].length > 0);
const needed = new Set(demonstratedNeed);
const coveringSets = subsets(eligible).filter((subset) => {
  const union = new Set(subset.flatMap((primitive) => coverage[primitive]));
  return [...needed].every((fixtureClass) => union.has(fixtureClass));
});
const minSize = coveringSets.length ? Math.min(...coveringSets.map((set) => set.length)) : null;
const minimalSets = coveringSets
  .filter((set) => set.length === minSize)
  .map((set) => [...set].sort())
  .sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
const selected = minimalSets[0] || [];

const controlCandidateEquivalence = Object.fromEntries(
  primitiveNames.map((primitive) => [
    primitive,
    sameOutput(controlBaseline, fixtureResults[control.id].candidates[primitive].output),
  ]),
);
const targetBaselinesAllFail = targetFixtures.every((fixture) => !fixtureResults[fixture.id].baseline.score.pass);
const selectedHaveGain = selected.every((primitive) => coverage[primitive].length > 0);
const acceptance = {
  clean_control_baseline_passes: fixtureResults[control.id].baseline.score.pass,
  clean_control_candidates_byte_identical: Object.values(controlCandidateEquivalence).every(Boolean),
  fail_open_cases_byte_identical: Object.values(failOpen).every(Boolean),
  every_target_fixture_demonstrates_baseline_need: targetBaselinesAllFail,
  every_selected_primitive_has_gain: selectedHaveGain,
  selected_vocabulary_covers_all_demonstrated_need: selected.length > 0 && minimalSets.length > 0,
};
acceptance.pass = Object.values(acceptance).every(Boolean);

const report = {
  schema_version: 1,
  phase: "4E",
  policy_sha: "4f688d6983a209b2cee6e5dd378450e6b53ab559",
  fixture_file_sha256: sha256(fs.readFileSync(fixturesPath)),
  engine: {
    readability: requireFromWorker("@mozilla/readability/package.json").version,
    jsdom: requireFromWorker("jsdom/package.json").version,
    turndown: requireFromWorker("turndown/package.json").version,
  },
  content_selector_multiple_match_semantics: fixturesDoc.content_selector_multiple_match_semantics,
  fixture_results: fixtureResults,
  fail_open: failOpen,
  control_candidate_equivalence: controlCandidateEquivalence,
  demonstrated_need: demonstratedNeed,
  primitive_coverage: coverage,
  eligible_primitives: eligible,
  minimal_covering_sets: minimalSets,
  selected_primitive_set: selected,
  rejected_primitive_reasons: Object.fromEntries(
    primitiveNames.filter((name) => !selected.includes(name)).map((name) => [
      name,
      coverage[name].length === 0 ? "no_preregistered_fixture_gain" : "redundant_under_minimum_primitive_cover",
    ]),
  ),
  acceptance,
};

const text = JSON.stringify(report, null, 2);
console.log("PHASE4E_REPORT_BEGIN");
console.log(text);
console.log("PHASE4E_REPORT_END");
if (!acceptance.pass) process.exitCode = 1;
