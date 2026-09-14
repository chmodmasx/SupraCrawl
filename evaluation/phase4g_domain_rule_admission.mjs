import crypto from "node:crypto";
import fs from "node:fs";
import { createRequire } from "node:module";
import { isIP } from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const evaluatorPath = fileURLToPath(import.meta.url);
const here = path.dirname(evaluatorPath);
const requireFromWorker = createRequire(new URL("../services/extractor-worker/package.json", import.meta.url));
const { Readability } = requireFromWorker("@mozilla/readability");
const { JSDOM } = requireFromWorker("jsdom");
const TurndownModule = requireFromWorker("turndown");
const TurndownService = TurndownModule.default || TurndownModule;

const policySha = "77b3996d9c4ab89a6649abec57cb26ba6bcdbc11";
const fixturesPath = path.join(here, "phase4g_domain_rule_admission_fixtures.json");
const fixturesBytes = fs.readFileSync(fixturesPath);
const fixturesDoc = JSON.parse(fixturesBytes.toString("utf8"));

function cleanMarkdown(markdown) {
  return markdown
    .replace(/\u00a0/g, " ")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{4,}/g, "\n\n\n")
    .trim();
}

function sha256(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonicalize(value[key])]),
    );
  }
  return value;
}

function canonicalJson(value) {
  return JSON.stringify(canonicalize(value));
}

function sameOutput(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function validateAndNormalizeHost(value) {
  const host = String(value ?? "").replace(/\.+$/, "").toLowerCase();
  if (!host || host.length > 253) return { valid: false, host };
  if (isIP(host)) return { valid: false, host };
  const labelPattern = /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/;
  if (host.split(".").some((label) => !labelPattern.test(label))) {
    return { valid: false, host };
  }
  return { valid: true, host };
}

function normalizeHost(host) {
  return validateAndNormalizeHost(host).host;
}

function hostFromUrl(url) {
  const parsed = new URL(url);
  return normalizeHost(parsed.hostname);
}

function hostMatches(ruleHost, url) {
  const validated = validateAndNormalizeHost(ruleHost);
  return validated.valid && validated.host === hostFromUrl(url);
}

function normalizeCandidateRule(rule) {
  const allowed = new Set(["remove_selectors", "force_browser"]);
  const unknown = Object.keys(rule || {}).filter((key) => !allowed.has(key));
  if (unknown.length) return { valid: false, reason: "unknown_rule_field", rule: null };
  const normalized = {};
  if (rule?.remove_selectors !== undefined) {
    if (!Array.isArray(rule.remove_selectors) || rule.remove_selectors.length > 32) {
      return { valid: false, reason: "invalid_remove_selectors_shape", rule: null };
    }
    const selectors = [];
    for (const value of rule.remove_selectors) {
      if (typeof value !== "string") return { valid: false, reason: "invalid_remove_selector_type", rule: null };
      const selector = value.trim();
      if (!selector || selector.length > 512) {
        return { valid: false, reason: "invalid_remove_selector_length", rule: null };
      }
      selectors.push(selector);
    }
    if (selectors.length) normalized.remove_selectors = selectors;
  }
  if (rule?.force_browser !== undefined) {
    if (typeof rule.force_browser !== "boolean") {
      return { valid: false, reason: "invalid_force_browser_type", rule: null };
    }
    if (rule.force_browser) normalized.force_browser = true;
  }
  if (!normalized.remove_selectors && !normalized.force_browser) {
    return { valid: false, reason: "candidate_rule_has_no_enabled_primitive", rule: null };
  }
  return { valid: true, reason: null, rule: normalized };
}

function extractReadable(html, url, removeSelectors = []) {
  const dom = new JSDOM(html, { url });
  const document = dom.window.document;
  if (removeSelectors.length) {
    let removals;
    try {
      removals = removeSelectors.flatMap((selector) => [...document.querySelectorAll(selector)]);
    } catch {
      return { accepted: false, output: null };
    }
    for (const element of new Set(removals)) element.remove();
  }
  for (const selector of ["script", "style", "noscript", "template", "svg", "canvas", "form", "iframe"]) {
    for (const element of document.querySelectorAll(selector)) element.remove();
  }
  const article = new Readability(document, { keepClasses: false }).parse();
  if (!article?.content) {
    return {
      accepted: true,
      output: { title: document.title || "", markdown: "", text: "" },
    };
  }
  const turndown = new TurndownService({
    headingStyle: "atx",
    bulletListMarker: "-",
    codeBlockStyle: "fenced",
    emDelimiter: "_",
  });
  return {
    accepted: true,
    output: {
      title: String(article.title || document.title || "").trim(),
      markdown: cleanMarkdown(turndown.turndown(article.content)),
      text: String(article.textContent || "").trim(),
    },
  };
}

function baseline(input, htmlField = "static_html") {
  return extractReadable(input[htmlField], input.url).output;
}

function applyRuleDirect(input, rule, options = {}) {
  const browserEnabled = options.browserEnabled ?? true;
  const renderFailure = options.renderFailure ?? false;
  const selectors = Array.isArray(rule.remove_selectors) ? rule.remove_selectors : [];
  const baseStatic = baseline(input);
  const staticAttempt = extractReadable(input.static_html, input.url, selectors);
  const staticOutput = staticAttempt.accepted ? staticAttempt.output : baseStatic;

  if (!rule.force_browser || !browserEnabled || renderFailure || !input.rendered_html) {
    return {
      output: staticOutput,
      selector_valid: staticAttempt.accepted,
      rendered_attempted: false,
      rendered_selected: false,
    };
  }

  const baseRendered = baseline({ ...input, url: input.url }, "rendered_html");
  const renderedAttempt = extractReadable(input.rendered_html, input.url, selectors);
  const renderedOutput = renderedAttempt.accepted ? renderedAttempt.output : baseRendered;
  return {
    output: renderedOutput,
    selector_valid: staticAttempt.accepted && renderedAttempt.accepted,
    rendered_attempted: true,
    rendered_selected: true,
  };
}

function applyConfiguredRule(input, bundle, options = {}) {
  if (!hostMatches(bundle.host, input.url)) {
    return {
      output: baseline(input),
      matched: false,
      selector_valid: true,
      rendered_attempted: false,
      rendered_selected: false,
    };
  }
  const result = applyRuleDirect(input, bundle.candidate_rule, options);
  return { ...result, matched: true };
}

function markerText(output) {
  return String(output?.markdown || "").replace(/\\_/g, "_");
}

function score(output, contract) {
  const text = markerText(output);
  const missingRequired = contract.required_markers.filter((marker) => !text.includes(marker));
  const presentForbidden = contract.forbidden_markers.filter((marker) => text.includes(marker));
  return {
    pass: missingRequired.length === 0 && presentForbidden.length === 0,
    missing_required: missingRequired,
    present_forbidden: presentForbidden,
  };
}

function enabledPrimitives(rule) {
  const result = [];
  if (Array.isArray(rule.remove_selectors) && rule.remove_selectors.length) result.push("remove_selectors");
  if (rule.force_browser === true) result.push("force_browser");
  return result;
}

function ruleForSubset(candidateRule, subset) {
  const result = {};
  if (subset.includes("remove_selectors")) result.remove_selectors = candidateRule.remove_selectors;
  if (subset.includes("force_browser")) result.force_browser = true;
  return result;
}

function primitiveSubsets(primitives) {
  const subsets = [];
  for (let mask = 1; mask < (1 << primitives.length); mask += 1) {
    const subset = primitives.filter((_, index) => mask & (1 << index));
    subsets.push(subset);
  }
  return subsets.sort((a, b) => a.length - b.length || JSON.stringify(a).localeCompare(JSON.stringify(b)));
}

function evaluateBundle(bundle, cleanControl) {
  const hostValidation = validateAndNormalizeHost(bundle.host);
  const normalizedHost = hostValidation.host;
  const targetHost = hostFromUrl(bundle.url);
  const ruleValidation = normalizeCandidateRule(bundle.candidate_rule);
  const candidateRule = ruleValidation.rule || {};
  const primitives = enabledPrimitives(candidateRule);
  const base = baseline(bundle);
  const baseScore = score(base, bundle);
  const candidateResult = applyRuleDirect(bundle, candidateRule);
  const candidateScore = score(candidateResult.output, bundle);
  const repeatedCandidate = applyRuleDirect(bundle, candidateRule);
  const deterministic = sameOutput(candidateResult.output, repeatedCandidate.output);

  const passingSubsets = [];
  for (const subset of primitiveSubsets(primitives)) {
    const rule = ruleForSubset(candidateRule, subset);
    const result = applyRuleDirect(bundle, rule);
    if (score(result.output, bundle).pass) {
      passingSubsets.push({ subset, rule, output: result.output });
    }
  }
  const selected = passingSubsets[0] || null;

  const nonmatchingControl = applyConfiguredRule(cleanControl, bundle);
  const cleanControlBaseline = baseline(cleanControl);
  const controlEquivalent = !nonmatchingControl.matched && sameOutput(cleanControlBaseline, nonmatchingControl.output);

  const reasons = [];
  if (!hostValidation.valid) reasons.push("invalid_exact_dns_host");
  if (normalizedHost !== targetHost) reasons.push("bundle_host_does_not_match_target_url");
  if (!ruleValidation.valid) reasons.push(ruleValidation.reason);
  if (baseScore.pass) reasons.push("no_demonstrated_need");
  if (!candidateScore.pass) reasons.push("candidate_did_not_pass");
  if (!deterministic) reasons.push("candidate_not_deterministic");
  if (!controlEquivalent) reasons.push("clean_control_regressed");
  if (candidateScore.pass && !selected) reasons.push("no_passing_primitive_subset");

  if (selected?.subset.includes("remove_selectors")) {
    const withoutRemove = selected.subset.filter((primitive) => primitive !== "remove_selectors");
    const comparison = withoutRemove.length
      ? applyRuleDirect(bundle, ruleForSubset(candidateRule, withoutRemove)).output
      : base;
    if (sameOutput(selected.output, comparison)) reasons.push("remove_selectors_not_effective");
  }
  if (selected?.subset.includes("force_browser")) {
    if (!bundle.rendered_html) reasons.push("force_browser_missing_rendered_fixture");
    if (baseScore.pass) reasons.push("force_browser_static_baseline_already_passes");
  }

  const decision = reasons.length === 0 ? "ADMIT" : "REJECT";
  return {
    bundle_id: bundle.bundle_id,
    class: bundle.class,
    host: normalizedHost,
    bundle_hash_sha256: sha256(canonicalJson(bundle)),
    input_hashes: {
      static_html_sha256: sha256(bundle.static_html),
      rendered_html_sha256: bundle.rendered_html ? sha256(bundle.rendered_html) : null,
    },
    baseline: { output: base, score: baseScore },
    candidate: {
      rule: candidateRule,
      output: candidateResult.output,
      score: candidateScore,
      selector_valid: candidateResult.selector_valid,
      rendered_attempted: candidateResult.rendered_attempted,
      rendered_selected: candidateResult.rendered_selected,
    },
    passing_subsets: passingSubsets.map(({ subset, rule }) => ({ primitives: subset, rule })),
    selected_rule: selected?.rule || null,
    controls: {
      deterministic,
      clean_nonmatching_control_byte_identical: controlEquivalent,
    },
    decision,
    reasons,
  };
}

const normalizedBundleHosts = fixturesDoc.bundles.map((bundle) => validateAndNormalizeHost(bundle.host));
const bundleHostsValid = normalizedBundleHosts.every((item) => item.valid);
const normalizedHostValues = normalizedBundleHosts.map((item) => item.host);
const bundleHostsUnique = new Set(normalizedHostValues).size === normalizedHostValues.length;

const cleanControl = fixturesDoc.controls.clean_nonmatching_control;
const cleanControlBaseline = baseline(cleanControl);
const cleanControlScore = score(cleanControlBaseline, cleanControl);
const bundleResults = Object.fromEntries(
  fixturesDoc.bundles.map((bundle) => [bundle.bundle_id, evaluateBundle(bundle, cleanControl)]),
);

const expectedDecisions = {};
for (const bundle of fixturesDoc.bundles) {
  const result = bundleResults[bundle.bundle_id];
  const selectedMatches = bundle.expected_selected_rule === undefined
    || canonicalJson(result.selected_rule) === canonicalJson(bundle.expected_selected_rule);
  const expectedReasonMatches = bundle.expected_reason === undefined || result.reasons.includes(bundle.expected_reason);
  expectedDecisions[bundle.bundle_id] = {
    decision_matches: result.decision === bundle.expected_decision,
    selected_rule_matches: selectedMatches,
    expected_reason_present: expectedReasonMatches,
  };
}

const isolationResults = fixturesDoc.isolation_cases.map((test) => {
  const bundle = fixturesDoc.bundles.find((item) => item.bundle_id === test.bundle_id);
  if (!bundle) throw new Error(`Isolation bundle missing: ${test.bundle_id}`);
  const actual = hostMatches(bundle.host, test.url);
  return {
    bundle_id: test.bundle_id,
    url: test.url,
    expected_match: test.expected_match,
    actual_match: actual,
    pass: actual === test.expected_match,
  };
});

const removeBundle = fixturesDoc.bundles.find((bundle) => bundle.bundle_id === "remove_selectors_admissible");
const forceBundle = fixturesDoc.bundles.find((bundle) => bundle.bundle_id === "force_browser_admissible");
if (!removeBundle || !forceBundle) throw new Error("Required Phase 4G self-test bundles missing");

const invalidSelectorOutput = applyRuleDirect(removeBundle, {
  remove_selectors: [fixturesDoc.fail_open_cases.invalid_remove_selector],
}).output;
const missingSelectorOutput = applyRuleDirect(removeBundle, {
  remove_selectors: [fixturesDoc.fail_open_cases.missing_remove_selector],
}).output;
const forceBrowserDisabledOutput = applyRuleDirect(forceBundle, { force_browser: true }, { browserEnabled: false }).output;
const renderFailureOutput = applyRuleDirect(forceBundle, { force_browser: true }, { renderFailure: true }).output;
const failOpenControls = {
  invalid_remove_selector_baseline_equivalent: sameOutput(baseline(removeBundle), invalidSelectorOutput),
  missing_remove_selector_baseline_equivalent: sameOutput(baseline(removeBundle), missingSelectorOutput),
  force_browser_disabled_static_baseline_equivalent: sameOutput(baseline(forceBundle), forceBrowserDisabledOutput),
  render_failure_static_baseline_equivalent: sameOutput(baseline(forceBundle), renderFailureOutput),
};

const canonicalIsolationBundle = { ...removeBundle, host: removeBundle.host };
const canonicalIsolation = applyConfiguredRule(cleanControl, canonicalIsolationBundle);
const canonicalIsolationPass = !canonicalIsolation.matched && sameOutput(cleanControlBaseline, canonicalIsolation.output);

const admittedSelfTestRules = fixturesDoc.bundles
  .filter((bundle) => bundleResults[bundle.bundle_id].decision === "ADMIT")
  .map((bundle) => ({
    bundle_id: bundle.bundle_id,
    host: normalizeHost(bundle.host),
    selected_rule: bundleResults[bundle.bundle_id].selected_rule,
  }));

const acceptance = {
  bundle_hosts_are_valid_and_unique: bundleHostsValid && bundleHostsUnique,
  clean_control_baseline_passes: cleanControlScore.pass,
  expected_self_test_decisions_match: Object.values(expectedDecisions).every((checks) => Object.values(checks).every(Boolean)),
  all_bundle_runs_deterministic: Object.values(bundleResults).every((result) => result.controls.deterministic),
  all_nonmatching_controls_byte_identical: Object.values(bundleResults).every(
    (result) => result.controls.clean_nonmatching_control_byte_identical,
  ),
  exact_host_isolation_passes: isolationResults.every((result) => result.pass),
  canonical_url_does_not_influence_matching: canonicalIsolationPass,
  fail_open_controls_pass: Object.values(failOpenControls).every(Boolean),
  combined_candidate_reduced_to_minimal_subset:
    canonicalJson(bundleResults.combined_rule_reduced_to_minimal_subset.selected_rule)
    === canonicalJson({ remove_selectors: [".phase4g-combined-noise"] }),
  phase4g_admitted_production_rule_set_is_empty:
    canonicalJson(fixturesDoc.expected_admitted_production_rules) === canonicalJson([]),
};
acceptance.pass = Object.values(acceptance).every(Boolean);

const report = {
  schema_version: 1,
  phase: "4G",
  decision: acceptance.pass ? "PASS_DOMAIN_RULE_ADMISSION_PROTOCOL" : "FAIL_DOMAIN_RULE_ADMISSION_PROTOCOL",
  base_certified_sha: "9ec6e9aae85dac6d68c7877d6385fa8ad4db64f5",
  policy_sha: policySha,
  fixture_file_sha256: sha256(fixturesBytes),
  evaluator_source_sha256: sha256(fs.readFileSync(evaluatorPath)),
  engine: {
    readability: requireFromWorker("@mozilla/readability/package.json").version,
    jsdom: requireFromWorker("jsdom/package.json").version,
    turndown: requireFromWorker("turndown/package.json").version,
  },
  self_test_only: true,
  bundle_host_validation: {
    all_valid: bundleHostsValid,
    unique_normalized_hosts: bundleHostsUnique,
    normalized_hosts: normalizedHostValues,
  },
  bundle_results: bundleResults,
  expected_decision_checks: expectedDecisions,
  isolation_results: isolationResults,
  fail_open_controls: failOpenControls,
  canonical_isolation_pass: canonicalIsolationPass,
  self_test_admitted_rules: admittedSelfTestRules,
  admitted_production_rules: fixturesDoc.expected_admitted_production_rules,
  acceptance,
};

console.log("PHASE4G_REPORT_BEGIN");
console.log(JSON.stringify(report, null, 2));
console.log("PHASE4G_REPORT_END");
if (!acceptance.pass) process.exitCode = 1;
