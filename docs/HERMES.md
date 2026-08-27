# Hermes Agent integration

Hermes supports independent providers for `web_search` and `web_extract`. SupraCrawl Phase 1 is deliberately extract-only, so it can be paired with any search provider while the local index is still being built.

## Install the plugin

Copy:

```text
integrations/hermes/supracrawl/
```

to:

```text
~/.hermes/plugins/web/supracrawl/
```

Then enable the plugin using Hermes' plugin tooling and set:

```bash
export SUPRACRAWL_URL=http://127.0.0.1:8080
```

Configure Hermes:

```yaml
web:
  search_backend: ddgs       # temporary discovery backend, choose your preferred provider
  extract_backend: supracrawl
```

Phase 2 will add `supports_search()` and `/v1/search`, allowing both capabilities to point at SupraCrawl.

The provider follows Hermes' fixed extraction envelope and intentionally leaves `raw_content` empty. The `content` field contains only the selected passages that fit the configured context budget.
