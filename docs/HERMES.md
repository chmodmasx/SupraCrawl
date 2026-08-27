# Hermes Agent integration

SupraCrawl implements both Hermes web capabilities:

- `web_search` over the local SupraCrawl/OpenSearch corpus;
- `web_extract` for clean, context-budgeted extraction from explicit URLs.

Hermes can point both capabilities at SupraCrawl, or keep a different discovery/search provider while using SupraCrawl only for extraction.

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

Optional timeout override:

```bash
export SUPRACRAWL_TIMEOUT_S=20
```

## Use SupraCrawl for both search and extraction

```yaml
web:
  search_backend: supracrawl
  extract_backend: supracrawl
```

`web_search` queries only content already indexed by SupraCrawl. Populate the corpus through `/v1/index` or `/v1/crawl` before expecting search results.

## Use a different discovery provider

A mixed configuration remains valid:

```yaml
web:
  search_backend: ddgs
  extract_backend: supracrawl
```

This is useful while building the local corpus or when broad whole-web discovery is still required.

## Contract details

The provider is tested against a pinned real Hermes Agent `WebSearchProvider` ABC.

Search returns Hermes' fixed search envelope:

```text
{
  "success": true,
  "data": {
    "web": [
      {
        "title": "...",
        "url": "https://...",
        "description": "relevant indexed passage",
        "position": 1
      }
    ]
  }
}
```

Extraction returns the list of document dictionaries expected by the Hermes provider contract. `raw_content` is intentionally empty; `content` contains only selected passages that fit the configured context budget.
