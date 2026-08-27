from .provider import SupraCrawlWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(SupraCrawlWebSearchProvider())
