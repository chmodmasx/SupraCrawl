from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUPRACRAWL_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "SupraCrawl"
    user_agent: str = "SupraCrawl/0.1 (+https://github.com/chmodmasx/SupraCrawl)"
    request_timeout_s: float = Field(default=10.0, gt=0)
    max_redirects: int = Field(default=5, ge=0, le=20)
    max_html_bytes: int = Field(default=5_000_000, ge=16_384)
    min_useful_chars: int = Field(default=500, ge=1)
    max_context_tokens: int = Field(default=3500, ge=128)
    max_chunks_per_document: int = Field(default=3, ge=1, le=20)

    extractor_worker_url: str = "http://extractor-worker:3000"
    extractor_worker_timeout_s: float = Field(default=12.0, gt=0)
    browser_enabled: bool = True
    obey_robots_txt: bool = True

    redis_url: str | None = "redis://redis:6379/0"
    cache_ttl_s: int = Field(default=21_600, ge=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
