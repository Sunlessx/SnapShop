from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # everything here can be overridden from .env (same name, uppercase)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    typesense_host: str = ""
    typesense_port: int = 443
    typesense_protocol: str = "https"
    typesense_api_key: str = ""
    typesense_collection: str = "products"

    # list of datasets to index (shein, temu, ...). if the file doesn't exist
    # we just fall back to the typesense catalog like before
    sources_file: str = "sources.json"

    index_path: str = "data/embeddings.index"
    metadata_path: str = "data/metadata.pkl"

    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_device: str = "cpu"

    search_top_k: int = 20
    max_upload_bytes: int = 10_485_760  # 10 MB

    # live web search (google shopping through serpapi), optional
    serpapi_key: str = ""
    web_country: str = "us"

    redis_url: str = "redis://localhost:6379"
    redis_key_prefix: str = "vs:"


settings = Settings()
