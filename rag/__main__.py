"""Entrypoint: `python -m rag_app`."""
import asyncio
import logging

from dotenv import load_dotenv

from .indexing import sync_documents
from .rag import ask


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    sync_documents()
    asyncio.run(ask("What are the different set operations?"))


if __name__ == "__main__":
    main()