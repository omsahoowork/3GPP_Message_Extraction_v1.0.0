# chunking/__init__.py
"""Public API for the chunking package."""


def convert_setext_to_atx(*args, **kwargs):
    from chunking.chunking import convert_setext_to_atx as impl

    return impl(*args, **kwargs)


def split_markdown_files(*args, **kwargs):
    from chunking.chunking import split_markdown_files as impl

    return impl(*args, **kwargs)


def build_vectorstore(*args, **kwargs):
    from chunking.chunking import build_vectorstore as impl

    return impl(*args, **kwargs)


def split_508_markdown_files(*args, **kwargs):
    from chunking.chunking_508 import split_508_markdown_files as impl

    return impl(*args, **kwargs)


def build_508_vectorstore(*args, **kwargs):
    from chunking.chunking_508 import build_508_vectorstore as impl

    return impl(*args, **kwargs)

__all__ = [
    "convert_setext_to_atx",
    "split_markdown_files",
    "build_vectorstore",
    "split_508_markdown_files",
    "build_508_vectorstore",
]
