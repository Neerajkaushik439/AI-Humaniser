"""Concrete corpus source adapters."""

from data_sources.sources.blog_authorship import BlogAuthorshipSource
from data_sources.sources.hc3 import HC3Source
from data_sources.sources.mage import MAGESource
from data_sources.sources.par3 import PAR3Source
from data_sources.sources.pg19 import PG19Source
from data_sources.sources.raid import RAIDSource

__all__ = [
    "BlogAuthorshipSource",
    "HC3Source",
    "MAGESource",
    "PAR3Source",
    "PG19Source",
    "RAIDSource",
]
