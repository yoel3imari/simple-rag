from __future__ import annotations
from enum import Enum


class Language(str, Enum):
    EN = "en"
    FR = "fr"
    AR = "ar"


class TransformMode(str, Enum):
    NONE = "none"
    REWRITE = "rewrite"
    HYDE = "hyde"
    HYDE_FUSION = "hyde_fusion"
