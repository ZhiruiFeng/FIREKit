"""DataStream — unified market data pipeline (offline MVP).

Public API:

- Schema: :class:`OHLCVBar`, :func:`normalize_frame`, :func:`bars_to_frame`
- Sources: :class:`BaseSource`, :class:`SyntheticSource`, :class:`FileSource`
- Storage: :class:`ParquetStore`
- Quality: :class:`QualityEngine`, :class:`IssueType`, report dataclasses
- Universe: :class:`PointInTimeUniverse`, :class:`Membership`
"""

from datastream.quality import (
    AggregateQualityReport,
    IssueType,
    QualityEngine,
    QualityIssue,
    SymbolQualityReport,
)
from datastream.schema import (
    CANONICAL_COLUMNS,
    OPTIONAL_COLUMNS,
    OHLCVBar,
    bars_to_frame,
    normalize_frame,
)
from datastream.sources import REGIMES, BaseSource, FileSource, SyntheticSource
from datastream.store import ParquetStore
from datastream.universe import Membership, PointInTimeUniverse

__version__ = "0.1.0"

__all__ = [
    "CANONICAL_COLUMNS",
    "OPTIONAL_COLUMNS",
    "REGIMES",
    "AggregateQualityReport",
    "BaseSource",
    "FileSource",
    "IssueType",
    "Membership",
    "OHLCVBar",
    "ParquetStore",
    "PointInTimeUniverse",
    "QualityEngine",
    "QualityIssue",
    "SymbolQualityReport",
    "SyntheticSource",
    "bars_to_frame",
    "normalize_frame",
    "__version__",
]
