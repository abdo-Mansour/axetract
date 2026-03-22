"""Custom exceptions for the Axetract package."""


class AXEError(Exception):
    """Base class for all Axetract exceptions."""

    pass


class PreprocessingError(AXEError):
    """Raised when an error occurs during the preprocessing stage."""

    pass


class PruningError(AXEError):
    """Raised when an error occurs during the pruning stage."""

    pass


class ExtractionError(AXEError):
    """Raised when an error occurs during the extraction stage."""

    pass


class PostprocessingError(AXEError):
    """Raised when an error occurs during the postprocessing stage."""

    pass


class ModelLoadError(AXEError):
    """Raised when an error occurs while loading an LLM model."""

    pass


class ConfigurationError(AXEError):
    """Raised when there is an invalid configuration for the pipeline."""

    pass
