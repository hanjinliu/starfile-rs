class ValidationError(Exception):
    """Schema validation failed."""


class BlockValidationError(ValidationError):
    """Block validation failed."""


class ColumnValidationError(BlockValidationError):
    """Column validation failed."""
