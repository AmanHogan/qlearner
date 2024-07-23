class PackageError(Exception):
    """Base class for all exceptions raised by this package."""
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class ValidationError(PackageError):
    """Exception raised for validation errors."""
    def __init__(self, message):
        super().__init__(message)

class ProcessingError(PackageError):
    """Exception raised for errors during processing."""
    def __init__(self, message):
        super().__init__(message)

class ConfigurationError(PackageError):
    """Exception raised for configuration-related errors."""
    def __init__(self, message):
        super().__init__(message)