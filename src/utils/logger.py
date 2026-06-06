"""
Structured logging utility for the RAG platform.

Provides JSON and text-based logging with context inspection capability.
Follows structured logging best practices for auditability and debugging.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Any
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Include extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class StructuredLogger:
    """
    Structured logger for the RAG platform.
    
    Supports both JSON and text formatting. Provides methods for logging
    with contextual information and explicit error handling.
    """
    
    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: str = "INFO",
        format_type: str = "json",
        max_file_size: int = 10485760,  # 10 MB
        backup_count: int = 5,
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically __name__)
            log_file: Path to log file (None = console only)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_type: Output format ("json" or "text")
            max_file_size: Max size per log file in bytes
            backup_count: Number of backup log files to keep
        
        Raises:
            ValueError: If level or format_type is invalid
            IOError: If log file directory cannot be created
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.format_type = format_type
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Setup formatter
        if format_type == "json":
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if log_file specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **context: Any) -> None:
        """Log info-level message with context."""
        if context:
            self.logger.info(message, extra={"extra_fields": context})
        else:
            self.logger.info(message)
    
    def debug(self, message: str, **context: Any) -> None:
        """Log debug-level message with context."""
        if context:
            self.logger.debug(message, extra={"extra_fields": context})
        else:
            self.logger.debug(message)
    
    def warning(self, message: str, **context: Any) -> None:
        """Log warning-level message with context."""
        if context:
            self.logger.warning(message, extra={"extra_fields": context})
        else:
            self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False, **context: Any) -> None:
        """Log error-level message with context."""
        if context:
            self.logger.error(message, exc_info=exc_info, extra={"extra_fields": context})
        else:
            self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = False, **context: Any) -> None:
        """Log critical-level message with context."""
        if context:
            self.logger.critical(message, exc_info=exc_info, extra={"extra_fields": context})
        else:
            self.logger.critical(message, exc_info=exc_info)


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_type: str = "json",
) -> StructuredLogger:
    """
    Factory function to get or create a structured logger.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        format_type: Output format
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(
        name=name,
        log_file=log_file,
        level=level,
        format_type=format_type,
    )
