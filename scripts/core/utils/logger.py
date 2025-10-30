"""
Logging utilities with rich console output
"""

import sys
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from typing import Optional

console = Console()


def setup_logger(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "30 days"
):
    """
    Setup loguru logger with file and console output

    Args:
        log_file: Path to log file (if None, only console output)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        rotation: When to rotate log file
        retention: How long to keep old logs
    """
    # Remove default logger
    logger.remove()

    # Console output with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )

    # File output if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8"
        )

    return logger


def get_logger(name: str):
    """Get a logger with the specified name"""
    return logger.bind(name=name)


def create_progress_bar():
    """Create a rich progress bar for pipeline operations"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    )


def print_section(title: str):
    """Print a section header"""
    console.rule(f"[bold blue]{title}[/bold blue]")


def print_success(message: str):
    """Print success message"""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str):
    """Print error message"""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str):
    """Print warning message"""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str):
    """Print info message"""
    console.print(f"[blue]ℹ[/blue] {message}")


if __name__ == "__main__":
    # Test logger
    from pathlib import Path

    setup_logger(
        log_file=Path("outputs/logs/test.log"),
        level="DEBUG"
    )

    log = get_logger("test")

    print_section("Testing Logger")
    log.debug("This is a debug message")
    log.info("This is an info message")
    log.warning("This is a warning message")
    log.error("This is an error message")

    print_success("Success message")
    print_error("Error message")
    print_warning("Warning message")
    print_info("Info message")

    with create_progress_bar() as progress:
        task = progress.add_task("Processing...", total=100)
        for i in range(100):
            progress.update(task, advance=1)
            import time
            time.sleep(0.01)

    print_section("Test Complete")
