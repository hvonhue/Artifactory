import logging

from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level="ERROR",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        ),
    ],
)

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
