import multiprocessing
import platform

import torch
import typer
from logger import logger

print()
logger.info("All imports successfully executed.")


def main() -> None:
    print()
    logger.info("GPU: %s", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("CPU: %s", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print()
    logger.info("Machine: %s", platform.machine())
    logger.info("Version: %s", platform.version())
    logger.info("Platform: %s", platform.platform())
    logger.info("Processor: %s", platform.processor())
    logger.info("CPU Count: %s", multiprocessing.cpu_count())
    logger.info("PyTorch Version: %s", torch.__version__)

    print("Job successfully executed.")


if __name__ == "__main__":
    typer.run(main)
