import logging


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)s] %(message)s",
        force=True,
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    return logger
