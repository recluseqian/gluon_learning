#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
log_utils
"""
import sys
import logging


def get_logger(name, consol_handler=True, consol_level=logging.DEBUG,
              file_name=None, file_level=logging.DEBUG):

    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(levelname)s]-[%(asctime)s]-"
                                  "[%(filename)s:%(lineno)s]\t%(message)s")
    if consol_handler:
        ch = logging.StreamHandler(stream=sys.stderr)
        ch.setLevel(consol_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


if __name__ == "__main__":
    logger = get_logger(__name__, file_name="log_utils.log")
    logger.debug("debug")
