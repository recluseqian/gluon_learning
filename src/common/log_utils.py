#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
log_utils
"""
import sys
import logging


def get_logger(name, with_debug_info=False,
               console_handler=True, console_level=logging.DEBUG, console_stream=sys.stdout,
               file_name=None, file_level=logging.DEBUG):

    _logger = logging.getLogger(name=name)
    _logger.setLevel(logging.DEBUG)

    if with_debug_info:
        formatter = logging.Formatter("[%(levelname)s]-[%(asctime)s]-"
                                      "[%(filename)s:%(lineno)s]\t%(message)s")
    else:
        formatter = logging.Formatter("%(message)s")

    if console_handler:
        ch = logging.StreamHandler(stream=console_stream)
        ch.setLevel(console_level)
        ch.setFormatter(formatter)
        _logger.addHandler(ch)

    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        _logger.addHandler(fh)

    return _logger


if __name__ == "__main__":
    logger = get_logger(__name__, file_name="log_utils.log")
    logger.debug("debug")
