#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/27 22:27
# @Author  : zhangpeng /zpskt
# @File    : log.py
# @Software: PyCharm
import logging


def init_loging_config():
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s (%(filename)s:%(lineno)d) - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    _logger = logging.getLogger("MediaCrawler")
    _logger.setLevel(level)
    return _logger

logger = init_loging_config()