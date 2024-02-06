import logging
from logging_tree import printout

logging.basicConfig(
    level=logging.INFO,
    filename="py_log.log",
    filemode="w",
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)
logging.info("this is from logging")

logger.debug("A DEBUG Message")
logger.info("An INFO")
logger.warning("A WARNING")
logger.error("An ERROR")
logger.critical("A message of CRITICAL severity")
printout()
