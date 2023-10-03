import time
import psycopg2
import psycopg2.extensions
from psycopg2.extras import LoggingConnection, LoggingCursor
import logging
#
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# MyLoggingCursor simply sets self.timestamp at start of each query
