from loguru import logger

LOG_PATH = './logs/app.log'
logger.add(LOG_PATH, rotation='10 MB')

# class Logger:
#     def __new__(cls):
#         if cls._instance is None:
#             logger
#             cls._instance = super(Logger, cls).__new__(cls)
#         return cls._instance
#
#     def getLogger(self):

