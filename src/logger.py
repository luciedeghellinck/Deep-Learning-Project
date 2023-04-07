import logging
def setup_logging():
    logging.basicConfig(
        filename="../cfcv_log.txt",
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    logging.info("Starting new logger!")
