import logging

def configure_logging():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)-15s %(filename)15s: %(message)s")

configure_logging()  # Call the function to configure logging when the module is imported
log = logging.getLogger()