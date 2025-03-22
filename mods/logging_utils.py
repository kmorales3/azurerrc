import logging
import os

def initialize_logger():
    """Initializes and returns a logger instance."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if reinitializing
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if not os.getenv("LOCAL_RUN") == "True":
            try:
                from opencensus.ext.azure.log_exporter import AzureLogHandler

                conn_str = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
                if conn_str:
                    logger.addHandler(AzureLogHandler(connection_string=conn_str))
            except Exception as e:
                logger.warning(f"⚠️ Could not attach AzureLogHandler: {e}")

    return logger