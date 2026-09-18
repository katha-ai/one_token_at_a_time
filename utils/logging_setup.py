import logging
import os

def setup_logging(save_dir, log_file_name="extraction.log"):
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler() # Also print to console
        ]
    )