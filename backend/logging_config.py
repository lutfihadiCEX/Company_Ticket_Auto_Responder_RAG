import logging
import os

LOG_FOLDER = "logs"
os.makedirs(LOG_FOLDER, exist_ok=True)

LOG_FILE = os.path.join(LOG_FOLDER, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

