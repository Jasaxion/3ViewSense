import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(verbose: bool = False) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Unique log filename per script + timestamp
    if sys.argv and len(sys.argv) > 0:
        script_name = Path(sys.argv[0]).stem
    else:
        script_name = "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"{script_name}_{timestamp}.log"
    log_filepath = logs_dir / log_filename
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    root_logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    root_logger.info(f"Logging to {log_filepath}")

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

# Auto-configure logging on import
if not logging.root.handlers:
    setup_logging(verbose=False)
