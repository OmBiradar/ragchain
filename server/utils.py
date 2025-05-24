import os
import logging
import time
import json
import platform
from functools import wraps
from pathlib import Path
from typing import Dict, Any, Callable, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for required environment variables
required_env_vars = [
    'GOOGLE_API_KEY'
]

def check_environment() -> bool:
    """Check if all required environment variables are set"""
    missing_vars = []
    for var in required_env_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

def get_env_var(name: str, default: Any = None) -> Any:
    """Get environment variable with fallback to default"""
    return os.environ.get(name, default)

def get_system_info() -> Dict[str, str]:
    """Get system information"""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine()
    }

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def ensure_directory_exists(directory_path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def check_file_exists(file_path: str) -> bool:
    """Check if file exists"""
    return Path(file_path).exists()
