import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
USER_DB_PATH = os.path.join(DATA_DIR, "users.json")
ACTIVITY_LOG_PATH = os.path.join(LOG_DIR, "user_activity.txt")


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def _load_users() -> Dict[str, Dict]:
    if not os.path.exists(USER_DB_PATH):
        return {}
    try:
        with open(USER_DB_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _persist_users(users: Dict[str, Dict]) -> None:
    with open(USER_DB_PATH, "w", encoding="utf-8") as fh:
        json.dump(users, fh, indent=2)


def register_user(full_name: str, phone: str, username: str, password: str) -> Tuple[bool, str]:
    """Register a brand-new user.

    Returns:
        (True, "") on success
        (False, "username_exists") if username already taken
        (False, "phone_exists") if phone already used by another account
    """
    users = _load_users()

    # Enforce unique username
    if username in users:
        return False, "username_exists"

    # Enforce unique phone number across users
    phone_clean = phone.strip()
    for record in users.values():
        if record.get("phone") == phone_clean:
            return False, "phone_exists"

    users[username] = {
        "full_name": full_name.strip(),
        "phone": phone_clean,
        "password_hash": generate_password_hash(password),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    _persist_users(users)
    log_user_action(username, "register", "/register", "New account created")
    return True, ""


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Validate credentials and return the user record when valid."""
    users = _load_users()
    record = users.get(username)
    if not record:
        return None
    if not check_password_hash(record["password_hash"], password):
        return None
    return record


def authenticate_user_by_phone(phone: str, password: str) -> Optional[Tuple[str, Dict]]:
    """Validate credentials using phone (+91XXXXXXXXXX). Returns (username, record) when valid."""
    users = _load_users()
    for uname, record in users.items():
        if record.get("phone") == phone and check_password_hash(record["password_hash"], password):
            return uname, record
    return None


def get_user_by_phone(phone: str) -> Optional[Tuple[str, Dict]]:
    """Return (username, record) for the given phone, if present."""
    users = _load_users()
    for uname, record in users.items():
        if record.get("phone") == phone:
            return uname, record
    return None


def get_user(username: str) -> Optional[Dict]:
    return _load_users().get(username)


def log_user_action(username: str, action: str, path: str, details: str = "") -> None:
    """Append a human-friendly log entry for auditing purposes."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} UTC | {username or 'anonymous'} | {action} | {path}"
    if details:
        entry += f" | {details}"
    entry += "\n"
    with open(ACTIVITY_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(entry)

