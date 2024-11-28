import bcrypt
import logging


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed_password: str, logger: logging.Logger) -> bool:
    """Verify a password against a hash."""
    try:
        encoded_password = password.encode('utf-8')
        encoded_hash = hashed_password.encode('utf-8')
        return bcrypt.checkpw(encoded_password, encoded_hash)
    except Exception as e:
        logger.error(f"Error verifying password: {str(e)}")
        return False
