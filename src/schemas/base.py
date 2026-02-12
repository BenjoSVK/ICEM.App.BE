"""
Domain models for auth: token payload and user record.
"""


class TokenData:
    """Payload extracted from a valid JWT (e.g. username)."""

    def __init__(self, username: str) -> None:
        self.username = username


class User:
    """User record as returned from the database (username and hashed password)."""

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
