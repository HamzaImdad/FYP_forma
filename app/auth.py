"""
FORMA auth primitives — bcrypt password hashing, JWT cookie issuance,
and Flask/Socket.IO decorators.

Keys:
  - FORMA_JWT_SECRET (env) — HS256 signing secret, loaded from .env
  - forma_auth cookie — httpOnly, 30-day JWT with {uid, iat, exp}
"""

import datetime as dt
import functools
import os
from typing import Optional

import bcrypt
import jwt
from flask import g, jsonify, request

JWT_ALGO = "HS256"
JWT_COOKIE = "forma_auth"
JWT_TTL_DAYS = 30


def _secret() -> str:
    s = os.environ.get("FORMA_JWT_SECRET")
    if not s:
        raise RuntimeError(
            "FORMA_JWT_SECRET is not set. Load .env via python-dotenv in "
            "app/server.py before importing app.auth."
        )
    return s


# ── Passwords ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    # bcrypt caps input at 72 bytes — truncate defensively so long
    # passphrases don't silently lose their tail.
    pwd = (plain or "").encode("utf-8")[:72]
    return bcrypt.hashpw(pwd, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    if not plain or not hashed:
        return False
    pwd = plain.encode("utf-8")[:72]
    try:
        return bcrypt.checkpw(pwd, hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ── JWT ────────────────────────────────────────────────────────────────

def create_jwt(user_id: int) -> str:
    now = dt.datetime.now(tz=dt.timezone.utc)
    payload = {
        "uid": int(user_id),
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(days=JWT_TTL_DAYS)).timestamp()),
    }
    return jwt.encode(payload, _secret(), algorithm=JWT_ALGO)


def decode_jwt(token: Optional[str]) -> Optional[int]:
    if not token:
        return None
    try:
        payload = jwt.decode(token, _secret(), algorithms=[JWT_ALGO])
        uid = payload.get("uid")
        return int(uid) if uid is not None else None
    except jwt.PyJWTError:
        return None


def set_auth_cookie(response, token: str) -> None:
    """Apply the FORMA auth cookie to a Flask Response."""
    response.set_cookie(
        JWT_COOKIE,
        token,
        max_age=JWT_TTL_DAYS * 86400,
        httponly=True,
        samesite="Lax",
        secure=False,  # localhost dev — flip to True behind HTTPS
        path="/",
    )


def clear_auth_cookie(response) -> None:
    response.delete_cookie(JWT_COOKIE, path="/")


# ── Decorators ─────────────────────────────────────────────────────────

def require_auth(fn):
    """Flask route decorator — enforce a valid JWT cookie, stash user_id on g."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        token = request.cookies.get(JWT_COOKIE)
        uid = decode_jwt(token)
        if uid is None:
            return jsonify({"error": "unauthorized"}), 401
        g.user_id = uid
        return fn(*args, **kwargs)

    return wrapper


def current_user_id() -> Optional[int]:
    """Read the current user_id from the cookie without enforcing auth."""
    return decode_jwt(request.cookies.get(JWT_COOKIE))
