"""
Pydantic models for authentication.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class UserSignup(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)
    age: Optional[int] = Field(None, ge=1, le=150)
    phone: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: str = "user"
    age: Optional[int] = None
    phone: Optional[str] = None
    avatar: Optional[str] = None
    created_at: datetime
    total_scans: int = 0


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class UserUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = Field(None, ge=1, le=150)
    phone: Optional[str] = None
    avatar: Optional[str] = None
