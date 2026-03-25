"""
Authentication routes — signup, login, current user.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status, Depends
from passlib.context import CryptContext
from bson import ObjectId

from app.database import get_database
from app.auth.jwt_handler import create_access_token, get_current_user
from app.auth.models import UserSignup, UserLogin, UserResponse, TokenResponse, UserUpdate

router = APIRouter(prefix="/api/auth", tags=["Authentication"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def user_doc_to_response(user: dict) -> UserResponse:
    """Convert a MongoDB user doc to a UserResponse."""
    return UserResponse(
        id=str(user["_id"]),
        name=user["name"],
        email=user["email"],
        role=user.get("role", "user"),
        age=user.get("age"),
        phone=user.get("phone"),
        avatar=user.get("avatar"),
        created_at=user.get("created_at", datetime.now(timezone.utc)),
        total_scans=user.get("total_scans", 0),
    )


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserSignup):
    """Register a new user."""
    db = get_database()

    # Check if user already exists
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    # Create user document
    user_doc = {
        "name": user_data.name,
        "email": user_data.email,
        "password_hash": pwd_context.hash(user_data.password),
        "role": "user",
        "age": user_data.age,
        "phone": user_data.phone,
        "avatar": None,
        "total_scans": 0,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    result = await db.users.insert_one(user_doc)
    user_doc["_id"] = result.inserted_id

    # Generate token
    token = create_access_token(
        data={"sub": str(result.inserted_id), "email": user_data.email, "role": "user"}
    )

    return TokenResponse(
        access_token=token,
        user=user_doc_to_response(user_doc),
    )


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    """Authenticate user and return JWT token."""
    db = get_database()

    user = await db.users.find_one({"email": credentials.email})
    if not user or not pwd_context.verify(credentials.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token(
        data={"sub": str(user["_id"]), "email": user["email"], "role": user.get("role", "user")}
    )

    return TokenResponse(
        access_token=token,
        user=user_doc_to_response(user),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user's profile."""
    db = get_database()
    user = await db.users.find_one({"_id": ObjectId(current_user["user_id"])})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user_doc_to_response(user)


@router.put("/profile", response_model=UserResponse)
async def update_profile(
    update_data: UserUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Update current user's profile."""
    db = get_database()

    update_fields = {k: v for k, v in update_data.model_dump().items() if v is not None}
    update_fields["updated_at"] = datetime.now(timezone.utc)

    await db.users.update_one(
        {"_id": ObjectId(current_user["user_id"])},
        {"$set": update_fields},
    )

    user = await db.users.find_one({"_id": ObjectId(current_user["user_id"])})
    return user_doc_to_response(user)
