"""
Admin routes — view all users, scans, system stats.
"""

from fastapi import APIRouter, Depends, Query
from bson import ObjectId

from app.database import get_database
from app.auth.jwt_handler import get_admin_user
from app.predictions.routes import prediction_doc_to_response
from app.auth.routes import user_doc_to_response

router = APIRouter(prefix="/api/admin", tags=["Admin"])


@router.get("/users")
async def get_all_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    admin: dict = Depends(get_admin_user),
):
    """Get all registered users (admin only)."""
    db = get_database()
    total = await db.users.count_documents({})
    skip = (page - 1) * per_page

    cursor = db.users.find({}).sort("created_at", -1).skip(skip).limit(per_page)
    users = []
    async for user in cursor:
        users.append(user_doc_to_response(user).model_dump())

    return {"users": users, "total": total, "page": page, "per_page": per_page}


@router.get("/scans")
async def get_all_scans(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    admin: dict = Depends(get_admin_user),
):
    """Get all prediction scans (admin only)."""
    db = get_database()
    total = await db.predictions.count_documents({})
    skip = (page - 1) * per_page

    cursor = db.predictions.find({}).sort("created_at", -1).skip(skip).limit(per_page)
    scans = []
    async for doc in cursor:
        scan = prediction_doc_to_response(doc).model_dump()
        # Get user name
        user = await db.users.find_one({"_id": ObjectId(doc["user_id"])})
        scan["user_name"] = user["name"] if user else "Unknown"
        scan["user_email"] = user["email"] if user else "Unknown"
        scans.append(scan)

    return {"scans": scans, "total": total, "page": page, "per_page": per_page}


@router.get("/stats")
async def get_system_stats(admin: dict = Depends(get_admin_user)):
    """Get system-wide statistics (admin only)."""
    db = get_database()

    total_users = await db.users.count_documents({})
    total_scans = await db.predictions.count_documents({})
    high_risk_scans = await db.predictions.count_documents({"risk_level": "HIGH"})

    # Class distribution
    pipeline = [
        {"$group": {"_id": "$predicted_class", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    class_dist = {}
    async for doc in db.predictions.aggregate(pipeline):
        class_dist[doc["_id"]] = doc["count"]

    # Daily scans (last 30 days)
    daily_pipeline = [
        {
            "$group": {
                "_id": {
                    "year": {"$year": "$created_at"},
                    "month": {"$month": "$created_at"},
                    "day": {"$dayOfMonth": "$created_at"},
                },
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id.year": -1, "_id.month": -1, "_id.day": -1}},
        {"$limit": 30},
    ]
    daily_scans = []
    async for doc in db.predictions.aggregate(daily_pipeline):
        daily_scans.append({
            "date": f"{doc['_id']['year']}-{doc['_id']['month']:02d}-{doc['_id']['day']:02d}",
            "count": doc["count"],
        })

    return {
        "total_users": total_users,
        "total_scans": total_scans,
        "high_risk_scans": high_risk_scans,
        "low_risk_scans": total_scans - high_risk_scans,
        "class_distribution": class_dist,
        "daily_scans": daily_scans,
    }
