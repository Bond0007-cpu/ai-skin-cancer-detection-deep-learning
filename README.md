# 🏥 DermAI — AI Skin Cancer Detection System

A production-grade full-stack web application for AI-powered skin cancer detection using deep learning. Upload dermoscopic images and receive instant classification across 7 skin lesion types with confidence scores, disease information, precautions, and downloadable PDF reports.

![Tech Stack](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)
![MongoDB](https://img.shields.io/badge/MongoDB-7.0-47A248?logo=mongodb)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-FF6F00?logo=tensorflow)
![Tailwind](https://img.shields.io/badge/Tailwind-3.3-06B6D4?logo=tailwindcss)

---

## ✨ Features

### Core
- 🔐 **JWT Authentication** — Secure login/signup with bcrypt password hashing
- 🧠 **AI Prediction** — EfficientNetB4 model for 7-class skin lesion classification
- 📊 **Interactive Dashboard** — Summary cards, recent scans, quick actions
- 📁 **Scan History** — Browse, filter, and manage all past predictions
- 📈 **Analytics** — Pie charts, bar charts, and trend lines with Recharts
- 👤 **User Profile** — Personal details management
- 🛡️ **Admin Panel** — View all users, scans, and system statistics
- 📄 **PDF Reports** — Download detailed clinical reports

### UI/UX
- 🌗 **Dark/Light Mode** — Toggle with system preference detection
- 🎨 **Glass Morphism UI** — Modern, premium design
- 📱 **Fully Responsive** — Mobile, tablet, desktop
- ✨ **Smooth Animations** — Framer Motion transitions, loading states
- 🖼️ **Drag & Drop Upload** — Image preview before analysis

---

## 🗂️ Project Structure

```
ai_skin_cancer_detection/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry
│   │   ├── config.py            # Environment configuration
│   │   ├── database.py          # MongoDB connection
│   │   ├── auth/
│   │   │   ├── routes.py        # Auth endpoints (signup/login/profile)
│   │   │   ├── jwt_handler.py   # JWT token utilities
│   │   │   └── models.py        # Auth Pydantic models
│   │   ├── predictions/
│   │   │   ├── routes.py        # Prediction endpoints
│   │   │   └── models.py        # Prediction Pydantic models
│   │   ├── admin/
│   │   │   └── routes.py        # Admin endpoints
│   │   └── ml/
│   │       └── model_service.py # ML model loading & inference
│   ├── uploads/                 # Uploaded images storage
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Root with routing
│   │   ├── main.jsx             # React entry
│   │   ├── index.css            # Global styles + Tailwind
│   │   ├── components/
│   │   │   └── DashboardLayout.jsx  # Sidebar + header layout
│   │   ├── context/
│   │   │   ├── AuthContext.jsx  # Authentication state
│   │   │   └── ThemeContext.jsx # Dark/light mode state
│   │   ├── pages/
│   │   │   ├── LoginPage.jsx    # Login form
│   │   │   ├── SignupPage.jsx   # Registration form
│   │   │   ├── DashboardHome.jsx # Dashboard overview
│   │   │   ├── UploadScan.jsx   # Image upload & results
│   │   │   ├── History.jsx      # Scan history list
│   │   │   ├── Analytics.jsx    # Charts & insights
│   │   │   ├── Profile.jsx      # User profile editor
│   │   │   ├── Settings.jsx     # App settings
│   │   │   └── AdminPanel.jsx   # Admin management
│   │   └── services/
│   │       └── api.js           # Axios API client
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── index.html
│
├── models/                      # Trained model files
├── src/                         # Original ML pipeline
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** and npm
- **MongoDB** (local or Atlas cloud)

### 1. Clone & Setup

```bash
cd "ai_skin_cancer_detection"
```

### 2. Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Create .env from template
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# Edit .env with your MongoDB URL and JWT secret key
```

### 3. Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Create .env
copy .env.example .env
```

### 4. Start MongoDB

```bash
# If using local MongoDB
mongod

# Or use MongoDB Atlas (update MONGODB_URL in backend/.env)
```

### 5. Run the Application

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

🌐 Open **http://localhost:5173** in your browser.

---

## 📡 API Documentation

### Authentication

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/api/auth/signup` | Register new user | ❌ |
| `POST` | `/api/auth/login` | Login & get JWT token | ❌ |
| `GET` | `/api/auth/me` | Get current user profile | ✅ |
| `PUT` | `/api/auth/profile` | Update user profile | ✅ |

### Predictions

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/api/predictions/analyze` | Upload image & get prediction | ✅ |
| `GET` | `/api/predictions/history` | Get prediction history (paginated) | ✅ |
| `GET` | `/api/predictions/stats` | Get user statistics | ✅ |
| `GET` | `/api/predictions/{id}` | Get single prediction | ✅ |
| `DELETE` | `/api/predictions/{id}` | Delete a prediction | ✅ |
| `GET` | `/api/predictions/info/diseases` | Get all disease info | ✅ |

### Admin (role: admin)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET` | `/api/admin/users` | List all users | 🛡️ Admin |
| `GET` | `/api/admin/scans` | List all scans | 🛡️ Admin |
| `GET` | `/api/admin/stats` | System-wide statistics | 🛡️ Admin |

### Interactive Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🗃️ Database Schema (MongoDB)

### Users Collection
```json
{
  "_id": "ObjectId",
  "name": "string",
  "email": "string (unique)",
  "password_hash": "string (bcrypt)",
  "role": "user | admin",
  "age": "number | null",
  "phone": "string | null",
  "avatar": "string | null",
  "total_scans": "number",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### Predictions Collection
```json
{
  "_id": "ObjectId",
  "user_id": "string (ref: users._id)",
  "image_filename": "string",
  "image_base64": "string",
  "predicted_class": "string (nv|mel|bkl|bcc|akiec|vasc|df)",
  "confidence": "float (0-1)",
  "class_probabilities": "object {class: probability}",
  "disease_name": "string",
  "description": "string",
  "risk_level": "HIGH | LOW",
  "precautions": ["string"],
  "recommendation": "string",
  "created_at": "datetime"
}
```

---

## 🧬 Supported Skin Lesion Classes

| Code | Disease | Risk Level |
|------|---------|-----------|
| `nv` | Melanocytic Nevi (Benign Mole) | 🟢 LOW |
| `mel` | Melanoma (Malignant) | 🔴 HIGH |
| `bkl` | Benign Keratosis-like Lesions | 🟢 LOW |
| `bcc` | Basal Cell Carcinoma | 🔴 HIGH |
| `akiec` | Actinic Keratoses | 🔴 HIGH |
| `vasc` | Vascular Lesions | 🟢 LOW |
| `df` | Dermatofibroma | 🟢 LOW |

---

## 🔧 Environment Variables

### Backend (`backend/.env`)
```env
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=skin_cancer_db
JWT_SECRET_KEY=your-super-secret-key-change-this
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
MODEL_PATH=../models/exported/efficientnetb4_savedmodel
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
UPLOAD_DIR=./uploads
MAX_IMAGE_SIZE_MB=10
```

### Frontend (`frontend/.env`)
```env
VITE_API_URL=http://localhost:8000
```

---

## 🛡️ Admin Access

To grant yourself admin access, run in MongoDB shell:

```javascript
db.users.updateOne(
  { email: "your@email.com" },
  { $set: { role: "admin" } }
)
```

---

## 🎯 Demo Mode

If no trained model is available at `MODEL_PATH`, the backend runs in **demo mode** with simulated predictions — perfect for UI development and testing.

---

## 📄 License

This project is for educational purposes. AI predictions should NOT replace professional medical diagnosis.

---

**Built with ❤️ for Final Year Project**
