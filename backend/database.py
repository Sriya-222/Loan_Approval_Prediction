import os
import json
import hashlib
import secrets
from datetime import datetime

# Configure MongoDB Connection
MONGODB_URI = os.environ.get("MONGODB_URI", "")
USE_MONGODB = False
db = None
users_col = None
logs_col = None

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Verify connection by running a command
        client.admin.command('ping')
        db = client.get_database("credit_risk_db")
        users_col = db["users"]
        logs_col = db["prediction_logs"]
        USE_MONGODB = True
        print("Successfully connected to MongoDB Atlas!")
    except Exception as e:
        print(f"Warning: MongoDB Atlas connection failed ({e}). Falling back to local file-based database.")
        USE_MONGODB = False

# Fallback Local JSON Database Setup
LOCAL_DB_PATH = os.path.join(os.path.dirname(__file__), "local_db.json")

def load_local_db():
    if os.path.exists(LOCAL_DB_PATH):
        try:
            with open(LOCAL_DB_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"users": [], "prediction_logs": []}

def save_local_db(data):
    try:
        with open(LOCAL_DB_PATH, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error writing to local database file: {e}")

# Secure Password Hashing Utilities
def hash_password(password: str) -> str:
    # 16-byte random salt
    salt = secrets.token_hex(16)
    # Derive hash using PBKDF2-HMAC-SHA256 (native Python, safe and no compiled binary issues)
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    return f"{salt}:{pwd_hash}"

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, pwd_hash = stored_hash.split(":")
        calc_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        return secrets.compare_digest(calc_hash, pwd_hash)
    except Exception:
        return False

# Database Operations
def db_signup(username: str, password: str) -> dict:
    username = username.strip().lower()
    if not username or not password:
        return {"success": False, "message": "Username and password cannot be empty."}
        
    password_hash = hash_password(password)
    new_user = {
        "username": username,
        "password_hash": password_hash,
        "created_at": datetime.now().isoformat()
    }
    
    if USE_MONGODB:
        try:
            existing = users_col.find_one({"username": username})
            if existing:
                return {"success": False, "message": "Username already exists."}
            
            result = users_col.insert_one(new_user)
            return {
                "success": True, 
                "message": "User registered successfully.", 
                "user_id": str(result.inserted_id),
                "username": username
            }
        except Exception as e:
            return {"success": False, "message": f"Database error: {str(e)}"}
    else:
        # Local JSON file fallback
        local_data = load_local_db()
        for u in local_data["users"]:
            if u["username"] == username:
                return {"success": False, "message": "Username already exists."}
        
        # Simulate ID
        user_id = secrets.token_hex(8)
        new_user["id"] = user_id
        local_data["users"].append(new_user)
        save_local_db(local_data)
        
        return {
            "success": True,
            "message": "User registered successfully (Local DB).",
            "user_id": user_id,
            "username": username
        }

def db_login(username: str, password: str) -> dict:
    username = username.strip().lower()
    if not username or not password:
        return {"success": False, "message": "Username and password cannot be empty."}
        
    if USE_MONGODB:
        try:
            user = users_col.find_one({"username": username})
            if not user:
                return {"success": False, "message": "Invalid username or password."}
            
            if verify_password(password, user["password_hash"]):
                return {
                    "success": True,
                    "user_id": str(user["_id"]),
                    "username": username
                }
            return {"success": False, "message": "Invalid username or password."}
        except Exception as e:
            return {"success": False, "message": f"Database error: {str(e)}"}
    else:
        local_data = load_local_db()
        for u in local_data["users"]:
            if u["username"] == username:
                if verify_password(password, u["password_hash"]):
                    return {
                        "success": True,
                        "user_id": u["id"],
                        "username": username
                    }
                break
        return {"success": False, "message": "Invalid username or password."}

def db_log_prediction(user_id: str, inputs: dict, outputs: dict) -> str:
    log_entry = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "inputs": inputs,
        "outputs": outputs
    }
    
    if USE_MONGODB:
        try:
            result = logs_col.insert_one(log_entry)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error logging prediction to MongoDB: {e}")
            return ""
    else:
        local_data = load_local_db()
        log_id = secrets.token_hex(8)
        log_entry["id"] = log_id
        local_data["prediction_logs"].append(log_entry)
        save_local_db(local_data)
        return log_id

def db_get_logs(user_id: str) -> list:
    if USE_MONGODB:
        try:
            cursor = logs_col.find({"user_id": user_id}).sort("timestamp", -1)
            results = []
            for doc in cursor:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
                results.append(doc)
            return results
        except Exception as e:
            print(f"Error fetching logs from MongoDB: {e}")
            return []
    else:
        local_data = load_local_db()
        results = [l for l in local_data["prediction_logs"] if l["user_id"] == user_id]
        # Sort by timestamp descending
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results

def db_get_all_logs() -> list:
    if USE_MONGODB:
        try:
            cursor = logs_col.find().sort("timestamp", -1)
            results = []
            for doc in cursor:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
                results.append(doc)
            return results
        except Exception as e:
            print(f"Error fetching all logs from MongoDB: {e}")
            return []
    else:
        local_data = load_local_db()
        results = list(local_data["prediction_logs"])
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results
