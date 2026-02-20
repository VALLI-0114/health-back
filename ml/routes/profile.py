"""
PROFILE ROUTES - ml/routes/profile.py
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
from extensions import db
from ml.models.user_profile import UserProfile
from ml.utils.auth_utils import token_required
import os
from werkzeug.utils import secure_filename

profile_bp = Blueprint("profile", __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ================= OPTIONS HANDLERS (NO AUTH) =================
@profile_bp.route("/", methods=["OPTIONS"])
def options_root():
    return "", 200


@profile_bp.route("/upload-photo", methods=["OPTIONS"])
def options_upload():
    return "", 200


@profile_bp.route("/photo", methods=["OPTIONS"])
def options_photo():
    return "", 200


# ================= GET PROFILE (WITH AUTH) =================
@profile_bp.route("/", methods=["GET"])
@token_required
def get_profile(current_user):
    profile = UserProfile.query.get(current_user.id)
    if not profile:
        profile = UserProfile(user_id=current_user.id)
        db.session.add(profile)
        db.session.commit()

    return jsonify({
        "success": True,
        "name": profile.name,
        "age": profile.age,
        "height": profile.height,
        "weight": profile.weight,
        "college": profile.college,
        "district": profile.district,
        "phone": profile.phone,
        "profile_photo": profile.profile_photo
    })


# ================= UPDATE PROFILE (WITH AUTH) =================
@profile_bp.route("/", methods=["PUT"])
@token_required
def update_profile(current_user):
    data = request.get_json()

    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    profile = UserProfile.query.get(current_user.id)
    if not profile:
        profile = UserProfile(user_id=current_user.id)
        db.session.add(profile)

    if "name" in data:
        profile.name = data.get("name")
    if "age" in data:
        profile.age = data.get("age")
    if "height" in data:
        profile.height = data.get("height")
    if "weight" in data:
        profile.weight = data.get("weight")
    if "college" in data:
        profile.college = data.get("college")
    if "district" in data:
        profile.district = data.get("district")
    if "phone" in data:
        profile.phone = data.get("phone")

    profile.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Profile updated",
        "name": profile.name,
        "age": profile.age,
        "height": profile.height,
        "weight": profile.weight,
        "college": profile.college,
        "district": profile.district,
        "phone": profile.phone,
        "profile_photo": profile.profile_photo
    })


# ================= UPLOAD PHOTO (WITH AUTH) =================
@profile_bp.route("/upload-photo", methods=["POST"])
@token_required
def upload_photo(current_user):
    if 'profile_photo' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files['profile_photo']

    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file type. Allowed: PNG, JPG, JPEG, GIF, WEBP"}), 400

    # Check file size (2MB limit)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size > 2 * 1024 * 1024:
        return jsonify({"success": False, "error": "File too large. Max size is 2MB"}), 400

    # Generate safe filename
    filename = secure_filename(
        f"user_{current_user.id}_{int(datetime.utcnow().timestamp())}"
        f".{file.filename.rsplit('.', 1)[1].lower()}"
    )

    # Save to static directory
    static_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static"
    )
    os.makedirs(static_dir, exist_ok=True)
    filepath = os.path.join(static_dir, filename)
    file.save(filepath)

    # Update profile
    profile = UserProfile.query.get(current_user.id)
    if not profile:
        profile = UserProfile(user_id=current_user.id)
        db.session.add(profile)

    # Delete old photo if exists
    if profile.profile_photo:
        old_path = os.path.join(static_dir, profile.profile_photo)
        if os.path.exists(old_path):
            os.remove(old_path)

    profile.profile_photo = filename
    profile.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Photo uploaded successfully",
        "profile_photo": filename
    })


# ================= DELETE PHOTO (WITH AUTH) =================
@profile_bp.route("/photo", methods=["DELETE"])
@token_required
def delete_photo(current_user):
    profile = UserProfile.query.get(current_user.id)

    if not profile or not profile.profile_photo:
        return jsonify({"success": False, "error": "No photo found"}), 404

    static_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static"
    )
    photo_path = os.path.join(static_dir, profile.profile_photo)

    if os.path.exists(photo_path):
        os.remove(photo_path)

    profile.profile_photo = None
    profile.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({"success": True, "message": "Photo deleted successfully"})
