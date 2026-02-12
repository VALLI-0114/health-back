import os
import sys

from flask import Blueprint, jsonify

# Get the backend root directory
backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add to path if not already there
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

notification_bp = Blueprint("notification", __name__)

@notification_bp.route("/test", methods=["GET"])
def notification_test():
    return jsonify({
        "message": "Notification blueprint is working!"
    }), 200