#!/usr/bin/env python3
"""
Quick diagnostic script - Run this while your Flask server is running
"""
import requests

print("="*60)
print("DIAGNOSING PROFILE ENDPOINT")
print("="*60)

# Test 1: Check if profile routes are registered
print("\n1. Checking registered routes...")
try:
    response = requests.get("http://localhost:5000/routes")
    all_routes = response.text.split('\n')
    profile_routes = [r for r in all_routes if '/profile' in r]
    
    print(f"   Found {len(profile_routes)} profile routes:")
    for route in profile_routes:
        print(f"     {route}")
    
    # Check specifically for OPTIONS
    options_routes = [r for r in profile_routes if 'OPTIONS' in r]
    if options_routes:
        print(f"\n   ✅ OPTIONS routes found: {len(options_routes)}")
    else:
        print(f"\n   ❌ NO OPTIONS ROUTES FOUND!")
        print(f"   → The profile blueprint isn't registering OPTIONS properly")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Try OPTIONS directly
print("\n2. Testing OPTIONS /api/profile/...")
try:
    response = requests.options(
        "http://localhost:5000/api/profile/",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization"
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ OPTIONS working!")
        print(f"   Headers:")
        print(f"     Allow-Origin: {response.headers.get('Access-Control-Allow-Origin')}")
        print(f"     Allow-Methods: {response.headers.get('Access-Control-Allow-Methods')}")
    elif response.status_code == 404:
        print(f"   ❌ 404 - Route not found")
        print(f"   → Blueprint might not be imported/registered in app.py")
    else:
        print(f"   ⚠️  Unexpected status: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("RECOMMENDED ACTIONS:")
print("="*60)

print("""
If OPTIONS routes are NOT showing up in /routes:

1. Check app.py - verify this line exists:
   from ml.routes.profile import profile_bp
   
2. Check app.py - verify this line exists:
   app.register_blueprint(profile_bp, url_prefix="/api/profile")

3. Restart Flask server:
   - Press Ctrl+C
   - Run: python app.py
   
4. Check Flask startup output for errors

If OPTIONS routes ARE showing but still get 404:
   - This is very unusual
   - Share your complete app.py file
""")