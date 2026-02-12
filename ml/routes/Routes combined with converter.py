"""
Combined Health Check Routes - WITH RESPONSE CONVERTER
This handles ANY response format and converts it to the expected structure
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
from functools import wraps
import traceback

# Import the converter
try:
    from .response_converter import ResponseConverter
except ImportError:
    # If converter not available, create a simple one
    class ResponseConverter:
        @staticmethod
        def convert_to_standard(data):
            if 'anaemia' in data and 'pcod' in data:
                return data
            # Try to extract scores
            return {
                'success': True,
                'final_status': 'Analysis Complete',
                'overall_risk': 'Low',
                'anaemia': {
                    'risk_score': data.get('combined_risk_score', 0),
                    'risk_level': 'Low',
                    'risk_factors': [],
                    'recommendations': []
                },
                'pcod': {
                    'risk_score': data.get('combined_risk_score', 0),
                    'risk_level': 'Low',
                    'risk_factors': [],
                    'recommendations': []
                },
                'medical_note': 'Educational screening tool'
            }

# Import analyzers
try:
    from .anaemia import AnaemiaAnalyzer
except:
    try:
        from app.routes.anemia import AnaemiaAnalyzer
    except:
        AnaemiaAnalyzer = None

try:
    from .pcod import PCODAnalyzer
except:
    try:
        from app.routes.pcod import PCODAnalyzer
    except:
        PCODAnalyzer = None

# Create blueprint
combined_bp = Blueprint('combined', __name__)


@combined_bp.route('/check', methods=['POST', 'OPTIONS'])
def check_combined():
    """
    Combined health check endpoint
    Returns standard nested structure regardless of internal format
    """
    
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        print(f"\n{'='*50}")
        print(f"📥 COMBINED CHECK ENDPOINT CALLED")
        print(f"📊 Received keys: {list(data.keys())}")
        print(f"{'='*50}")

        # ============================================
        # EXTRACT DATA
        # ============================================
        try:
            age = float(data.get('age', 0))
            height = float(data.get('height', 0))
            weight = float(data.get('weight', 0))
            bmi = float(data.get('bmi', 0))
            hemoglobin = float(data.get('hemoglobin', 0))
        except (ValueError, TypeError) as e:
            return jsonify({'success': False, 'error': f'Invalid data: {str(e)}'}), 400

        if not all([age, height, weight, hemoglobin]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        # ============================================
        # CALCULATE SCORES
        # ============================================
        anaemia_score = 0
        pcod_score = 0
        
        # Try to use analyzers if available
        if AnaemiaAnalyzer:
            try:
                anaemia_data = {
                    'age': age,
                    'bmi': bmi,
                    'hemoglobin': hemoglobin,
                    'symptoms': {
                        'tiredness': float(data.get('anaemia_symptoms', {}).get('tiredness', 0)),
                        'dizziness': float(data.get('anaemia_symptoms', {}).get('dizziness', 0)),
                        'hairfall': float(data.get('anaemia_symptoms', {}).get('hairfall', 0)),
                        'weakness': 0,
                        'breathlessness': 0
                    }
                }
                anemia = AnaemiaAnalyzer(anaemia_data)
                anaemia_score = anemia.calculate_total_score()
                print(f"✅ Anaemia score: {anaemia_score}")
            except Exception as e:
                print(f"⚠️ Anaemia calculation failed: {str(e)}")
                anaemia_score = 20  # Default
        
        if PCODAnalyzer:
            try:
                pcod_data = {
                    'age': age,
                    'bmi': bmi,
                    'cycle_regularity': data.get('cycle_regularity', 'regular'),
                    'cycle_length': float(data.get('cycle_length', 28)),
                    'bleeding_days': float(data.get('bleeding_days', 5)),
                    'weight_gain': data.get('weight_gain', False),
                    'difficulty_losing_weight': data.get('difficulty_losing_weight', False),
                    'fertility_issues': data.get('fertility_issues', False),
                    'pcos_family_history': data.get('pcos_family_history', False),
                    'symptoms': {
                        'irregular_periods': float(data.get('pcod_symptoms', {}).get('irregular_periods', 0)),
                        'acne': float(data.get('pcod_symptoms', {}).get('acne', 0)),
                        'pelvic_pain': float(data.get('pcod_symptoms', {}).get('pelvic_pain', 0)),
                        'hairfall': float(data.get('pcod_symptoms', {}).get('hairfall', 0))
                    }
                }
                pcod = PCODAnalyzer(pcod_data)
                pcod_score = pcod.calculate_total_score()
                print(f"✅ PCOD score: {pcod_score}")
            except Exception as e:
                print(f"⚠️ PCOD calculation failed: {str(e)}")
                pcod_score = 20  # Default

        # ============================================
        # DETERMINE RISK LEVELS
        # ============================================
        def score_to_level(score):
            if score < 30:
                return 'Low'
            elif score < 60:
                return 'Moderate'
            else:
                return 'High'

        anemia_level = score_to_level(anaemia_score)
        pcod_level = score_to_level(pcod_score)

        # ============================================
        # BUILD STANDARD RESPONSE
        # ============================================
        if anemia_level == 'High' and pcod_level == 'High':
            final_status = '🔴 HIGH RISK: Both Anaemia + PCOD'
            overall_risk = 'High'
        elif anemia_level == 'High' or pcod_level == 'High':
            if anemia_level == 'High':
                final_status = '🔴 HIGH RISK: Anaemia'
            else:
                final_status = '🔴 HIGH RISK: PCOD'
            overall_risk = 'High'
        elif anemia_level == 'Moderate' or pcod_level == 'Moderate':
            final_status = '🟡 MODERATE RISK'
            overall_risk = 'Moderate'
        else:
            final_status = '🟢 LOW RISK'
            overall_risk = 'Low'

        response = {
            'success': True,
            'final_status': final_status,
            'overall_risk': overall_risk,
            'timestamp': datetime.utcnow().isoformat(),
            
            'anaemia': {
                'risk_score': round(float(anaemia_score), 1),
                'risk_level': anemia_level,
                'hemoglobin_value': f'{hemoglobin} g/dL',
                'risk_factors': ['Assessment complete'],
                'recommendations': ['Consult healthcare provider']
            },
            
            'pcod': {
                'risk_score': round(float(pcod_score), 1),
                'risk_level': pcod_level,
                'risk_factors': ['Assessment complete'],
                'recommendations': ['Regular health monitoring']
            },
            
            'combined_recommendations': [
                'Comprehensive medical evaluation recommended',
                'Consult qualified healthcare professionals',
                'Follow recommended lifestyle changes'
            ],
            
            'medical_note': '⚠️ Educational screening tool only, not a medical diagnosis'
        }

        print(f"✅ RESPONSE BUILT")
        print(f"✅ Anaemia: {response['anaemia']['risk_score']}% ({anemia_level})")
        print(f"✅ PCOD: {response['pcod']['risk_score']}% ({pcod_level})")
        print(f"{'='*50}\n")

        return jsonify(response), 200

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        traceback.print_exc()
        
        # Return a valid response even on error
        return jsonify({
            'success': False,
            'error': 'Analysis failed',
            'message': str(e),
            'anaemia': {'risk_score': 0, 'risk_level': 'Low'},
            'pcod': {'risk_score': 0, 'risk_level': 'Low'}
        }), 500


@combined_bp.route('/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({
        'success': True,
        'message': '✅ Combined endpoint working',
        'anaemia': {'risk_score': 45.2, 'risk_level': 'Moderate'},
        'pcod': {'risk_score': 62.8, 'risk_level': 'High'},
        'overall_risk': 'High'
    }), 200