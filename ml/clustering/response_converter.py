"""
COMPLETE response_converter.py - Response Format Converter
Converts any response format to the expected nested structure
"""


class ResponseConverter:
    """
    Converts various response formats to standard nested structure
    
    Standard output format:
    {
        "success": true,
        "final_status": "...",
        "overall_risk": "...",
        "anaemia": {
            "risk_score": X,
            "risk_level": "...",
            "risk_factors": [],
            "recommendations": []
        },
        "pcod": {
            "risk_score": X,
            "risk_level": "...",
            "risk_factors": [],
            "recommendations": []
        }
    }
    """
    
    @staticmethod
    def convert_to_standard(response_data):
        """
        Convert any response format to standard format
        
        Args:
            response_data (dict): Response from any endpoint
            
        Returns:
            dict: Standardized response format
        """
        
        # If already in correct format, return as-is
        if isinstance(response_data, dict):
            if 'anaemia' in response_data and 'pcod' in response_data:
                print("[ResponseConverter] ✅ Already in standard format")
                return response_data
        
        # CASE 1: Old ML-based format
        if 'combined_health_status' in response_data or 'combined_risk_score' in response_data:
            print("[ResponseConverter] Converting from ML format")
            return ResponseConverter._convert_ml_format(response_data)
        
        # CASE 2: Separate anaemia/pcod response
        if 'anaemia_status' in response_data and 'risk_score' in response_data:
            print("[ResponseConverter] Converting from anaemia format")
            return ResponseConverter._convert_anaemia_format(response_data)
        
        # Default: try to extract what we can
        print("[ResponseConverter] Converting from generic format")
        return ResponseConverter._convert_generic(response_data)
    
    @staticmethod
    def _convert_ml_format(data):
        """
        Convert old ML format to standard format
        
        Old format:
        {
            "combined_health_status": "Normal",
            "combined_risk_score": 1,
            "primary_concerns": [...],
            "recommendations": [...]
        }
        """
        
        print("[ResponseConverter._convert_ml_format] Converting...")
        
        # Extract overall risk from combined score
        combined_score = data.get('combined_risk_score', 0)
        
        def score_to_level(score):
            if score < 30:
                return 'Low'
            elif score < 60:
                return 'Moderate'
            else:
                return 'High'
        
        overall_risk = score_to_level(combined_score)
        
        # Estimate individual scores
        anaemia_score = 0
        pcod_score = 0
        
        # Look for hemoglobin to estimate anaemia score
        hemoglobin = data.get('hemoglobin', 12)
        if hemoglobin < 10:
            anaemia_score = 60
        elif hemoglobin < 12:
            anaemia_score = 40
        else:
            anaemia_score = 10
        
        # Look for cycle info to estimate pcod score
        if 'irregular_periods' in data and data['irregular_periods'] > 60:
            pcod_score = 65
        else:
            pcod_score = 25
        
        converted = {
            'success': True,
            'final_status': data.get('combined_health_status', overall_risk),
            'overall_risk': overall_risk,
            'anaemia': {
                'risk_score': anaemia_score,
                'risk_level': score_to_level(anaemia_score),
                'hemoglobin_value': f"{hemoglobin} g/dL",
                'hemoglobin_status': 'Normal' if hemoglobin >= 12 else 'Low',
                'risk_factors': data.get('primary_concerns', ['Assessment complete']),
                'recommendations': data.get('recommendations', ['Consult healthcare provider'])
            },
            'pcod': {
                'risk_score': pcod_score,
                'risk_level': score_to_level(pcod_score),
                'pcod_status': 'PCOD assessment complete',
                'risk_factors': data.get('primary_concerns', ['Assessment complete']),
                'recommendations': data.get('recommendations', ['Regular monitoring'])
            },
            'combined_recommendations': data.get('recommendations', []),
            'medical_note': data.get('medical_note', 'Educational screening tool only')
        }
        
        print("[ResponseConverter._convert_ml_format] ✅ Conversion complete")
        return converted
    
    @staticmethod
    def _convert_anaemia_format(data):
        """
        Convert anaemia-only format to standard
        
        Input format:
        {
            "risk_score": X,
            "risk_level": "...",
            "anaemia_status": "...",
            "risk_factors": [],
            "recommendations": []
        }
        """
        
        print("[ResponseConverter._convert_anaemia_format] Converting...")
        
        risk_score = data.get('risk_score', 0)
        
        def score_to_level(score):
            if score < 30:
                return 'Low'
            elif score < 60:
                return 'Moderate'
            else:
                return 'High'
        
        risk_level = score_to_level(risk_score)
        
        converted = {
            'success': True,
            'final_status': f"Anaemia: {risk_level}",
            'overall_risk': risk_level,
            'anaemia': {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'hemoglobin_value': data.get('hemoglobin', 'N/A'),
                'hemoglobin_status': data.get('anaemia_status', 'Assessment complete'),
                'risk_factors': data.get('risk_factors', []),
                'recommendations': data.get('recommendations', [])
            },
            'pcod': {
                'risk_score': 0,
                'risk_level': 'Low',
                'risk_factors': [],
                'recommendations': []
            },
            'combined_recommendations': [],
            'medical_note': 'Educational screening tool only'
        }
        
        print("[ResponseConverter._convert_anaemia_format] ✅ Conversion complete")
        return converted
    
    @staticmethod
    def _convert_generic(data):
        """
        Convert generic format to standard
        
        Used when format is unknown
        """
        
        print("[ResponseConverter._convert_generic] Converting generic format...")
        
        converted = {
            'success': True,
            'final_status': 'Analysis Complete',
            'overall_risk': 'Low',
            'anaemia': {
                'risk_score': data.get('risk_score', 0),
                'risk_level': 'Low',
                'hemoglobin_value': data.get('hemoglobin', 'N/A'),
                'risk_factors': data.get('risk_factors', []),
                'recommendations': data.get('recommendations', ['Consult healthcare provider'])
            },
            'pcod': {
                'risk_score': 0,
                'risk_level': 'Low',
                'risk_factors': [],
                'recommendations': ['Regular monitoring']
            },
            'combined_recommendations': data.get('recommendations', []),
            'medical_note': 'Educational screening tool only'
        }
        
        print("[ResponseConverter._convert_generic] ✅ Conversion complete")
        return converted


# ============================================
# WRAPPER DECORATOR
# ============================================
def convert_response(response_func):
    """
    Decorator to convert response to standard format
    
    Usage:
        @combined_bp.route('/check', methods=['POST'])
        @convert_response
        def check_combined():
            ...
    """
    from functools import wraps
    from flask import jsonify
    
    @wraps(response_func)
    def wrapper(*args, **kwargs):
        response = response_func(*args, **kwargs)
        
        # If it's a tuple (response, status_code)
        if isinstance(response, tuple):
            data, status_code = response
            if isinstance(data, dict):
                converted = ResponseConverter.convert_to_standard(data)
                return jsonify(converted), status_code
            return response
        
        # If it's just a dict
        if isinstance(response, dict):
            converted = ResponseConverter.convert_to_standard(response)
            return jsonify(converted)
        
        return response
    
    return wrapper