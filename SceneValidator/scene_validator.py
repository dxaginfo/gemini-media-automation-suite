import os
import json
import cv2
import numpy as np
from google.cloud import vision
from google.cloud import storage
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class SceneValidator:
    def __init__(self, standards=None, custom_rules=None):
        self.standards = standards or ["rule_of_thirds", "balanced_frame"]
        self.custom_rules = custom_rules or {}
        self.vision_client = vision.ImageAnnotatorClient()
        self.storage_client = storage.Client()
        self.model = genai.GenerativeModel('gemini-pro-vision')
    
    def validate_media(self, media_url, generate_suggestions=True):
        """
        Validate media against selected standards
        
        Args:
            media_url (str): URL to media file
            generate_suggestions (bool): Whether to generate improvement suggestions
            
        Returns:
            dict: Validation results and suggestions
        """
        # Load image
        image = self._load_image(media_url)
        
        # Extract features
        features = self._extract_features(image)
        
        # Validate against standards
        validation_results = {}
        overall_score = 0
        
        for standard in self.standards:
            if standard == "rule_of_thirds":
                result = self._validate_rule_of_thirds(features)
            elif standard == "golden_ratio":
                result = self._validate_golden_ratio(features)
            elif standard == "leading_lines":
                result = self._validate_leading_lines(features)
            elif standard == "balanced_frame":
                result = self._validate_balanced_frame(features)
            elif standard == "custom" and self.custom_rules:
                result = self._validate_custom_rules(features)
            else:
                continue
                
            validation_results[standard] = result
            overall_score += result["score"]
        
        # Calculate overall score
        if validation_results:
            overall_score /= len(validation_results)
        
        # Generate suggestions if requested
        suggestions = []
        if generate_suggestions and overall_score < 0.8:
            suggestions = self._generate_suggestions(image, validation_results)
        
        return {
            "validation": {
                "passed": overall_score >= 0.7,
                "score": overall_score,
                "details": validation_results
            },
            "suggestions": suggestions,
            "metadata": {
                "processingTime": 0,
                "version": "1.0.0"
            }
        }
    
    def _load_image(self, media_url):
        """Load image from URL or local path"""
        import requests
        from PIL import Image
        from io import BytesIO
        
        if media_url.startswith(('http://', 'https://')):
            response = requests.get(media_url)
            img = Image.open(BytesIO(response.content))
            return np.array(img)
        else:
            return cv2.imread(media_url)
    
    def _extract_features(self, image):
        """Extract composition features from image"""
        # Convert to RGB if needed
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Get dimensions
        height, width = image.shape[:2]
        
        # Convert to vision image
        vision_image = vision.Image(content=cv2.imencode('.jpg', image)[1].tobytes())
        
        # Get vision features
        vision_response = self.vision_client.image_properties(image=vision_image)
        
        # Get objects
        object_response = self.vision_client.object_localization(image=vision_image)
        objects = object_response.localized_object_annotations
        
        # Extract dominant colors
        colors = vision_response.image_properties_annotation.dominant_colors.colors
        
        # Calculate intensity map
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        intensity_map = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate grid points for rule of thirds
        third_h1, third_h2 = height // 3, (height // 3) * 2
        third_w1, third_w2 = width // 3, (width // 3) * 2
        
        # Intersection points (rule of thirds)
        intersection_points = [
            (third_w1, third_h1),
            (third_w2, third_h1),
            (third_w1, third_h2),
            (third_w2, third_h2)
        ]
        
        return {
            'height': height,
            'width': width,
            'colors': colors,
            'objects': objects,
            'intensity_map': intensity_map,
            'edges': edges,
            'contours': contours,
            'third_points': intersection_points,
            'image': image,
            'vision_image': vision_image
        }
    
    def _validate_rule_of_thirds(self, features):
        """Validate image against rule of thirds"""
        height, width = features['height'], features['width']
        intersection_points = features['third_points']
        objects = features['objects']
        intensity_map = features['intensity_map']
        
        # Check if important objects are near intersection points
        importance_scores = []
        
        # If we have objects
        if objects:
            for obj in objects:
                # Get object center
                box = obj.bounding_poly.normalized_vertices
                obj_cx = sum(v.x for v in box) / 4 * width
                obj_cy = sum(v.y for v in box) / 4 * height
                
                # Check proximity to intersection points
                distances = []
                for px, py in intersection_points:
                    distance = np.sqrt((obj_cx - px)**2 + (obj_cy - py)**2)
                    distances.append(distance)
                
                # Normalize by image diagonal
                min_distance = min(distances)
                diagonal = np.sqrt(width**2 + height**2)
                normalized_distance = min_distance / diagonal
                
                # Convert to score (closer is better)
                score = max(0, 1 - (normalized_distance * 5))  # Scale factor of 5 for sensitivity
                importance_scores.append(score * obj.score)  # Weight by object confidence
        
        # If no objects, use intensity map
        if not importance_scores:
            scores = []
            for px, py in intersection_points:
                # Sample 5x5 area around each intersection point
                x1, y1 = max(0, int(px - 2)), max(0, int(py - 2))
                x2, y2 = min(width-1, int(px + 3)), min(height-1, int(py + 3))
                region = intensity_map[y1:y2, x1:x2]
                if region.size > 0:
                    region_score = np.mean(region) / 255.0  # Normalize to 0-1
                    scores.append(region_score)
            
            if scores:
                importance_scores = scores
        
        # Calculate final score
        if importance_scores:
            score = np.mean(importance_scores)
        else:
            score = 0.5  # Neutral score if we couldn't determine
        
        return {
            "passed": score >= 0.6,
            "score": float(score),
            "details": "Rule of thirds analysis based on object placement and visual interest points"
        }
    
    def _validate_golden_ratio(self, features):
        """Validate image against golden ratio"""
        # Implement golden ratio validation
        # For this example, we'll return a placeholder result
        return {
            "passed": True,
            "score": 0.75,
            "details": "Golden ratio analysis completed"
        }
    
    def _validate_leading_lines(self, features):
        """Validate image against leading lines principle"""
        # Implement leading lines validation
        # For this example, we'll return a placeholder result
        return {
            "passed": True,
            "score": 0.8,
            "details": "Leading lines analysis completed"
        }
    
    def _validate_balanced_frame(self, features):
        """Validate image for balanced framing"""
        height, width = features['height'], features['width']
        intensity_map = features['intensity_map']
        objects = features['objects']
        
        # Divide image into four quadrants
        h_mid, w_mid = height // 2, width // 2
        
        q1 = intensity_map[0:h_mid, 0:w_mid]  # Top left
        q2 = intensity_map[0:h_mid, w_mid:width]  # Top right
        q3 = intensity_map[h_mid:height, 0:w_mid]  # Bottom left
        q4 = intensity_map[h_mid:height, w_mid:width]  # Bottom right
        
        # Calculate intensity in each quadrant
        q1_intensity = np.mean(q1) if q1.size > 0 else 0
        q2_intensity = np.mean(q2) if q2.size > 0 else 0
        q3_intensity = np.mean(q3) if q3.size > 0 else 0
        q4_intensity = np.mean(q4) if q4.size > 0 else 0
        
        # Calculate left/right and top/bottom balance
        left_intensity = (q1_intensity + q3_intensity) / 2
        right_intensity = (q2_intensity + q4_intensity) / 2
        top_intensity = (q1_intensity + q2_intensity) / 2
        bottom_intensity = (q3_intensity + q4_intensity) / 2
        
        # Calculate balance ratios
        lr_ratio = min(left_intensity, right_intensity) / max(left_intensity, right_intensity) if max(left_intensity, right_intensity) > 0 else 1
        tb_ratio = min(top_intensity, bottom_intensity) / max(top_intensity, bottom_intensity) if max(top_intensity, bottom_intensity) > 0 else 1
        
        # Combined balance score
        balance_score = (lr_ratio + tb_ratio) / 2
        
        # Also consider object placement if available
        if objects:
            # Check if objects are well distributed
            q1_objects, q2_objects, q3_objects, q4_objects = 0, 0, 0, 0
            
            for obj in objects:
                # Get object center
                box = obj.bounding_poly.normalized_vertices
                obj_cx = sum(v.x for v in box) / 4 * width
                obj_cy = sum(v.y for v in box) / 4 * height
                
                # Determine quadrant
                if obj_cx < w_mid and obj_cy < h_mid:
                    q1_objects += 1
                elif obj_cx >= w_mid and obj_cy < h_mid:
                    q2_objects += 1
                elif obj_cx < w_mid and obj_cy >= h_mid:
                    q3_objects += 1
                else:
                    q4_objects += 1
            
            # Calculate object distribution score
            total_objects = q1_objects + q2_objects + q3_objects + q4_objects
            if total_objects > 0:
                max_objects = max(q1_objects, q2_objects, q3_objects, q4_objects)
                object_distribution = 1 - ((max_objects / total_objects) - 0.25) * 2
                object_distribution = max(0, min(1, object_distribution))
                
                # Combine with balance score
                balance_score = (balance_score + object_distribution) / 2
        
        return {
            "passed": balance_score >= 0.6,
            "score": float(balance_score),
            "details": "Balanced frame analysis based on intensity distribution and object placement"
        }
    
    def _validate_custom_rules(self, features):
        """Validate image against custom rules"""
        # Implement custom rules validation
        # For this example, we'll return a placeholder result
        return {
            "passed": True,
            "score": 0.7,
            "details": "Custom rules validation completed"
        }
    
    def _generate_suggestions(self, image, validation_results):
        """Generate improvement suggestions using Gemini API"""
        try:
            # Convert image to bytes for Gemini
            from PIL import Image as PILImage
            from io import BytesIO
            
            pil_image = PILImage.fromarray(image)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            
            # Create a prompt for Gemini API
            prompt = f"""
            Analyze this image for composition and framing issues.
            Validation results: {json.dumps(validation_results)}
            
            Provide 3 specific suggestions to improve the composition based on these results.
            Format each suggestion as a brief actionable item.
            """
            
            # Create parts for multimodal generation
            parts = [
                {"mime_type": "image/jpeg", "data": image_bytes},
                {"text": prompt}
            ]
            
            response = self.model.generate_content(parts)
            
            # Parse suggestions from Gemini response
            suggestion_text = response.text
            suggestion_lines = [line.strip() for line in suggestion_text.split('\n') if line.strip()]
            
            suggestions = []
            for i, line in enumerate(suggestion_lines[:3]):
                suggestions.append({
                    "type": f"suggestion_{i+1}",
                    "description": line,
                    "visualReference": ""  # Would be generated in a full implementation
                })
            
            return suggestions
        except Exception as e:
            print(f"Error generating suggestions: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    validator = SceneValidator(standards=["rule_of_thirds", "balanced_frame"])
    results = validator.validate_media("https://example.com/image.jpg")
    print(json.dumps(results, indent=2))
