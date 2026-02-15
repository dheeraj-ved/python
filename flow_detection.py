import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, measure

class FlowDetector:
    """
    Detects flow lines and connections between P&ID elements
    """
    def __init__(self):
        self.flow_lines = []
        self.connections = []
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for flow line detection
        """
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges, img
    
    def detect_flow_lines(self, image_path):
        """
        Detect flow lines using morphological operations and contour detection
        """
        edges, original_img = self.preprocess_image(image_path)
        
        # Apply morphological operations to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Detect contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that represent flow lines
        self.flow_lines = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (lines are longer than wide)
            aspect_ratio = max(w, h) / (min(w, h) + 1)
            if aspect_ratio > 3:  # Line-like contours have high aspect ratio
                self.flow_lines.append({
                    'contour': contour,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'aspect_ratio': aspect_ratio
                })
        
        return self.flow_lines, original_img
    
    def detect_line_direction(self, line_contour):
        """
        Detect the direction of flow line (horizontal, vertical, or diagonal)
        """
        x, y, w, h = cv2.boundingRect(line_contour)
        
        if w > h:
            return 'horizontal'
        elif h > w:
            return 'vertical'
        else:
            return 'diagonal'
    
    def detect_flow_direction(self, image_path, use_arrows=True):
        """
        Detect flow direction using arrow heads or line direction
        """
        flow_lines, img = self.detect_flow_lines(image_path)
        
        if use_arrows:
            flow_info = self._detect_arrow_direction(img)
        else:
            flow_info = []
            for line in flow_lines:
                direction = self.detect_line_direction(line['contour'])
                flow_info.append({
                    'line': line,
                    'direction': direction
                })
        
        return flow_info
    
    def _detect_arrow_direction(self, image):
        """
        Detect arrow heads to determine flow direction
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to enhance arrow heads
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours for arrow detection
        contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        arrows = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # Arrow head size range
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) >= 3:  # Arrow heads are triangular
                    arrows.append({
                        'contour': contour,
                        'approx': approx,
                        'area': area
                    })
        
        return arrows
    
    def connect_elements(self, flow_lines, detected_elements):
        """
        Connect detected flow lines to P&ID elements
        """
        connections = []
        
        for line in flow_lines:
            x, y, w, h = line['x'], line['y'], line['w'], line['h']
            
            # Check which elements are connected to this line
            for element in detected_elements:
                elem_x, elem_y, elem_w, elem_h = element['bbox']
                
                # Check if line touches element
                if self._line_intersects_element(x, y, w, h, elem_x, elem_y, elem_w, elem_h):
                    connections.append({
                        'line': line,
                        'element': element,
                        'intersection_point': (x, y)
                    })
        
        self.connections = connections
        return connections
    
    def _line_intersects_element(self, lx, ly, lw, lh, ex, ey, ew, eh, threshold=10):
        """
        Check if a line intersects with an element (with tolerance)
        """
        # Line bounds
        line_x1, line_y1 = lx, ly
        line_x2, line_y2 = lx + lw, ly + lh
        
        # Element bounds
        elem_x1, elem_y1 = ex, ey
        elem_x2, elem_y2 = ex + ew, ey + eh
        
        # Check intersection with threshold
        x_overlap = not (line_x2 + threshold < elem_x1 or line_x1 - threshold > elem_x2)
        y_overlap = not (line_y2 + threshold < elem_y1 or line_y1 - threshold > elem_y2)
        
        return x_overlap and y_overlap
    
    def draw_flow_detection(self, image_path, flow_lines, connections=None):
        """
        Draw detected flow lines and connections on image
        """
        img = cv2.imread(image_path)
        
        # Draw flow lines
        for line in flow_lines:
            cv2.drawContours(img, [line['contour']], 0, (0, 255, 0), 2)
        
        # Draw connections
        if connections:
            for conn in connections:
                cv2.circle(img, conn['intersection_point'], 5, (255, 0, 0), -1)
        
        return img
    
    def analyze_flow_network(self, flow_lines, elements):
        """
        Analyze the complete flow network
        """
        network_info = {
            'total_lines': len(flow_lines),
            'total_elements': len(elements),
            'connections': self.connect_elements(flow_lines, elements),
            'flow_paths': self._trace_flow_paths(flow_lines)
        }
        
        return network_info
    
    def _trace_flow_paths(self, flow_lines):
        """
        Trace complete flow paths from source to destination
        """
        paths = []
        visited = set()
        
        for i, line in enumerate(flow_lines):
            if i not in visited:
                path = [line]
                visited.add(i)
                
                # Try to connect to adjacent lines
                for j, other_line in enumerate(flow_lines):
                    if j not in visited:
                        if self._lines_connected(line, other_line):
                            path.append(other_line)
                            visited.add(j)
                
                paths.append(path)
        
        return paths
    
    def _lines_connected(self, line1, line2, threshold=20):
        """
        Check if two lines are connected
        """
        """
        x1, y1, w1, h1 = line1['x'], line1['y'], line1['w'], line1['h']
        x2, y2, w2, h2 = line2['x'], line2['y'], line2['w'], line2['h']
        
        # Check if endpoints are close
        endpoints1 = [(x1, y1), (x1 + w1, y1 + h1)]
        endpoints2 = [(x2, y2), (x2 + w2, y2 + h2)]
        
        for p1 in endpoints1:
            for p2 in endpoints2:
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist < threshold:
                    return True
        
        return False