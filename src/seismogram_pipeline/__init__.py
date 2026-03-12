"""
Skates 3 : Python 3 version of skates
"""

__version__ = "1.0.0"

# Import main functions for easy access
from .core.roi_detection import get_roi, corners_to_geojson
from .core.meanline_detection import detect_meanlines, meanlines_to_geojson
from .core.intersection_detection import find_intersections
from .core.trace_segmentation import get_segments, segments_to_geojson
from .core.segment_assignment import assign_segments_to_meanlines
from .core.endpoints import get_endpoint_data

__all__ = [
    'get_roi',
    'corners_to_geojson', 
    'detect_meanlines',
    'meanlines_to_geojson',
    'find_intersections',
    'get_segments',
    'segments_to_geojson',
    'assign_segments_to_meanlines',
    'get_endpoint_data'
]
