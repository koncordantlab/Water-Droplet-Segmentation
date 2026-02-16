# Import libraries
import time
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import os

model = YOLO('app_root/weights_DP(6).pt')

class DropletTracker:
    """
    Tracks the centroids of droplets and assigns unique IDs to those objects across frames.
    """
    def __init__(self, maxDisappeared=50, maxDistance=50):
        """
        Initialize the tracker.
        
        Args:
            maxDisappeared: Number of frames an object can be missing before deletion
            maxDistance: Maximum distance (pixels) to match objects between frames
        """
        # Counter for assigning new unique IDs
        self.nextObjectID = 0
        
        # Dictionary mapping object ID to its centroid (x, y)
        self.objects = OrderedDict()
        
        # Dictionary mapping object ID to its full bounding box
        self.bboxes = OrderedDict()
        
        # Dictionary counting how many frames each object has been missing
        self.disappeared = OrderedDict()
        
        # Configuration parameters
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
    def register(self, centroid, bbox):
        """
        Register a new object with a unique ID.
        
        Called when we detect an object that doesn't match any existing tracked object.
        """
        self.objects[self.nextObjectID] = centroid
        self.bboxes[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    
    def deregister(self, objectID):
        """
        Remove an object ID from tracking.
        
        Called when an object has been missing for too many frames.
        """
        del self.objects[objectID]
        del self.bboxes[objectID]
        del self.disappeared[objectID]
    
    def update(self, detections):
        """
        Update tracked objects with new detections.
        
        Args:
            detections: List of bounding boxes
        
        Returns:
            Dictionary that maps object IDs to bounding boxes
        """
        
        # No detections
        if len(detections) == 0:
            # Mark all existing objects as missing for one more frame
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                # If missing for too long, delete the ID
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            
            return self.bboxes
        
        # Calculate centroids for all new detections
        inputCentroids = np.zeros((len(detections), 2), dtype="int")
        inputBboxes = []
        
        for (i, (x1, y1, x2, y2)) in enumerate(detections):
            cX = int((x1 + x2) / 2.0) # Midpoint of X values
            cY = int((y1 + y2) / 2.0) # Midpoint of Y Values
            inputCentroids[i] = (cX, cY)
            inputBboxes.append((x1, y1, x2, y2))
        
        # CASE 2: No objects are currently being tracked
        if len(self.objects) == 0:
            # Register all detections as new objects
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], inputBboxes[i])
        
        # CASE 3: We have both tracked objects AND new detections
        else:
            # Get IDs and centroids of currently tracked objects
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            # Calculate distance matrix between all pairs of 
            # existing centroids and new centroids
            # This tells us how far each new detection is from each tracked object
            D = np.zeros((len(objectCentroids), len(inputCentroids)))
            
            for i, oCentroid in enumerate(objectCentroids):
                for j, iCentroid in enumerate(inputCentroids):
                    # Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2)
                    D[i, j] = np.linalg.norm(np.array(oCentroid) - iCentroid)
            
            # Hungarian algorithm: finds optimal matching with minimum total distance
            # This efficiently solves the assignment problem
            rows, cols = linear_sum_assignment(D)
            
            # Track which objects and detections have been matched
            usedRows = set()
            usedCols = set()
            
            # Process each match
            for (row, col) in zip(rows, cols):
                # If distance is too large, don't match them
                # (likely different objects, not the same one moved)
                if D[row, col] > self.maxDistance:
                    continue
                
                # Update the tracked object with new position
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bboxes[objectID] = inputBboxes[col]
                self.disappeared[objectID] = 0
                
                usedRows.add(row)
                usedCols.add(col)
            
            # Find unmatched objects (disappeared in current frame)
            unusedRows = set(range(D.shape[0])) - usedRows
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            
            # Find unmatched detections (new objects appeared)
            unusedCols = set(range(D.shape[1])) - usedCols
            for col in unusedCols:
                self.register(inputCentroids[col], inputBboxes[col])
        
        return self.bboxes

def process_frame():
    return None


def count_all_frames() -> int:
    

    return 0

def process_video_tracking(video_path: str):
    # Track how long it takes
    start_time = time.time()
    
    # Verify video file access
    if not video_path or not os.path.isfile(video_path):
        print(f"❌ Invalid video file path: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return None
    
    # Output video writer initialization
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Build output path
    base_dir = os.path.dirname(video_path)
    video_fname = os.path.basename(video_path)
    video_fname_base = os.path.splitext(video_fname)[0]
    seg_dir = os.path.join(base_dir, "segmentation results")
    os.makedirs(seg_dir, exist_ok=True)
    output_video_path = os.path.join(seg_dir, f"{video_fname_base}_overlay.mp4")

    # Video writer saves results to a video file
    out_video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

    


    print(f"Final Processing Time: {time.time() - start_time}") 
    return None

def main(path):
    # Loop through each frame
    # For each frame, run detection
    # Now compare centroids
    # New things get new ids
    # Things that dropped off get marked differently
    process_video_tracking(path)

    return None

video_path = 'D:\\Github\\NASA-Water-Droplet\\Nasa_Backend\\10 seconds.mp4'

main(video_path)