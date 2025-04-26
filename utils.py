import cv2
import numpy as np
import time

class Tracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = dict()
        self.object_data = dict()
        self.disappeared = dict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.events = []
        self.history = []

    def register(self, centroid, class_id, class_name):
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.object_data[object_id] = {
            'class': class_name if class_id >= 0 else "Unknown",
            'class_id': class_id,
            'first_seen': time.time()
        }
        self.disappeared[object_id] = 0
        self.events.append(f"New {self.object_data[object_id]['class']} (ID: {object_id}) detected")
        self.history.append({
            'event': 'appear',
            'id': object_id,
            'class': self.object_data[object_id]['class'],
            'time': time.time()
        })
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id):
        if object_id in self.object_data:
            self.events.append(f"Missing {self.object_data[object_id]['class']} (ID: {object_id})")
            self.history.append({
                'event': 'disappear',
                'id': object_id,
                'class': self.object_data[object_id]['class'],
                'time': time.time()
            })
            deregistered_class = self.object_data[object_id]['class']
            del self.objects[object_id]
            del self.object_data[object_id]
            del self.disappeared[object_id]
            return object_id, deregistered_class
        return None, None

    def update(self, frame, rects, class_ids=None, class_names=None, draw_bbox=True, draw_centroid=True, draw_id=True):
        self.events = []
        deregistered_ids = {}
        current_ids = {}
        
        if class_ids is None:
            class_ids = [-1] * len(rects)
        
        if class_names is None:
            class_names = {-1: "Unknown"}

        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    obj_id, obj_class = self.deregister(object_id)
                    if obj_id is not None:
                        deregistered_ids[obj_id] = {'class': obj_class}
            
            for obj_id in self.objects:
                current_ids[obj_id] = {'class': self.object_data[obj_id]['class']}
                
            return frame, self.events, current_ids, deregistered_ids

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                class_id = class_ids[i]
                class_name = class_names.get(class_id, f"Class {class_id}")
                obj_id = self.register(input_centroids[i], class_id, class_name)
                current_ids[obj_id] = {'class': class_name}
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    D[i, j] = np.sqrt(
                        (object_centroids[i][0] - input_centroids[j][0]) ** 2 +
                        (object_centroids[i][1] - input_centroids[j][1]) ** 2
                    )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if D[row, col] > self.max_distance:
                    continue
                
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                
                class_id = class_ids[col]
                if class_id >= 0:
                    class_name = class_names.get(class_id, f"Class {class_id}")
                    self.object_data[object_id]['class'] = class_name
                    self.object_data[object_id]['class_id'] = class_id
                
                self.disappeared[object_id] = 0
                current_ids[object_id] = {'class': self.object_data[object_id]['class']}
                
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    obj_id, obj_class = self.deregister(object_id)
                    if obj_id is not None:
                        deregistered_ids[obj_id] = {'class': obj_class}
                else:
                    current_ids[object_id] = {'class': self.object_data[object_id]['class']}

            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                class_id = class_ids[col]
                class_name = class_names.get(class_id, f"Class {class_id}")
                obj_id = self.register(input_centroids[col], class_id, class_name)
                current_ids[obj_id] = {'class': class_name}

        for i, (startX, startY, endX, endY) in enumerate(rects):
            closest_centroid = None
            min_dist = float('inf')
            closest_id = None
            
            cx = int((startX + endX) / 2)
            cy = int((startY + endY) / 2)
            
            for obj_id, centroid in self.objects.items():
                dist = np.sqrt((centroid[0] - cx)**2 + (centroid[1] - cy)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_centroid = centroid
                    closest_id = obj_id
            
            if closest_id is not None and min_dist < 30:
                if draw_bbox:
                    class_name = self.object_data[closest_id]['class']
                    color = self.get_color_for_class(self.object_data[closest_id]['class_id'])
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(frame, class_name, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if draw_centroid:
                    cv2.circle(frame, (closest_centroid[0], closest_centroid[1]), 4, (0, 255, 0), -1)
                
                if draw_id:
                    id_text = f"ID: {closest_id}"
                    cv2.putText(frame, id_text, (closest_centroid[0] - 10, closest_centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if len(self.history) > 100:
            self.history = self.history[-100:]

        return frame, self.events, current_ids, deregistered_ids
    
    def get_color_for_class(self, class_id):
        """Generate a consistent color for a class ID."""
        if class_id < 0:
            return (200, 200, 200)
            
        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128),
            (128, 128, 0),
            (128, 0, 128),
            (0, 128, 128),
        ]
        
        return colors[class_id % len(colors)]