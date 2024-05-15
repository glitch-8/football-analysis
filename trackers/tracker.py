import os
import pickle

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm

from common.utils import get_center_of_bbox, get_bbox_width


class Tracker:

    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [
            x.get(1, {}).get('bbox', [])
            for x in ball_positions
        ]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()
        ]

        return ball_positions

    def detect_frames(self, frames):
        # add a batch size to not run into memory issues
        # when predicting on the whole video together
        batch_size: int = 20
        detections = list()
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i: i + batch_size], conf=0.1)
            detections.extend(detections_batch)

        # each element in detections is a prediction 
        # for a frame, that contains all the boxes identified in the frame
        # to access the third box coordinates for the second frame in the
        # video do: detections[1][2].boxes

        return detections 

    def get_object_tracks(self, frames, read_from_stub: bool = False, stub_path: str | None = None):
        '''
        Here the read_from_stub will help during development,
        as we don't need to generate the tracks again,
        because once generated can be generated from the pickle file
        '''

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as fp:
                tracks = pickle.load(fp)
            
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            'players': [],
            'referees': [],
            'ball': []
        }

        print(f'Getting player tracks...')
        for frame_num, detection in enumerate(tqdm(detections)):
            cls_names = detection.names
            cls_names_inv = {
                v: k for k, v in cls_names.items()
            }

            # convert to supervision detection format
            # reorganizes the data for all the boxes in a single frame
            # xyxy contains an array of all the bounding box coords
            # class_id contains an array of label for all the boxes
            # confidence is the array containing conf scores for 
            # predictions against all the bounding boxes
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player object
            # by iterating over all the boxes labels
            for object_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_idx] = cls_names_inv["player"]
            
            # track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({}) # since there is only one ball, don't track the ball, on the other hand there can be multiple players/refrees

            # iterate over all the detections (bboxes in all frames)
            # here the frame detection is a tuple containing the following values
            # (bbox, mask, confidence, class_id, tracker_id, data)
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # here in the track object
                # against each class type
                # against each frame
                # we are adding bboxes against each track id encountered
                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {
                        'bbox': bbox
                    }

                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {
                        'bbox': bbox
                    }
                
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {
                        'bbox': bbox
                    }

        print(f'Finished getting player tracks')

        if stub_path is not None:
            with open(stub_path, 'wb') as fp:
                pickle.dump(tracks, fp)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox=bbox)
        width = get_bbox_width(bbox=bbox)

        cv2.ellipse(
            frame, 
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame, 
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f'{track_id}',
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox=bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, -1)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]

        # get the number of times each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        team_1 = team_1_num_frames/(team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f'Team 1 Ball Control: {team_1*100: .2f}%', (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f'Team 2 Ball Control: {team_2*100: .2f}%', (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, frames, tracks, team_ball_control):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # draw annotations for players
            for track_id, player in player_dict.items():
                player_color = player.get('team_color', (0, 0, 225))
                frame = self.draw_ellipse(frame, player["bbox"], player_color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0, 255, 0))
            
            # draw annotations for referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 225))
            
            # draw annotations for referees
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_frames.append(frame)
        
        return output_frames