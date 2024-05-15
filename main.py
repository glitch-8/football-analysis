import cv2
import numpy as np

from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from common.utils import (
    read_video,
    save_video,
    get_path
)
from camera_movement_estimator import CameraMovementEstimator


INPUT_VIDEO_PATH = get_path('data/input/08fd33_4.mp4')
YOLO_MODEL_PATH = get_path('models/last.pt')
STUB_PATH = get_path('data/stubs/track_stub.pkl')
OUTPUT_VIDEO_PATH = get_path('data/output/with_tracks.avi')
CROPPED_IMAGE_PATH = get_path('data/output/cropped_player_image.jpg')
CAMERA_MOVEMENT_PICKLE_PATH = get_path('data/stubs/camera_movement_stub.pkl')


def main():
    global INPUT_VIDEO_PATH, YOLO_MODEL_PATH, STUB_PATH, OUTPUT_VIDEO_PATH
    global CROPPED_IMAGE_PATH, CAMERA_MOVEMENT_PICKLE_PATH

    # read video
    video_frames: list[np.ndarray] = read_video(INPUT_VIDEO_PATH)

    # Initialize tracker
    tracker = Tracker(YOLO_MODEL_PATH)
    tracks = tracker.get_object_tracks(frames=video_frames, read_from_stub=True, stub_path=STUB_PATH)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=CAMERA_MOVEMENT_PICKLE_PATH
    )

    # interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    # assign ball acquisition
    team_ball_control = []
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
        
    team_ball_control = np.array(team_ball_control)

    # draw output
    ## draw output tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # save video
    save_video(
        output_video_frames=output_video_frames,
        output_video_path=get_path(OUTPUT_VIDEO_PATH)
    )


if __name__ == '__main__':
    main()