import cv2


def read_video(video_path: str):
    capturer = cv2.VideoCapture(video_path)
    frames = list()

    while True:
        ret, frame = capturer.read()
        if not ret:
            break

        frames.append(frame)
    
    return frames


def save_video(output_video_frames, output_video_path):
    output_fmt = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path,
        output_fmt,
        24,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
    )   
    
    for frame in output_video_frames:
        out.write(frame)
    
    out.release()


if __name__ == '__main__':
    from common.utils.filesystem_helpers import get_path

    def main():
        VID_PATH: str = 'data/input/08fd33_4.mp4'
        frames = read_video(VID_PATH)

        save_video(output_video_frames=frames, output_video_path='data/temp.mp4')
    
    main()