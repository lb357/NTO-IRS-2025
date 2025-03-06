import cv2

CAMERA = "http://root:admin@10.128.73.80/mjpg/video.mjpg"

if __name__ == "__main__":
    print("Starting...")
    video_stream = cv2.VideoCapture(CAMERA)
    _, frame = video_stream.read()
    cv2.imwrite("init.png", frame)
    video_stream.release()
    print("Done!")