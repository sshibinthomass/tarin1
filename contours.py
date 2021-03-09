import cv2 as cv


def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)  # INTER_CUBIC when enlarging


# Read Video
capture = cv.VideoCapture(0)  # 'videos/vid2.mp4'
while True:
    _, frame = capture.read()  # Take input
    frame_resized = rescaleFrame(frame)  # Resize
    grey = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)  # To Grey
    blur = cv.GaussianBlur(grey, (7, 7), cv.BORDER_DEFAULT)  # To Blur
    edge = cv.Canny(blur, 125, 175)  # To detect edge
    #contours, hierarchies = cv.findContours(edge, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Display
    cv.imshow('Resized', edge)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
