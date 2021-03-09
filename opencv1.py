import cv2 as cv


def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)  # INTER_CUBIC when enlarging


# #Read Image
# img = cv.imread('photos/dog.jpg')
# frame_resized=rescaleFrame(img)
# cv.imshow('Resized',frame_resized)
# cv.imshow('Cat', img)
# cv.waitKey(1000)

# Read Video
capture = cv.VideoCapture(0)  # 'videos/vid2.mp4'
while True:
    _, frame = capture.read()  # Take input
    frame_resized = rescaleFrame(frame)  # Resize
    grey = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)  # To Grey
    blur = cv.GaussianBlur(frame_resized, (7, 7), cv.BORDER_DEFAULT)  # To Blur
    edge = cv.Canny(blur, 125, 175)  # To detect edge
    dilation = cv.dilate(edge, (3, 3), iterations=3)  # To increase edge size
    eroded = cv.erode(dilation, (3, 3), iterations=3)  # To reduce edge size
    contours, hierarchies = cv.findContours(edge, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Display
    cv.imshow('Resized', frame_resized)
    cv.imshow('Video', frame)
    cv.imshow('Grey', grey)
    cv.imshow('Blur', blur)
    cv.imshow('Edge', edge)
    cv.imshow('Dilate', dilation)
    cv.imshow('erode', eroded)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
