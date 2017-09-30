import numpy as np
import cv2

def extract_lane(road_lines):
    left_lane = []
    right_lane = []
    left_slope = []
    right_slope = []

    for x in range(0, len(road_lines)):
        for x1,y1,x2,y2 in road_lines[x]:
            slope = compute_slope(x1,y1,x2,y2)
            if (slope < 0):
                left_lane.append(road_lines[x])
                left_slope.append(slope)
            else:
                if (slope > 0):
                    right_lane.append(road_lines[x])
                    right_slope.append(slope)

    return left_lane, right_lane , left_slope, right_slope
    

def compute_slope(x1,y1,x2,y2):
    if x2!=x1:
        return ((y2-y1)/(x2-x1))

def print_lanes(left_lane, right_lane, left_slope, right_slope):
    print("Left lane")
    for x in range(0, len(left_lane)):
        print(left_lane[x], left_slope[x])
    print("Right lane")
    for x in range(0, len(right_lane)):
        print(right_lane[x], right_slope[x])
    
    
    
if __name__== "__main__":
    cap = cv2.VideoCapture('challenge.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()

        #Escape when no frame is captured / End of Video
        if frame is None:
            break
        
        # Color space conversion
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2. cvtColor(frame, cv2.COLOR_BGR2HLS)
        ysize = img_gray.shape[0]
        xsize = img_gray.shape[1]

        #Detecting yellow and white colors
        low_yellow = np.array([20, 100, 100])
        high_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(img_hsv, low_yellow, high_yellow)
        mask_white = cv2.inRange(img_gray, 200, 255)

        mask_yw = cv2.bitwise_or(mask_yellow, mask_white)
        mask_onimage = cv2.bitwise_and(img_gray, mask_yw)

        #Smoothing for removing noise
        gray_blur = cv2.GaussianBlur(mask_onimage, (5,5), 0)

        #Region of Interest Extraction
        mask_roi = np.zeros(img_gray.shape, dtype=np.uint8) 
        left_bottom = [0, ysize]
        right_bottom = [xsize-0, ysize]
        apex_left = [((xsize/2)-50), ((ysize/2)+50)]
        apex_right = [((xsize/2)+50), ((ysize/2)+50)]
        mask_color = 255

        roi_corners = np.array([[left_bottom, apex_left, apex_right, right_bottom]], dtype=np.int32)
        
        cv2.fillPoly(mask_roi, roi_corners, mask_color)
        image_roi = cv2.bitwise_and(gray_blur, mask_roi)

        #Thresholding before edge
        ret, img_postthresh = cv2.threshold(image_roi, 50, 255, cv2.THRESH_BINARY)

        #Use canny edge detection
        edge_low = 50
        edge_high = 200
        img_edge = cv2.Canny(img_postthresh, edge_low, edge_high)

        #Hough Line Draw
        minLength = 20
        maxGap = 10
        road_lines = cv2.HoughLinesP(img_postthresh, 1, np.pi/180, 20, minLength, maxGap)
        left_lane, right_lane, left_slope, right_slope = extract_lane(road_lines)
        if road_lines is not None:
            for x in range(0, len(road_lines)):
                for x1,y1,x2,y2 in road_lines[x]:
                    cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),3)
                    #print(road_lines[x])

        #print_lanes(left_lane, right_lane, left_slope, right_slope)
        #print("Frame Completed")
        

        #cv2.imshow('Overlay',mask_onimage)
        cv2.imshow('Image',frame)
        cv2.imshow('Post Threshold',img_postthresh)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


