import numpy as np
import cv2
import os
import math
import time

from midetection.Utils import echo_config
from torch import greater
from midetection.Utils.utilities import log_text, create_directory, colors

l_data = echo_config.l_traindata                      # labels for training set
input_data = echo_config.training_mask_path
l_testdata = echo_config.l_testdata                   # labels for test set
testInput_data = echo_config.testing_mask_path
l_validdata = echo_config.l_validdata                 # labels for test set
validInput_data = echo_config.validating_mask_path

ldata_dir = sorted(os.listdir(l_data))
ltestdata_dir = sorted(os.listdir(l_testdata))
lvaliddata_dir = sorted(os.listdir(l_validdata))
label_dir = [ldata_dir, lvaliddata_dir, ltestdata_dir]

input_dir = sorted(os.listdir(input_data))
testInput_dir = sorted(os.listdir(testInput_data))
testValid_dir = sorted(os.listdir(validInput_data))

labelsdata_folder = './contours'
labelsTraining_folder = './contours/training'
labelsTest_folder = './contours/test'

create_directory(labelsdata_folder, "contours directory")
create_directory(labelsTraining_folder, "contours directory")
create_directory(labelsTest_folder, "contours directory")

if os.path.isfile("./logger/contours directory.txt"):
    os.remove("./logger/contours directory.txt")
if os.path.isfile("./logger/wallMotion.txt"):
    os.remove("./logger/wallMotion.txt")
if os.path.isfile("./logger/myocardialThickening.txt"):
    os.remove("./logger/myocardialThickening.txt")

# Finds a pixel position at ratio distance from lowerPoint to upperPoint
def find_pixelInBetween(lowerPoint, upperPoint, ratio):
     return (((1-ratio)*lowerPoint[0]) + (ratio*upperPoint[0]), ((1-ratio)*lowerPoint[1]) + (ratio*upperPoint[1]))

# Finds the end points of a fitted line passing through either the left or right contour of left ventricle
def get_pointsFromFittedLine(contour, apex, reverse=False):
    contourBottom = tuple(contour[0][contour[0][:, :, 1].argmax()][0])     
    vx, vy, x, y = cv2.fitLine(contour[0], cv2.DIST_WELSCH, 0, 0.01, 0.01)      # (x,y) is a point in the contour and (vx,vy) is unit vector parallel to the fitted line
    fittedSlope = vy / vx                                                       # slope of fitted line  = vy/vx
    x2 = ((y - apex[1]) / fittedSlope) + x                                      # get x-value of fitted line at apex height ---> upperPoint
    x3 = ((y - contourBottom[1]) / fittedSlope) + x                             # get x-value of fitted line at bottom height ---> bottomPoint
    if reverse == True:                                                         # for right contour
        upperPoint = (x3, apex[1])
        bottomPoint = (x2, contourBottom[1])
    else:                                                                       # for left contour
        upperPoint = (x2, apex[1])
        bottomPoint = (x3, contourBottom[1])

    # Uncomment codes below to plot the fitted line of left contour
    # lefty = int((-x * fittedSlope) + y)                                         # get y-value at x=0 (slope intercept eq) ---> for point 1                                             
    # righty = int(((apex[0] + 1 - x) * fittedSlope) + y)                         # get y-value at x=upper right edge (point slope eq) ---> for point 2                  
    # point1 = (0, lefty)
    # point2 = (apex[0], righty)
    # cv2.line(img, point1, point2, colors['GREEN'], 1)                           # lefty and righty are out of bounds of img dim
                  
    return bottomPoint, upperPoint

# Returns the 3 segments on either side of the left ventricle
def get_segmentsFromContour(contour, referencePoints):
    segment1 = []
    segment2 = []
    segment3 = []
    apicalCap = []
    for point in contour[0]:
        if point[0][1] > referencePoints[0][1]:
            segment1.append([[point[0][0],point[0][1]]])
        elif point[0][1] > referencePoints[1][1]:
            segment2.append([[point[0][0],point[0][1]]])
        elif point[0][1] > referencePoints[2][1]:               # revert to referencePoints[2][1] if you want to include apical cap
            segment3.append([[point[0][0],point[0][1]]])
        else:
            apicalCap.append([[point[0][0],point[0][1]]])

    segment1 = (np.array(segment1), )
    segment2 = (np.array(segment2), )
    segment3 = (np.array(segment3), )

    return segment1, segment2, segment3

# Finds the center of mass of each segment using moments --> use for irregular shapes such as segment 3
def get_centerOfSegment(segment):
    M = cv2.moments(segment[0])
    centerSeg = (M['m10'] / (M['m00'] + 1e-5), M['m01'] / (M['m00'] + 1e-5))

    return centerSeg

# Finds the center of all segments
# minAreaRect returns centroid, dimension, angle
def get_centerOfAllSegments(segment1, segment2, segment3, segment5, segment6, segment7):
    log_text("      Finding the center of each segment.\n", "contours directory.txt")
    centerSeg1, _, _ = cv2.minAreaRect(segment1[0])                                
    centerSeg2, _, _ = cv2.minAreaRect(segment2[0])
    centerSeg3, _, _ = cv2.minAreaRect(segment3[0])
    # centerSeg3       = get_centerOfSegment(segment3)
    centerSeg5, _, _ = cv2.minAreaRect(segment5[0])
    centerSeg6, _, _ = cv2.minAreaRect(segment6[0])
    centerSeg7, _, _ = cv2.minAreaRect(segment7[0])

    return [centerSeg1, centerSeg2, centerSeg3, centerSeg5, centerSeg6, centerSeg7]

# Finds the thickness of a segment ---> use for segments of irregular shape
# Get the extreme left and extreme right of the segment center
# Find the difference to get thickness
def get_thicknessOfSegment(segment, centers):
    alignedPoints = []
    for point in segment[0]:
        if point[0][1] == centers[1]:
            alignedPoints.append(point[0][0])
    if len(alignedPoints) > 1:
        alignedPoints.sort()
        return alignedPoints[-1] - alignedPoints[0]
    return 0

def find_smallerDimension(dim):
    if dim[0] < dim[1]:
        return dim[0]
    return dim[1]

# Finds the thickness of all segments of the left ventricle
def get_thicknessOfAllSegments(segment1, segment2, segment3, segment5, segment6, segment7, centers):
    log_text("      Measuring the thickness of each segment.\n", "contours directory.txt")
    _, dim1, _ = cv2.minAreaRect(segment1[0])                                
    _, dim2, _ = cv2.minAreaRect(segment2[0])
    _, dim3, _ = cv2.minAreaRect(segment3[0])
    _, dim5, _ = cv2.minAreaRect(segment5[0])
    _, dim6, _ = cv2.minAreaRect(segment6[0])
    _, dim7, _ = cv2.minAreaRect(segment7[0])

    thickness1 = find_smallerDimension(dim1)
    thickness2 = find_smallerDimension(dim2)
    thickness3 = find_smallerDimension(dim3)
    thickness5 = find_smallerDimension(dim5)
    thickness6 = find_smallerDimension(dim6)
    thickness7 = find_smallerDimension(dim7)

    # thickness1 = get_thicknessOfSegment(segment1, centers[0])
    # thickness2 = get_thicknessOfSegment(segment2, centers[1])
    # thickness3 = get_thicknessOfSegment(segment3, centers[2])
    # thickness5 = get_thicknessOfSegment(segment5, centers[3])
    # thickness6 = get_thicknessOfSegment(segment6, centers[4])
    # thickness7 = get_thicknessOfSegment(segment7, centers[5])
  
    # print(str([dim1, dim2, dim3, dim5, dim6, dim7]))
    return [thickness1, thickness2, thickness3, thickness5, thickness6, thickness7]

def get_centerAndThicknessOfAllSegments():
    start = time.time()
    centers = {}
    thickness = {}
    im_index = []
    count = 0
    log_text("Saving contours of labels.\n", "contours directory.txt")
    for dir in label_dir:
        for i, image in enumerate(dir):
            # Get the boundary of the left ventricle
            log_text(str(image) + ": \n", "contours directory.txt")
            if dir == ldata_dir:
                img = cv2.imread(l_data + image)
                # org_img = cv2.imread(input_data + image)
                org_img = cv2.imread(input_data + input_dir[i])
            elif dir == ltestdata_dir:
                img = cv2.imread(l_testdata + image)
                # org_img = cv2.imread(testInput_data + image)
                org_img = cv2.imread(testInput_data + testInput_dir[i])
            else:
                img = cv2.imread(l_validdata + image)
                # org_img = cv2.imread(testInput_data + image)
                org_img = cv2.imread(validInput_data + testValid_dir[i])
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # org_imgray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
            org_imgray = cv2.cvtColor(org_img, cv2.COLOR_RGBA2GRAY)
            _, thresh = cv2.threshold(imgray, 128, 255, 0)
            boundary, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            boundary = sorted(boundary, key=cv2.contourArea, reverse=True)   # get only the contour for the largest area aka the LV
            
            # Find the pixels inside the boundary
            padding = []
            mask_pts = np.where(imgray[:,:] > 128)
            numPts = len(mask_pts[0])
            for index in range(numPts):
                padding.append([[mask_pts[1][index], mask_pts[0][index]]])

            removed = []

            # Remove pixels in padding which already exists in boundary
            paddingCopy = padding.copy()
            for current in paddingCopy:
                for point in boundary[0]:
                    if (current[0][0] == point[0][0]) and (current[0][1] == point[0][1]) and (current[0] not in removed):
                        padding.remove(current)
                        removed.append(current[0])
            padding = (np.array(padding), )
          
            # Get all pixels of the left ventricle
            contours = np.concatenate((boundary[0], padding[0]), axis=0)
            contours = (np.array(contours), )
           
            # Determine the extreme points of the left ventricle, axis 1 looks at horizontal slices, axis 0 looks at vertical slices
            extLeft = tuple(contours[0][contours[0][:, :, 0].argmin()][0])
            extRight = tuple(contours[0][contours[0][:, :, 0].argmax()][0])
            extTop = tuple(contours[0][contours[0][:, :, 1].argmin()][0])
            extBottom = tuple(contours[0][contours[0][:, :, 1].argmax()][0])
            centerTop = ((((extRight[0] - extLeft[0])/2) + extLeft[0]), extTop[1])   # get the horizontal center between the extreme left and extreme right of the contour
            apex = ((((centerTop[0] - extTop[0])/2) + extTop[0]), extTop[1])         # approximate the apex of the LV

            # Separate the left side and right side pixels of the left ventricle
            log_text("      Dividing the contours into left and right contours.\n", "contours directory.txt")
            leftContour = []
            rightContour = []
            belowApex = []
            
            for point in contours[0]:
                if point[0][0] < (apex[0]):                                            # if x coordinate of a point is on the left of the apex, store on leftContour
                    leftContour.append([[point[0][0],point[0][1]]])
                if point[0][0] > (apex[0]):                                            # if x coordinate of a point is on the right of the apex, store on rightContour
                    rightContour.append([[point[0][0],point[0][1]]])
                if point[0][0] == (apex[0]):                                           # if point has same x coordinate as apex, add to left and right contour
                    leftContour.append([[point[0][0],point[0][1]]])
                    rightContour.append([[point[0][0],point[0][1]]])
                if point[0][0] == int(apex[0]):
                    belowApex.append(point[0][1])                                       # store all y-values of exiting pixels below extTop
            belowApex.sort(reverse=True)
            # bottomApex = (apex[0], belowApex[0])

            leftContour = (np.array(leftContour), )
            rightContour = (np.array(rightContour), )

            # Get the 7 segments of the left ventricle: 3 on left, 3 on right, and apical cap
            # CONCEPT:
            # let L be the total length of the left segment, L = 2L/7 + 2L/7 + 2L/7 + L/7
            # let R be the total length of the left segment, R = 2R/7 + 2R/7 + 2R/7 + R/7
            # approximate a line on each contour to be able to approximate its length
            log_text("      Dividing the left contour into different segments.\n", "contours directory.txt")
            bottomLPoint, upperLPoint = get_pointsFromFittedLine(leftContour, apex)
            refSeg1 = find_pixelInBetween(bottomLPoint, upperLPoint, ratio=2/7)
            refSeg2 = find_pixelInBetween(bottomLPoint, upperLPoint, ratio=4/7)
            refSeg3 = find_pixelInBetween(bottomLPoint, upperLPoint, ratio=5.1/7)
            segment1, segment2, segment3 = get_segmentsFromContour(leftContour, [refSeg1, refSeg2, refSeg3])

            # Do it also for the right contour, segment positions are now reversed
            log_text("      Dividing the right contour into different segments.\n", "contours directory.txt")
            bottomRPoint, upperRPoint = get_pointsFromFittedLine(rightContour, apex, reverse=True)
            refSeg7 = find_pixelInBetween(bottomRPoint, upperRPoint, ratio=2/7)
            refSeg6 = find_pixelInBetween(bottomRPoint, upperRPoint, ratio=4/7)
            refSeg5 = find_pixelInBetween(bottomRPoint, upperRPoint, ratio=5.4/7)
            segment7, segment6, segment5 = get_segmentsFromContour(rightContour, [refSeg7, refSeg6, refSeg5])

            # Find the center of each segment
            centers[count] = get_centerOfAllSegments(segment1, segment2, segment3, segment5, segment6, segment7)

            # Find the thickness of each segment
            thickness[count] = get_thicknessOfAllSegments(segment1, segment2, segment3, segment5, segment6, segment7, centers[count])


            rect1 = cv2.minAreaRect(segment1[0])
            rect1 = cv2.boxPoints(rect1)
            rect1 = np.int0(rect1)                                
            rect2 = cv2.minAreaRect(segment2[0])
            rect2 = cv2.boxPoints(rect2)
            rect2 = np.int0(rect2) 
            rect3 = cv2.minAreaRect(segment3[0])
            rect3 = cv2.boxPoints(rect3)
            rect3 = np.int0(rect3) 
            rect5 = cv2.minAreaRect(segment5[0])
            rect5 = cv2.boxPoints(rect5)
            rect5 = np.int0(rect5) 
            rect6 = cv2.minAreaRect(segment6[0])
            rect6 = cv2.boxPoints(rect6)
            rect6 = np.int0(rect6) 
            rect7 = cv2.minAreaRect(segment7[0])
            rect7 = cv2.boxPoints(rect7)
            rect7 = np.int0(rect7) 

            # Show and save the contours and points
            src_img = cv2.cvtColor(org_imgray, cv2.COLOR_BGR2BGRA)
            segmentation_img = cv2.cvtColor(imgray, cv2.COLOR_BGR2BGRA)
            # cv2.drawContours(segmentation_img, boundary, 0, colors['WHITE'], 1)      # set thickness of contour to green and thickness 2 so minimal impact on quantification

            cv2.drawContours(segmentation_img, segment1, 0, colors['RED'], -1)
            cv2.drawContours(segmentation_img, segment2, 0, colors['ORANGE'], -1) 
            cv2.drawContours(segmentation_img, segment3, 0, colors['TEAL'], -1)
            cv2.drawContours(segmentation_img, segment5, 0, colors['GREEN'], -1)
            cv2.drawContours(segmentation_img, segment6, 0, colors['BLUE'], -1) 
            cv2.drawContours(segmentation_img, segment7, 0, colors['VIOLET'], -1)

            cv2.circle(segmentation_img, (int(apex[0]), int(apex[1])), 1, colors['GREEN'])
            cv2.circle(segmentation_img, (int(centers[count][0][0]), int(centers[count][0][1])), 1, colors['WHITE'])
            cv2.circle(segmentation_img, (int(centers[count][1][0]), int(centers[count][1][1])), 1, colors['RED'])  
            cv2.circle(segmentation_img, (int(centers[count][2][0]), int(centers[count][2][1])), 1, colors['RED'])  
            cv2.circle(segmentation_img, (int(centers[count][3][0]), int(centers[count][3][1])), 1, colors['RED'])  
            cv2.circle(segmentation_img, (int(centers[count][4][0]), int(centers[count][4][1])), 1, colors['RED'])  
            cv2.circle(segmentation_img, (int(centers[count][5][0]), int(centers[count][5][1])), 1, colors['WHITE'])

            # cv2.drawContours(segmentation_img, [rect1], 0, colors["GREEN"], 1)
            # cv2.drawContours(segmentation_img, [rect2], 0, colors["GREEN"], 1)
            # cv2.drawContours(segmentation_img, [rect3], 0, colors["GREEN"], 1)
            # cv2.drawContours(segmentation_img, [rect5], 0, colors["GREEN"], 1)
            # cv2.drawContours(segmentation_img, [rect6], 0, colors["GREEN"], 1)
            # cv2.drawContours(segmentation_img, [rect7], 0, colors["GREEN"], 1)

            final_img = cv2.addWeighted(src_img, 1, segmentation_img, 0.5, 0.0)   
        
            # cv2.imshow('contours', img)
            # cv2.waitKey(0)
            if dir == ldata_dir:
                cv2.imwrite(labelsTraining_folder + "/" + os.path.splitext(image)[0] + ".png", final_img)
            else:
                cv2.imwrite(labelsTest_folder + "/" + os.path.splitext(image)[0] + ".png", final_img)
            # cv2.destroyAllWindows()

            if count%2 == 1:
                im_index.append(os.path.splitext(image)[0].split("_", 1)[0])

            count += 1
    # log_text(str(centers) + "\n", "centers.txt")
    # log_text(str(thickness) + "\n", "thickness.txt")
    print("Center and thickness of all segments acquired.")
    elapsed = time.time() - start
    log_text('Elapsed time: {:.0f}m {:.2f}s\n'.format(elapsed // 60, elapsed % 60), "contours directory.txt")
    return centers, thickness, im_index

def measure_wallMotion(centers, im_index):
    # Find the distance between 2 frames: 𝑑=√(〖(𝑥_𝐸𝑆−𝑥_𝐸𝐷)〗^2  +〖(𝑦_𝐸𝑆−𝑦_𝐸𝐷)〗^2 )
    # MI: unsymmetrical LV wall motion towards LV cavity and reduced myocardial thickening
    start = time.time()
    wallMotion = {}
    numTrainingImages = len(ldata_dir)
    numTestImages = len(ltestdata_dir)
    numValidImages = len(lvaliddata_dir)
    index = 0
    for count in range(numTrainingImages + numTestImages + numValidImages):
        if (count % 2) == 1:
            dist1 = math.dist(centers[count][0], centers[count-1][0])
            dist2 = math.dist(centers[count][1], centers[count-1][1])
            dist3 = math.dist(centers[count][2], centers[count-1][2])
            dist5 = math.dist(centers[count][3], centers[count-1][3])
            dist6 = math.dist(centers[count][4], centers[count-1][4])
            dist7 = math.dist(centers[count][5], centers[count-1][5])
            wallMotion[im_index[index]] = [dist1, dist2, dist3, dist5, dist6, dist7]
            index += 1

    log_text(str(wallMotion) + "\n", "wallMotion.txt")
    print("Wall motion measured for all segments.")
    elapsed = time.time() - start
    log_text('Elapsed time: {:.0f}m {:.2f}s\n'.format(elapsed // 60, elapsed % 60), "wallMotion.txt")
    return wallMotion

def measure_myocardialThickening(thickness, im_index):
    # find the difference between 2 frames: Δ𝑡=𝑡_𝐸𝑆  −𝑡_𝐸𝐷
    start = time.time()
    myocardialThickening = {}
    numTrainingImages = len(ldata_dir)
    numTestImages = len(ltestdata_dir)
    numValidImages = len(lvaliddata_dir)
    index = 0
    for count in range(numTrainingImages + numTestImages + numValidImages):
        if (count % 2) == 1:
            thick1 = thickness[count][0] - thickness[count-1][0]
            thick2 = thickness[count][1] - thickness[count-1][1]
            thick3 = thickness[count][2] - thickness[count-1][2]
            thick5 = thickness[count][3] - thickness[count-1][3]
            thick6 = thickness[count][4] - thickness[count-1][4]
            thick7 = thickness[count][5] - thickness[count-1][5]
            myocardialThickening[im_index[index]] = [thick1, thick2, thick3, thick5, thick6, thick7]
            index += 1

    log_text(str(myocardialThickening) + "\n", "myocardialThickening.txt")
    print("Myocardial thickening measured for all segments.")
    elapsed = time.time() - start
    log_text('Elapsed time: {:.0f}m {:.2f}s\n'.format(elapsed // 60, elapsed % 60), "myocardialThickening.txt")
    return myocardialThickening