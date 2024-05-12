from audioop import reverse
import numpy as np
import cv2
import os
import math
import time
import polylabel

from midetection.Utils import mri_config
from torch import greater, outer
from midetection.Utils.utilities import log_text, create_directory, colors
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

l_data = mri_config.partial_paths[1]                              # labels for training set
input_data = mri_config.training_mask_path
l_testdata = mri_config.partial_paths[3]                          # labels for test set
testInput_data = mri_config.testing_mask_path
l_validdata = mri_config.partial_paths[5]                         # labels for test set
validInput_data = mri_config.validating_mask_path 

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

# Returns the 3 segments on either side of the left ventricle
def get_segmentsFromContour(contour, referencePoints):
    segment1 = []
    segment2 = []
    segment3 = []
    for point in contour[0]:
        if point[0][0] > referencePoints[1][0]:
            segment1.append([[point[0][0],point[0][1]]])
        elif point[0][0] > referencePoints[2][0]:
            segment2.append([[point[0][0],point[0][1]]])
        else:            
            segment3.append([[point[0][0],point[0][1]]])

    # # TO DO: Connect segment 2 to ends of segment 1 and segment 3
    # # print("segment1: " + str(segment1))
    # # print("segment2: " + str(segment2))
    # # print("segment3: " + str(segment3))
    # # Find 2 leftmost points of segment 1: smallest Xs. add to middle of segment 2
    # if len(segment1) != 0:
    #     segment2Copy = segment2.copy()
    #     for i,point in enumerate(segment2Copy):
    #         if point[0][0] > segment1[0][0][0]:
    #             segment2.insert(i, [[segment1[0][0][0], segment1[0][0][1]]])
    #             segment2.insert(i+2, [[segment1[0][-1][0], segment1[0][-1][1]]])
    #         break
    # # Find 2 rightmost points of segment 3: largest Xs. add to start and end of segment 2
    # if len(segment3) != 0:
    #     rightmost = 0
    #     for i,point in enumerate(segment3):
    #         if point[0][0] > segment3[rightmost][0][0]:
    #             rightmost = i
    #     segment2.insert(0, [[segment3[rightmost][0][0], segment3[rightmost][0][1]]])
    #     if rightmost+1 < len(segment3):
    #         segment2.append([[segment3[rightmost+1][0][0], segment3[rightmost+1][0][1]]])

    # # segment1 = (np.array(segment1), )
    # # segment2 = (np.array(segment2), )
    # # segment3 = (np.array(segment3), )

    return segment1, segment2, segment3

# Finds the center of mass of each segment using moments --> use for irregular shapes such as segment 3
def get_centerOfSegment(segment):
    # Code below gets centroid of polygon, but may lie outside of shape if non-convex
    M = cv2.moments(segment[0])
    centerSeg = (M['m10'] / (M['m00'] + 1e-6), M['m01'] / (M['m00'] + 1e-6))
    # print(centerSeg)

    # Code below guarantees point inside the polygon but, not may not be centered
    setOfPoints = []
    for point in segment[0]:
        setOfPoints.append([point[0][0], point[0][1]])
    if len(setOfPoints) != 0:
        centerSeg = polylabel.polylabel([setOfPoints])
    # print(centerSeg)

    # Find a point nearest to centroid. Rotate this point by 180deg around centroid and find the nearest neighbor. Find midpoint.
    # setOfPoints = []
    # for point in segment[0]:
    #     setOfPoints.append([point[0][0]])
    # setOfPoints = np.array(setOfPoints)
    # knn = NearestNeighbors(n_neighbors=1, p=2)
    # knn.fit(setOfPoints)
    # centroidX = np.array([[centerSeg[0]]])
    # nearestXIndices = knn.kneighbors(centroidX, return_distance=False)
    # pt1 = (segment[0][nearestXIndices[0][0]][0][0], segment[0][nearestXIndices[0][0]][0][1])

    # theta = math.pi
    # xr = math.cos(theta)*(pt1[0]-centerSeg[0]) - math.sin(theta)*(pt1[1]-centerSeg[1]) + centerSeg[0]
    # yr = math.sin(theta)*(pt1[0]-centerSeg[0]) + math.cos(theta)*(pt1[1]-centerSeg[1]) + centerSeg[1]
    
    # setOfPoints = []
    # for point in segment[0]:
    #     setOfPoints.append([point[0][0], point[0][1]])
    # setOfPoints = np.array(setOfPoints)
    # knn = NearestNeighbors(n_neighbors=1, p=2)
    # knn.fit(setOfPoints)
    # rotatedPoint = np.array([[xr, yr]])
    # nearestIndices2 = knn.kneighbors(rotatedPoint, return_distance=False)   
    # pt2 = (segment[0][nearestIndices2[0][0]][0][0], segment[0][nearestIndices2[0][0]][0][1])

    # centerSeg = find_pixelInBetween(pt1, pt2, 1/2)
    # print(pt1)
    # print((xr,yr))
    # print(pt2)
    # print(centerSeg)

    return centerSeg

# Finds the center of all segments
def get_centerOfAllSegments(segment1, segment2, segment3, segment4, segment5, segment6):
    log_text("      Finding the center of each segment.\n", "contours directory.txt")
    centerSeg1 = get_centerOfSegment(segment1)
    centerSeg2 = get_centerOfSegment(segment2)
    centerSeg3 = get_centerOfSegment(segment3)
    centerSeg4 = get_centerOfSegment(segment4)
    centerSeg5 = get_centerOfSegment(segment5)
    # centerSeg5 = (centerSeg5[0], centerSeg5[1] + 1)
    centerSeg6 = get_centerOfSegment(segment6)

    return [centerSeg1, centerSeg2, centerSeg3, centerSeg4, centerSeg5, centerSeg6]

# Finds the thickness of a segment ---> use for segments of irregular shape
def get_thicknessOfSegment(segment, centers):
    # Get 1 point nearest to segment center X coordinate
    setOfPoints = []
    for point in segment[0]:
        setOfPoints.append([point[0][0]])
    setOfPoints = np.array(setOfPoints)
    if len(setOfPoints) != 0:
        knn = NearestNeighbors(n_neighbors=1, p=2)
        knn.fit(setOfPoints)  
        center = np.array([[centers[0]]])
        nearestIndices = knn.kneighbors(center, return_distance=False)
        pt1 = (segment[0][nearestIndices[0][0]][0][0], segment[0][nearestIndices[0][0]][0][1])

        # Rotate pt1 by 180 deg about centers and get the nearest neighbor
        theta = math.pi
        xr = math.cos(theta)*(pt1[0]-centers[0]) - math.sin(theta)*(pt1[1]-centers[1]) + centers[0]
        yr = math.sin(theta)*(pt1[0]-centers[0]) + math.cos(theta)*(pt1[1]-centers[1]) + centers[1]
        
        setOfPoints = []
        for point in segment[0]:
            setOfPoints.append([point[0][0], point[0][1]])
        setOfPoints = np.array(setOfPoints)
        knn = NearestNeighbors(n_neighbors=1, p=2)
        knn.fit(setOfPoints)
        rotatedPoint = np.array([[xr, yr]])
        nearestIndices2 = knn.kneighbors(rotatedPoint, return_distance=False)   
        pt2 = (segment[0][nearestIndices2[0][0]][0][0], segment[0][nearestIndices2[0][0]][0][1])

        # Find thickeness of segment by computing the euclidean distance between pt1 and pt2
        # print(pt1)
        # print(pt2)
        thickness = distance.euclidean(pt1, pt2)
    else:
        thickness = 0

    return thickness

# Finds the thickness of all segments of the left ventricle
def get_thicknessOfAllSegments(segment1, segment2, segment3, segment4, segment5, segment6, centers):
    log_text("      Measuring the thickness of each segment.\n", "contours directory.txt")
    thickness1 = get_thicknessOfSegment(segment1, centers[0])
    thickness2 = get_thicknessOfSegment(segment2, centers[1])
    thickness3 = get_thicknessOfSegment(segment3, centers[2])
    thickness4 = get_thicknessOfSegment(segment4, centers[3])
    thickness5 = get_thicknessOfSegment(segment5, centers[4])
    thickness6 = get_thicknessOfSegment(segment6, centers[5])
  
    # print(str([dim1, dim2, dim3, dim5, dim6, dim7]))
    return [thickness1, thickness2, thickness3, thickness4, thickness5, thickness6]

def get_centerAndThicknessOfAllSegments():
    start = time.time()
    centers = {}
    thickness = {}
    im_index = []
    count = 0
    log_text("Saving contours of labels.\n", "contours directory.txt")
    for dir in label_dir:
        for i, image in enumerate(dir):
            #  Get the boundary of the left ventricle
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
            contours = sorted(boundary, key=cv2.contourArea, reverse=True)   # get only the contour for the 2 largest area aka the LV           

            # Get center of contour of LV cavity
            (x,y), radius = cv2.minEnclosingCircle(contours[0])
            center = (int(x), int(y))
            radius = int(radius)

            # Separate the upper side and lower side pixels of the left ventricle
            log_text("      Dividing the contours into upper and lower contours.\n", "contours directory.txt")
            outerUpperContour = []
            outerLowerContour = []
            innerUpperContour = []
            innerLowerContour = []
            temp = []
                     
            for point in contours[0]:
                if point[0][1] > center[1]:                                            
                    outerLowerContour.append([[point[0][0],point[0][1]]])
                elif point[0][1] < center[1] and point[0][0] > center[0]:                                  
                    outerUpperContour.insert(0, [[point[0][0],point[0][1]]])
                elif point[0][1] < center[1] and point[0][0] < center[0]:                                  
                    temp.insert(0, [[point[0][0],point[0][1]]])
                elif point[0][1] < center[1] and point[0][0] == center[0]:                                  
                    outerUpperContour.append([[point[0][0],point[0][1]]])
            for pixel in temp:
                outerUpperContour.append(pixel)
            temp.clear()
            outerUpperContour = sorted(outerUpperContour, key = lambda k: [k[0][0], k[0][1]])
            outerLowerContour = sorted(outerLowerContour, key = lambda k: [k[0][0], k[0][1]])

            temp = []
            if (len(boundary) > 1):
                for point in contours[1]:
                    if point[0][1] > center[1]:                                            
                        innerLowerContour.append([[point[0][0],point[0][1]]])
                    elif point[0][1] < center[1] and point[0][0] > center[0]:                                            
                        innerUpperContour.insert(0, [[point[0][0],point[0][1]]])
                    elif point[0][1] < center[1] and point[0][0] < center[0]:                                  
                        temp.insert(0, [[point[0][0],point[0][1]]])
                    elif point[0][1] < center[1] and point[0][0] == center[0]:                                  
                        innerUpperContour.append([[point[0][0],point[0][1]]])
                for pixel in temp:
                    innerUpperContour.append(pixel)
                temp.clear()

                innerUpperContour = sorted(innerUpperContour, key = lambda k: [k[0][0], k[0][1]], reverse=True)
                innerLowerContour = sorted(innerLowerContour, key = lambda k: [k[0][0], k[0][1]], reverse=True)

            outerLowerContour = sorted(outerLowerContour, key = lambda k: [k[0][0], k[0][1]])

            # Find extreme left of lowercontour and add it to uppercontour
            # Find extreme right of lowercontour and add it to uppercontour
            outerExtLeft = (outerLowerContour[0][0][0], outerLowerContour[0][0][1])
            outerExtRight = (outerLowerContour[len(outerLowerContour) - 1][0][0], outerLowerContour[len(outerLowerContour) - 1][0][1])
            if [[outerExtLeft[0], outerExtLeft[1]]] not in outerUpperContour:
                outerUpperContour.insert(0, [[outerExtLeft[0], outerExtLeft[1]]])
            if [[outerExtRight[0], outerExtRight[1]]] not in outerUpperContour:
                outerUpperContour.append([[outerExtRight[0], outerExtRight[1]]])    
            
            if len(boundary) > 1 and len(innerLowerContour) > 1:
                innerExtRight = (innerLowerContour[0][0][0], innerLowerContour[0][0][1])
                innerExtLeft = (innerLowerContour[len(innerLowerContour) - 1][0][0], innerLowerContour[len(innerLowerContour) - 1][0][1])
                if [[innerExtLeft[0], innerExtLeft[1]]] not in innerUpperContour:
                    innerUpperContour.append([[innerExtLeft[0], innerExtLeft[1]]])
                if [[innerExtRight[0], innerExtRight[1]]] not in innerUpperContour:
                    innerUpperContour.insert(0, [[innerExtRight[0], innerExtRight[1]]])
            
            outerUpperContour = (np.array(outerUpperContour), )
            outerLowerContour = (np.array(outerLowerContour), )
            innerLowerContour = (np.array(innerLowerContour), )
            innerUpperContour = (np.array(innerUpperContour), )

            # Get the 7 segments of the left ventricle: 3 on upper, 3 on lower
            # CONCEPT:
            # Obtain the outer perimeter points which will be used to divide the circular myocardium into segments
            log_text("      Dividing the upper contour into different segments.\n", "contours directory.txt")

            # Get the outer boundary points which will be basis for dividing the contours into 6 segments
            outerPerimeter_x = []
            outerPerimeter_y = []

            for angle in range(0,360,60):
                x = (int)((center[0] + radius * math.cos(angle * math.pi / 180.0)))
                y = (int)((center[1] + radius * math.sin(angle * math.pi / 180.0)))
                outerPerimeter_x.append(x)
                outerPerimeter_y.append(y)

            ref0 = (outerPerimeter_x[0], outerPerimeter_y[0])
            ref60 = (outerPerimeter_x[1], outerPerimeter_y[1])
            ref120 = (outerPerimeter_x[2], outerPerimeter_y[2])
            ref240 = (outerPerimeter_x[4], outerPerimeter_y[4])
            ref300 = (outerPerimeter_x[5], outerPerimeter_y[5])
            ref360 = ref0

            segment1, segment2, segment3 = get_segmentsFromContour(outerUpperContour, [ref0, ref60, ref120])
            segment6, segment5, segment4 = get_segmentsFromContour(outerLowerContour, [ref360, ref300, ref240])

            if len(boundary) > 1:
                # Get the outer boundary points which will be basis for dividing the contours into 6 segments
                innerPerimeter_x = []
                innerPerimeter_y = []

                (x,y), radius = cv2.minEnclosingCircle(contours[1])
                center = (int(x), int(y))
                radius = int(radius)

                for angle in range(0,360,60):
                    x = (int)((center[0] + radius * math.cos(angle * math.pi / 180.0)))
                    y = (int)((center[1] + radius * math.sin(angle * math.pi / 180.0)))
                    innerPerimeter_x.append(x)
                    innerPerimeter_y.append(y)

                ref0 = (innerPerimeter_x[0], innerPerimeter_y[0])
                ref60 = (innerPerimeter_x[1], innerPerimeter_y[1])
                ref120 = (innerPerimeter_x[2], innerPerimeter_y[2])
                ref240 = (innerPerimeter_x[4], innerPerimeter_y[4])
                ref300 = (innerPerimeter_x[5], innerPerimeter_y[5])
                ref360 = ref0

                isegment1, isegment2, isegment3 = get_segmentsFromContour(innerUpperContour, [ref0, ref60, ref120])
                isegment6, isegment5, isegment4 = get_segmentsFromContour(innerLowerContour, [ref360, ref300, ref240])

                for point in isegment1:
                    segment1.append([[point[0][0], point[0][1]]])  
                for point in isegment2:
                    segment2.append([[point[0][0], point[0][1]]])
                for point in isegment3:
                    segment3.append([[point[0][0], point[0][1]]])
                for point in isegment4:
                    segment4.append([[point[0][0], point[0][1]]])
                for point in isegment5:
                    segment5.append([[point[0][0], point[0][1]]])
                for point in isegment6:
                    segment6.append([[point[0][0], point[0][1]]])

            # print(segment1)

            # TO DO: connect leftmost of seg 1/6 to rightmost of seg 2/5, leftmost of seg2/5 to rightmost of seg 3/4

            segment1 = (np.array(segment1), )
            segment2 = (np.array(segment2), )
            segment3 = (np.array(segment3), )
            segment4 = (np.array(segment4), )
            segment5 = (np.array(segment5), )
            segment6 = (np.array(segment6), )             

            # Find the center of each segment
            centers[count] = get_centerOfAllSegments(segment1, segment2, segment3, segment4, segment5, segment6)

            # Find the thickness of each segment
            thickness[count] = get_thicknessOfAllSegments(segment1, segment2, segment3, segment4, segment5, segment6, centers[count])

                    
            # Show and save the contours and points
            src_img = cv2.cvtColor(org_imgray, cv2.COLOR_BGR2BGRA)
            segmentation_img = cv2.cvtColor(imgray, cv2.COLOR_BGR2BGRA)
            # cv2.drawContours(segmentation_img, boundary, 0, colors['WHITE'], 1)      # set thickness of contour to green and thickness 2 so minimal impact on quantification

            # cv2.circle(segmentation_img, center, radius, colors['GREEN'],2) # ENCLOSING CIRCLE

            # cv2.drawContours(segmentation_img, contours, -1, colors['RED'], -1)
            # cv2.drawContours(segmentation_img, upperContour, 0, colors['ORANGE'], -1) 
            # cv2.drawContours(segmentation_img, lowerContour, 0, colors['TEAL'], -1)
            if len(segment1[0]) > 0:
                cv2.drawContours(segmentation_img, segment1, 0, colors['VIOLET'], -1)
                cv2.circle(segmentation_img, (int(centers[count][0][0]), int(centers[count][0][1])), 1, colors['RED'])
            if len(segment2[0]) > 0:
                cv2.drawContours(segmentation_img, segment2, 0, colors['RED'], -1) 
                cv2.circle(segmentation_img, (int(centers[count][1][0]), int(centers[count][1][1])), 1, colors['WHITE'])  
            if len(segment3[0]) > 0:
                cv2.drawContours(segmentation_img, segment3, 0, colors['ORANGE'], -1)
                cv2.circle(segmentation_img, (int(centers[count][2][0]), int(centers[count][2][1])), 1, colors['RED'])
            if len(segment4[0]) > 0:
                cv2.drawContours(segmentation_img, segment4, 0, colors['GREEN'], -1)
                cv2.circle(segmentation_img, (int(centers[count][3][0]), int(centers[count][3][1])), 1, colors['RED'])
            if len(segment5[0]) > 0:
                cv2.drawContours(segmentation_img, segment5, 0, colors['BLUE'], -1)
                cv2.circle(segmentation_img, (int(centers[count][4][0]), int(centers[count][4][1])), 1, colors['RED']) 
            if len(segment6[0]) > 0: 
                cv2.drawContours(segmentation_img, segment6, 0, colors['TEAL'], -1)
                cv2.circle(segmentation_img, (int(centers[count][5][0]), int(centers[count][5][1])), 1, colors['RED'])

            # cv2.circle(segmentation_img, (int(outerPerimeter_x[0]), int(outerPerimeter_y[0])), 1, colors['BLUE'])
            # cv2.circle(segmentation_img, (int(outerPerimeter_x[1]), int(outerPerimeter_y[1])), 1, colors['WHITE'])
            # cv2.circle(segmentation_img, (int(outerPerimeter_x[2]), int(outerPerimeter_y[2])), 1, colors['WHITE'])  
            # cv2.circle(segmentation_img, (int(outerPerimeter_x[3]), int(outerPerimeter_y[3])), 1, colors['WHITE'])  
            # cv2.circle(segmentation_img, (int(outerPerimeter_x[4]), int(outerPerimeter_y[4])), 1, colors['WHITE'])  
            # cv2.circle(segmentation_img, (int(outerPerimeter_x[5]), int(outerPerimeter_y[5])), 1, colors['WHITE'])  
            # cv2.circle(segmentation_img, (int(innerExtLeft[0]), int(innerExtLeft[1])), 1, colors['ORANGE'])
            # cv2.circle(segmentation_img, (int(outerExtLeft[0]), int(outerExtLeft[1])), 1, colors['VIOLET'])
              
            final_img = cv2.addWeighted(src_img, 1, segmentation_img, 0.5, 0.0)   
        
            # cv2.imshow('contours', img)
            # cv2.waitKey(0)
            if dir == ldata_dir:
                cv2.imwrite(labelsTraining_folder + "/" + os.path.splitext(image)[0] + ".png", final_img)
            else:
                cv2.imwrite(labelsTest_folder + "/" + os.path.splitext(image)[0] + ".png", final_img)
            # cv2.destroyAllWindows()

            if count%2 == 1:
                # im_index.append(os.path.splitext(image)[0].split("_", 1)[0])
                im_index.append(os.path.splitext(image)[0].split("_e", 1)[0])

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
            dist4 = math.dist(centers[count][3], centers[count-1][3])
            dist5 = math.dist(centers[count][4], centers[count-1][4])
            dist6 = math.dist(centers[count][5], centers[count-1][5])
            wallMotion[im_index[index]] = [dist1, dist2, dist3, dist4, dist5, dist6]
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
            thick4 = thickness[count][3] - thickness[count-1][3]
            thick5 = thickness[count][4] - thickness[count-1][4]
            thick6 = thickness[count][5] - thickness[count-1][5]
            myocardialThickening[im_index[index]] = [thick1, thick2, thick3, thick4, thick5, thick6]
            index += 1

    log_text(str(myocardialThickening) + "\n", "myocardialThickening.txt")
    print("Myocardial thickening measured for all segments.")
    elapsed = time.time() - start
    log_text('Elapsed time: {:.0f}m {:.2f}s\n'.format(elapsed // 60, elapsed % 60), "myocardialThickening.txt")
    return myocardialThickening