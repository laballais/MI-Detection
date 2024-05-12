# path below is for location of dataset of nifti files
mri_path = "../datasets/ACDC_dataset/diagnosis/training/"

# path below is location of where to store images from dataset
corrected_img = "../datasets/ACDC_dataset/diagnosis/correctedImg/"
corrected_contour = "../datasets/ACDC_dataset/diagnosis/correctedContour/"

# path below is location of where to store postprocessed images and labels
processedImg_path = "../datasets/mri/Data/"
processedContour_path = "../datasets/mri/Contour/"

# paths below are for the images and groundtruth
partial_paths = ["../datasets/mri/training/images/",
             "../datasets/mri/training/groundtruth/",
             "../datasets/mri/testing/images/",
             "../datasets/mri/testing/groundtruth/",
             "../datasets/mri/validating/images/",
             "../datasets/mri/validating/groundtruth/"]

# paths below are for the masks generated by the U-Net model
training_mask_path = "./model/training_masks/"
validating_mask_path = "./model/validating_masks/"
testing_mask_path = "./model/testing_masks/"