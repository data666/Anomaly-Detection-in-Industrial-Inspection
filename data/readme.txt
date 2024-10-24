Hi Team,


I have processed the data. You can see the detailed steps in my data_processing file. 
I resized the images to 128x128 to save memory and normalized the pixel values by dividing by 255. 
Then, I used OpenCV’s built-in function to reduce noise in the images. 
Finally, I labeled all the images and saved them as a .mat file. 
Because the file is quite large, I stored it in Google Drive. The link will be sent to you via Slack.
x_train contains 4283 images, and x_test contains 1071 images. 

Below is the mapping dictionary:
mapping_dict = {
    'defective_bottle': 0,
    'defect_free_bottle': 1,
    'defective_cable': 2,
    'defect_free_cable': 3,
    'defective_capsule': 4,
    'defect_free_capsule': 5,
    'defective_carpet': 6,
    'defect_free_carpet': 7,
    'defective_grid': 8,
    'defect_free_grid': 9,
    'defective_hazelnut': 10,
    'defect_free_hazelnut': 11,
    'defective_leather': 12,
    'defect_free_leather': 13,
    'defective_metal_nut': 14,
    'defect_free_metal_nut': 15,
    'defective_pill': 16,
    'defect_free_pill': 17,
    'defective_screw': 18,
    'defect_free_screw': 19,
    'defective_tile': 20,
    'defect_free_tile': 21,
    'defective_toothbrush': 22,
    'defect_free_toothbrush': 23,
    'defective_transistor': 24,
    'defect_free_transistor': 25,
    'defective_wood': 26,
    'defect_free_wood': 27,
    'defective_zipper': 28,
    'defect_free_zipper': 29
}
