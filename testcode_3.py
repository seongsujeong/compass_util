'''
Description:
To run extract_slc_coord_cr_stack_parallel on aurora
'''


import compass_util as cu
import json

path_stack_no_correction = '/home/jeong/scratch/CSLC/stack_processing_Rosamond/output_old/output_s1_cslc_no_correction/t064_135523_iw2'
path_json_no_correction = '/home/jeong/scratch/CSLC/stack_processing_Rosamond/output_old/output_s1_cslc_no_correction.json'
path_stack_with_correction = '/home/jeong/scratch/CSLC/stack_processing_Rosamond/output_old/output_s1_cslc_with_correction_rev4/t064_135523_iw2'
path_json_with_correction = '/home/jeong/scratch/CSLC/stack_processing_Rosamond/output_old/output_s1_cslc_with_correction_rev4.json'

latlon_cr = [34.80549368, -118.070803] # CR05

dict_1 = cu.extract_slc_coord_cr_stack_parallel(path_stack_no_correction, latlon_cr, 128, 32, True, 8)
with open(path_json_no_correction, 'w+') as jout1:
    json.dump(dict_1, jout1)

dict_2 = cu.extract_slc_coord_cr_stack_parallel(path_stack_with_correction, latlon_cr, 128, 32, True, 8)
with open(path_json_no_correction, 'w+') as jout2:
    json.dump(dict_2, jout2)


