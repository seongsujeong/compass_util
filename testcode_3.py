'''
Description:
To run extract_slc_coord_cr_stack_parallel on aurora
'''


import compass_util as cu

path_stack_no_correction = '/home/jeong/scratch/CSLC/stack_processing_Rosamond/output_old/output_s1_cslc_no_correction'
path_stack_with_correction = '/home/jeong/scratch/CSLC/stack_processing_Rosamond/output_old/output_s1_cslc_with_correction_rev4'
latlon_cr = latlon_cr = [34.80549368, -118.070803] # CR05
cu.extract_slc_coord_cr_stack_parallel(path_stack_no_correction, latlon_cr, 128, 32, True, 8)
cu.extract_slc_coord_cr_stack_parallel(path_stack_with_correction, latlon_cr, 128, 32, True, 8)

