'''
To testrun `def slc_to_amplitude_batch`
'''

import compass_util as cu
if __name__ == '__main__':
    dir_slc_in = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/LUT_CORRECTION_TEST_SITE/output_s1_cslc_all_correction/t064_135523_iw2'
    dir_amp_out = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/LUT_CORRECTION_TEST_SITE/amp_s1_cslc_all_correction'
    cu.slc_to_amplitude_batch(dir_slc_in, dir_amp_out, pol='VV', ncpu=4)

    dir_slc_in = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/LUT_CORRECTION_TEST_SITE/output_s1_cslc_no_correction/t064_135523_iw2'
    dir_amp_out = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/LUT_CORRECTION_TEST_SITE/amp_s1_cslc_no_correction'
    cu.slc_to_amplitude_batch(dir_slc_in, dir_amp_out, pol='VV', ncpu=4)
