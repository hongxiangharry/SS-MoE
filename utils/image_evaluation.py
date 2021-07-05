import csv
import sys
import os
import numpy as np
from skimage.measure import compare_ssim as ssim
import nibabel as nib
from skimage.measure import compare_psnr as psnr
from utils.ioutils import read_dataset, read_result_volume

def interp_evaluation(gen_conf, test_conf, case_name = 1):
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_info['sparse_scale'] = [1, 1, 1]
    dataset_info['in_postfix'] = dataset_info['interp_postfix']

    im_interp, im_gt = read_dataset(gen_conf, test_conf, 'test') # shape : (subject, modality, mri_shape)
    compare_images_and_get_stats(gen_conf, test_conf, im_gt, im_interp, case_name, save_filename='interp')
    return True

'''
    - read ground truth
    - read reconstructed image
    - compare image and get stats
    - save statistics
    - compare difference map
'''
def image_evaluation(gen_conf, test_conf, case_name = 1):
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    subject_lib = dataset_info['test_subjects']
    sparse_scale = dataset_info['sparse_scale']

    num_modalities = dataset_info['modalities']
    # case_name = case_name // num_modalities

    # load the ground truth image and result
    if sparse_scale == [1, 1, 1]:
        im_interp, im_gt = read_dataset(gen_conf, test_conf, 'test') # shape : (subject, modality, mri_shape)
    else:
        _, im_gt = read_dataset(gen_conf, test_conf, 'test')  # shape : (subject, modality, mri_shape)
    im_recon = read_result_volume(gen_conf, test_conf, case_name) # shape : (subject, modality, mri_shape)

    # compare image and get stats
    compare_images_and_get_stats(gen_conf, test_conf, im_gt, im_recon, case_name, save_filename='recon')
    if sparse_scale == [1, 1, 1]:
        compare_images_and_get_stats(gen_conf, test_conf, im_gt, im_interp, case_name, save_filename='interp')

def compare_images_and_get_stats(gen_conf, test_conf, im_gt, im_recon, case_name = 1, save_filename='ind'):
    num_volumes = im_gt.shape[0]
    modalities = im_gt.shape[1]
    # get stats with mask
    # mask = np.zeros(im_gt.shape[2:], dtype=int) == 0 # shape: mri_shape
    mask = (im_gt[0, 0] != 0)
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    subject_lib = dataset_info['test_subjects']
    modality_categories = dataset_info['modality_categories']
    # save_stats_dir = gen_conf['evaluation_path']
    # csv_file = os.path.join(save_stats_dir, save_filename)
    csv_file = generate_output_filename(
        gen_conf['evaluation_path'],
        test_conf['dataset'],
        'stat-'+save_filename+'-c'+str(case_name),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        'cvs')

    csv_folderpath = os.path.dirname(csv_file)
    if not os.path.isdir(csv_folderpath) :
        os.makedirs(csv_folderpath)
    headers = ['subject', 'modality', 'RMSE(whole)', 'Median(whole)', 'PSNR(whole)', 'MSSIM(whole)']

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            # compare image and get stats
            m, m2, p, s = _compare_images_and_get_stats(im_gt[img_idx, mod_idx],
                                                        im_recon[img_idx, mod_idx],
                                                        mask,
                                                        "whole: image no: {}, modality no: {}".format(img_idx, mod_idx))
            # save statistics
            stats = [m, m2, p, s]
            # Save the stats to a CSV file:
            save_stats(csv_file, subject_lib[img_idx], modality_categories[mod_idx], headers, stats)


def save_stats(csv_file, subject, modality, headers, stats):
    """
    Args:
        csv_file (str) : the whole path to the csv file
        subject (str): subject ID
        modality (str): modality name
        headers (list): list of metrics e.g. ['subject name', 'rmse ', 'median', 'psnr', 'mssim']
        stats (list): the errors for the corresponding subject e.g [1,2,3,4]

    """
    # if csv file exists, just update with the new entries:
    assert len(headers) == len([subject] + [modality] + stats)

    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            r = csv.reader(f)
            rows = list(r)
            rows_new = []
            new_row_flag = False ## if subject and modality is not existed in csv
            # copy the old table and update if necessary:
            for row in rows:
                if row[0] == subject and row[1] == modality: # update for the corresponding subject and modality
                    rows_new.append([subject]+[modality]+stats)
                    new_row_flag = True
                else:
                    rows_new.append(row)

            # add the new entry if it does not exist in the old table.
            # if not([subject, modality] in [[row for row in rows_new][0:2]]):
            if new_row_flag == False:
                rows_new.append([subject] + [modality]+ stats)
    else:
        rows_new = [headers, [subject]+ [modality]+stats]

    # save it to a csv file:
    with open(csv_file, 'w') as g:
        w = csv.writer(g)
        for row in rows_new:
            w.writerow(row)

def _compare_images_and_get_stats(img_gt, img_est, mask, name=''):
    """Compute RMSE, PSNR, MSSIM:
    Args:
         img_gt: (3D numpy array )
         ground truth volume
         img_est: (3D numpy array) predicted volume
         mask: (3D array) the mask whose the tissue voxles
         are labelled as 1 and the rest as 0
     Returns:
         m : RMSE
         m2: median of voxelwise RMSE
         p: PSNR
         s: MSSIM
     """
    blockPrint()
    m = compute_rmse(img_gt, img_est, mask)
    m2= compute_rmse_median(img_gt, img_est, mask)
    p = compute_psnr(img_gt, img_est, mask)
    s = compute_mssim(img_gt, img_est, mask)
    enablePrint()

    print("Errors (%s)"
          "\nRMSE: %.10f \nMedian: %.10f "
          "\nPSNR: %.6f \nSSIM: %.6f" % (name, m, m2, p, s))

    return m, m2, p, s


def compute_rmse(img1, img2, mask):
    if img1.shape != img2.shape:
        print("shape of img1 and img2: %s and %s" % (img1.shape, img2.shape))
        raise ValueError("the size of img 1 and img 2 do not match")
    mse = np.sum(((img1-img2)**2)*mask) \
          /(mask.sum())
    return np.sqrt(mse)

def compute_rmse_median(img1, img2, mask):
    if img1.shape != img2.shape:
        print("shape of img1 and img2: %s and %s" % (img1.shape, img2.shape))
        raise ValueError("the size of img 1 and img 2 do not match")

    # compute the voxel-wise average error:
    rmse_vol = np.sqrt((img1 - img2) ** 2 * mask)

    return np.median(rmse_vol[rmse_vol!=0])


def compute_mssim(img1, img2, mask, volume=False):
    if img1.shape != img2.shape:
        print("shape of img1 and img2: %s and %s" % (img1.shape, img2.shape))
        raise ValueError("the size of img 1 and img 2 do not match")
    img1=img1*mask
    img2=img2*mask

    m, S = ssim(img1,img2,
                dynamic_range=np.max(img1[mask])-np.min(img1[mask]),
                gaussian_weights=True,
                sigma= 2.5, #1.5,
                use_sample_covariance=False,
                full=True,
                multichannel=True)
    if volume:
        return S * mask
    else:
        mssim = np.sum(S * mask) / (mask.sum())
        return mssim


def compute_psnr(img1, img2, mask):
    """ Compute PSNR
    Arg:
        img1: ground truth image
        img2: test image
    """
    if img1.shape != img2.shape:
        print("shape of img1 and img2: %s and %s" % (img1.shape, img2.shape))
        raise ValueError("the size of img 1 and img 2 do not match")
    img1 = img1 * mask
    img2 = img2 * mask

    true_min, true_max = np.min(img1[mask]), np.max(img1)

    if true_min >= 0:
        # most common case (255 for uint8, 1 for float)
        dynamic_range = true_max
    else:
        dynamic_range = true_max - true_min

    rmse = compute_rmse(img1, img2, mask)
    return 10 * np.log10((dynamic_range ** 2) / (rmse**2))

def compute_differencemaps_t1t2(img_gt, img_est, mask, outputfile, no_channels,
                           save_as_ijk=True, gt_dir=None, gt_header=None, category=None):

    # Compute the L2 deviation and SSIM:
    rmse_volume = np.sqrt(((img_gt - img_est) ** 2)* mask[..., np.newaxis])
    blockPrint()
    ssim_volume = compute_mssim(img_gt, img_est, mask, volume=True)
    enablePrint()

    # Save the error maps:
    save_dir, file_name = os.path.split(outputfile)
    header, ext = os.path.splitext(file_name)

    for k in range(no_channels):
        if not (save_as_ijk):
            print("Fetching affine transform and header from GT.")
            if no_channels > 7:
                gt_file = gt_header + '%02i.nii' % (k + 1,)
                dt_gt = nib.load(os.path.join(gt_dir, gt_file))
            else:
                dt_gt = nib.load(os.path.join(gt_dir, gt_header + str(k + 1) + '.nii'))

            affine = dt_gt.get_affine()  # fetch its affine transfomation
            nii_header = dt_gt.get_header()  # fetch its header
            img_1 = nib.Nifti1Image(rmse_volume[:, :, :, k], affine=affine, header=nii_header)
            img_2 = nib.Nifti1Image(ssim_volume[:, :, :, k], affine=affine, header=nii_header)
        else:
            img_1 = nib.Nifti1Image(rmse_volume[:, :, :, k], np.eye(4))
            img_2 = nib.Nifti1Image(ssim_volume[:, :, :, k], np.eye(4))

        print('... saving the error (RMSE) and SSIM map for ' + str(k + 1) + ' th T1/T2 element')
        nib.save(img_1, os.path.join(save_dir, '_error_' + header + '.nii')) ## todo: no enough!
        nib.save(img_2, os.path.join(save_dir, '_ssim_' + header + '.nii'))  ## todo: no enough!


####### Misc ######

# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def generate_output_filename(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension):
    file_pattern = '{}/{}/{}-{}-{}-{}-{}.{}'
    return file_pattern.format(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension)