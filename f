# apply lungmask to the original non-transformed data dir
 1. lungmask "/home-local/adi/scripts/pycharm_project_lobeseg/lobe_seg_downsampled/data_dir" outdir
 manually used lungmask currently
# transform everthing now
 2. writing a new python file for this
  python3 traintransformeverything.py "/home-local/adi/scripts/pycharm_project_lobeseg/lobe_seg_downsampled/data_dir" "/home-local/adi/scripts/pycharm_project_lobeseg/lobe_seg_downsampled/label_dir" "/home-local/adi/scripts/pycharm_project_lobeseg/lobe_seg_downsampled/mask_dir" "/home-local/adi/scripts/pycharm_project_lobeseg/lobe_seg_downsampled/data_out_dir" "/home-local/adi/scripts/pycharm_project_lobeseg/lobe_seg_downsampled/label_out_dir" "/home-local/adi/scripts/pycharm_project_lobeseg/lobe_seg_downsampled/mask_out_dir"
 3. visualize the transforms image spaces using binarymask.py file
 4. Also the above file can be used to create four boxes
 5. Now we stitch the infered boxes , using inference_with_boxes.py file!
 6. Now apply post pred transforms


