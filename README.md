# smpl_from_point_clouds
1. Generate Calibration cameras with `MC-Calib/record_dataset`
2. Run `calibrate` to calibrate the cameras extrinsics
3. Run `MC-Calib/viewer` to produce the n point clouds
4. Run `preprocess.py` to run icp + filtering on the point clouds
5. Run `fitting_smpl` to fit the SMPL model to the scan, adjust `config.json` to change the fitting parameters