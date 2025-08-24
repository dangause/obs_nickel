REPO="/Users/dangause/Desktop/lick/lsst/data/nickel/062424"
RUN="Nickel/raw/all"                          # your existing raw run
CALIB_CHAIN="Nickel/calib/current"            # already chained curated/bias/flat/defects
PIPE="/Users/dangause/Desktop/lick/lsst/lsst_stack/stack/obs_nickel/pipelines/ProcessCcd.yaml"
BAD="1032,1051,1052"
PROCESS_CCD_RUN="Nickel/run/processCcd/$(date -u +%Y%m%dT%H%M%SZ)"

pipetask run \
  -b "$REPO" \
  -i "$RUN","$CALIB_CHAIN","refcats" \
  -o "$PROCESS_CCD_RUN" \
  -p "$PIPE#processCcd" \
  -C calibrateImage:configs/apcorr_overrides.py \
#   -C calibrateImage:configs/astrometry_overrides.py \
  -d "instrument='Nickel' AND exposure.observation_type='science' AND NOT (exposure IN (${BAD}))" \
  -j 8  --register-dataset-types \
  2>&1 | tee logs/processCcd_only_$TS.log