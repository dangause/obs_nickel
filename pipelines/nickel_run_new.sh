#!/usr/bin/env bash

# bad exposures from your log:
BAD="" #"1056,1047,1043,1052,1032,1033"

########## ABSOLUTE PATHS (edit if needed) ##########
REPO="/Users/dangause/Desktop/lick/lsst/data/nickel/062424"
RAWDIR="/Users/dangause/Desktop/lick/data/062424/raw"
OBS_NICKEL="/Users/dangause/Desktop/lick/lsst/lsst_stack/stack/obs_nickel"
REFCAT_REPO="/Users/dangause/Desktop/lick/lsst/lsst_stack/stack/refcats"

########## BASIC CONFIG ##########
INSTRUMENT="lsst.obs.nickel.Nickel"
RUN="Nickel/raw/all"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

echo "=== Nickel pipeline starting @ $TS ==="

########## CREATE & REGISTER ##########
if [ ! -f "$REPO/butler.yaml" ]; then
  butler create "$REPO"
fi
butler register-instrument "$REPO" "$INSTRUMENT" || true
butler ingest-raws "$REPO" "$RAWDIR" --transfer symlink --output-run "$RUN"
butler define-visits "$REPO" Nickel

########## CURATED ##########
CURATED="Nickel/run/curated/$TS"
butler write-curated-calibrations "$REPO" Nickel "$RUN" --collection "$CURATED"

########## BIAS ##########
CP_RUN_BIAS="Nickel/run/cp_bias/$TS"
pipetask run \
  -b "$REPO" \
  -i "$CURATED","$RUN" \
  -o "$CP_RUN_BIAS" \
  -p "$CP_PIPE_DIR/pipelines/_ingredients/cpBias.yaml" \
  -d "instrument='Nickel' AND exposure.observation_type='bias'" \
  --register-dataset-types

# certify bias broadly
butler certify-calibrations "$REPO" "$CP_RUN_BIAS" "$CURATED" bias \
  --begin-date 2020-01-01 --end-date 2030-01-01

########## FLATS ##########
CP_RUN_FLAT="Nickel/run/cp_flat/$TS"
pipetask run \
  -b "$REPO" \
  -i "$CURATED","$RUN","$CP_RUN_BIAS" \
  -o "$CP_RUN_FLAT" \
  -p "$CP_PIPE_DIR/pipelines/_ingredients/cpFlat.yaml" \
  -c cpFlatIsr:doDark=False \
  -d "instrument='Nickel' AND exposure.observation_type='flat'" \
  --register-dataset-types

########## DEFECTS (from flats; module has no detector args) ##########
DEF_TS="$(date -u +%Y%m%dT%H%M%SZ)"
DEFECTS_RUN="Nickel/calib/defects/$DEF_TS"
QA_DIR="$OBS_NICKEL/scripts/defects/qa_$DEF_TS"

python "$OBS_NICKEL"/scripts/defects/make_defects_from_flats.py \
  --repo "$REPO" \
  --collection "$CP_RUN_FLAT" \
  --register \
  --ingest \
  --defects-run "$DEFECTS_RUN" \
  --plot \
  --qa-dir "$QA_DIR"

# === AUTO-PICK LATEST DEFECTS RUN ===
DEFECTS_RUN="$(butler query-collections "$REPO" | awk '/^Nickel\/calib\/defects\//{print $1}' | tail -n1)"
echo "Using latest defects run: $DEFECTS_RUN"

# point current -> latest defects
butler collection-chain "$REPO" Nickel/calib/defects/current "$DEFECTS_RUN" --mode redefine

########## UNIFIED CALIB CHAIN ##########
CALIB_CHAIN="Nickel/calib/current"
butler collection-chain "$REPO" "$CALIB_CHAIN" \
  "$CURATED" "$CP_RUN_BIAS" "$CP_RUN_FLAT" Nickel/calib/defects/current \
  --mode redefine

########## REFCATS (run from your refcat repo; original commands) ##########
cd "$REFCAT_REPO"

# Gaia DR3
convertReferenceCatalog \
  data/gaia-refcat/ \
  scripts/gaia_dr3_config.py \
  ./data/gaia_dr3_all_cones/gaia_dr3_all_cones.csv \
  &> convert-gaia.log

butler register-dataset-type "$REPO" gaia_dr3_20250728 SimpleCatalog htm7

butler ingest-files \
  -t direct \
  "$REPO" \
  gaia_dr3_20250728 \
  refcats/gaia_dr3_20250728 \
  data/gaia-refcat/filename_to_htm.ecsv

butler collection-chain \
  "$REPO" \
  --mode extend \
  refcats \
  refcats/gaia_dr3_20250728

# PS1 DR2
convertReferenceCatalog \
  data/ps1-refcat/ \
  scripts/ps1_config.py \
  ./data/ps1_all_cones/merged_ps1_cones.csv \
  &> convert-ps1.log

butler register-dataset-type "$REPO" panstarrs1_dr2_20250730 SimpleCatalog htm7

butler ingest-files \
  -t direct \
  "$REPO" \
  panstarrs1_dr2_20250730 \
  refcats/panstarrs1_dr2_20250730 \
  data/ps1-refcat/filename_to_htm.ecsv

butler collection-chain \
  "$REPO" \
  --mode extend \
  refcats \
  refcats/panstarrs1_dr2_20250730

########## SCIENCE PROCESSING ##########
cd "$OBS_NICKEL"
PIPE="$OBS_NICKEL/pipelines/ProcessCcd.yaml"
PROCESS_CCD_RUN="Nickel/run/processCcd/$(date +%Y%m%dT%H%M%S)"

# quick sanity
butler query-collections "$REPO" | grep -E 'Nickel/calib/(current|defects/current)' || true

pipetask run \
  -b "$REPO" \
  -i "$RUN","$CALIB_CHAIN","refcats" \
  -o "$PROCESS_CCD_RUN" \
  -p "$PIPE#processCcd" \
  -d "instrument='Nickel' AND exposure.observation_type='science'" \
  --register-dataset-types \
  2>&1 | tee logs/processCcd_$TS.log
  #  -d "instrument='Nickel' AND exposure.observation_type='science' AND NOT (exposure IN (${BAD}))" \

echo "=== Done ==="
echo "Curated:     $CURATED"
echo "CP Bias:     $CP_RUN_BIAS"
echo "CP Flat:     $CP_RUN_FLAT"
echo "Defects run: $DEFECTS_RUN"
echo "Calib chain: $CALIB_CHAIN"
echo "Science run: $PROCESS_CCD_RUN"
