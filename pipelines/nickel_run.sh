#!/usr/bin/env bash
##############################################################################
# nickel_run.sh  –  Lick-Nickel reduction (SciPipes 9.0, macOS)
##############################################################################
set -eu
( set -o pipefail 2>/dev/null ) && set -o pipefail

# ----- macOS: expose EUPS libs ---------------------------------------------
setup sphgeom >/dev/null 2>&1 || true
EUPS_LIBS=$(env | grep '_DIR=' | awk -F= '{print $2"/lib"}' | tr '\n' ':')
export DYLD_FALLBACK_LIBRARY_PATH="${CONDA_PREFIX}/lib:${EUPS_LIBS}${DYLD_FALLBACK_LIBRARY_PATH:-}"
# ---------------------------------------------------------------------------

# ----------------------------- USER PATHS ----------------------------------
RAWDIR="${HOME}/Desktop/lick/data/062424/raw"
REPO="${HOME}/Desktop/lick/lsst/data/nickel/062424"
INSTRUMENT="lsst.obs.nickel.Nickel"
TS=$(date -u +%Y%m%dT%H%M%SZ)

RUN_RAW="Nickel/raw/all"
CURATED="Nickel/run/curated/$TS"
CP_BIAS="Nickel/run/cp_bias/$TS"
CP_FLAT="Nickel/run/cp_flat/$TS"
PROC_CCD="Nickel/run/processCcd/$TS"
PIPE_DIR="${CP_PIPE_DIR:-$PWD/obs_nickel}"
# ---------------------------------------------------------------------------

##############################################################################
# 0.  Create repo & ingest raws (once)
##############################################################################
if [ ! -f "$REPO/butler.yaml" ]; then
    echo "▶ Creating Butler repo: $REPO"
    mkdir -p "$REPO"
    butler create "$REPO"
    butler register-instrument "$REPO" "$INSTRUMENT"
    butler ingest-raws "$REPO" "$RAWDIR" --transfer symlink \
           --output-run "$RUN_RAW" --processes 4
    butler define-visits "$REPO" Nickel
fi

##############################################################################
# 1.  Curated reference calibs
##############################################################################
if ! butler collection-exists "$REPO" "$CURATED" 2>/dev/null; then
    butler write-curated-calibrations "$REPO" Nickel "$RUN_RAW" \
           --collection "$CURATED"
fi

##############################################################################
# 2.  Bias masters
##############################################################################
echo "▶ Creating bias masters..."
pipetask run -b "$REPO" \
    -p "$PIPE_DIR/pipelines/_ingredients/cpBias.yaml" \
    -i "$CURATED,$RUN_RAW" \
    -o "$CP_BIAS" \
    -d "instrument='Nickel' AND exposure.observation_type='bias'" \
    --register-dataset-types
butler certify-calibrations "$REPO" "$CP_BIAS" "$CURATED" bias \
    --begin-date 2020-01-01 --end-date 2030-01-01

##############################################################################
# 3.  Flat masters
##############################################################################
echo "▶ Creating flat masters..."
pipetask run -b "$REPO" \
    -p "$PIPE_DIR/pipelines/_ingredients/cpFlat.yaml" \
    -c cpFlatIsr:doDark=False \
    -i "$CURATED,$RUN_RAW,$CP_BIAS" \
    -o "$CP_FLAT" \
    -d "instrument='Nickel' AND exposure.observation_type='flat'" \
    --register-dataset-types
butler certify-calibrations "$REPO" "$CP_FLAT" "$CURATED" flat \
    --begin-date 2020-01-01 --end-date 2030-01-01

##############################################################################
# 4.  Science frames – full PSF fit
##############################################################################
echo "▶ Processing science frames (full PSF)..."
pipetask run -b "$REPO" \
    -p ./pipelines/ProcessCcd.yaml#processCcd \
    -i "$CURATED,$RUN_RAW,$CP_BIAS,$CP_FLAT" \
    -o "$PROC_CCD" \
    -d "instrument='Nickel' AND exposure.observation_type='science'" \
    --register-dataset-types || echo "⚠️  Full PSF pipeline encountered failures. Will check for fallback candidates."
