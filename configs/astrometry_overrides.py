# CalibrateImage → AstrometryTask tuning
a = config.astrometry

# Use bright *enough* stars, but not too strict (avoid starvation in B/V)
a.sourceSelector.name = "science"
ss = a.sourceSelector["science"]
ss.doSignalToNoise = True
ss.signalToNoise.minimum = 10.0          # was 20.0 — lower to keep candidates
ss.signalToNoise.maximum = None
ss.signalToNoise.fluxField = "slot_CalibFlux_instFlux"
ss.signalToNoise.errField  = "slot_CalibFlux_instFluxErr"
# keep defaults from setDefaults: doFlags True; flags.good=["calib_psf_candidate"]; unresolved/isolation off

# Widen initial search (handles bad header pointing/rotation)
m = a.matcher
if hasattr(m, "maxOffsetPix"):       m.maxOffsetPix = 3900     # ~13' at 0.2"/px; ~26' at 0.4"/px
if hasattr(m, "maxRotationDeg"):     m.maxRotationDeg = 5.0
if hasattr(m, "matcherIterations"):  m.matcherIterations = 10
if hasattr(m, "maxRefObjects"):      m.maxRefObjects = 30000
if hasattr(m, "maxStars"):           m.maxStars = 8000
if hasattr(m, "minMatchPairs"):      m.minMatchPairs = 8

# Relax convergence a bit for the first lock; we'll tighten later
a.maxMeanDistanceArcsec = 180.0       # was 120/60
a.matchDistanceSigma    = 10.0
a.maxIter               = 25

# Make sure the ref loader fetches a wide-enough sky area when WCS is off
rl = config.astrometry_ref_loader
if hasattr(rl, "pixelMargin"):
    rl.pixelMargin = 5000             # pull a much larger area of ref stars
