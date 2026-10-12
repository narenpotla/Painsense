"""
painsense - library code for the PainSense prototype.

Modules (imported by the scripts in the project root, never run directly):

    config       every path and setting
    utils        seed_everything
    face         face detection + the single crop recipe
    data         labels, 0-10 targets, windows, augmentation, splits
    models       networks + checkpoint loading
    features     feature cache, cross-validation engine, metrics
    alerts       AlertState: median of 5 + hysteresis
    live         live scoring loop (camera or video file)
    live_state   latest result for the dashboard
    db           SQLite score log
"""
