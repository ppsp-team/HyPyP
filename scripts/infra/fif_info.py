import mne, sys
for f in ["tests/data/preproc_S1.fif", "tests/data/preproc_S2.fif"]:
    ep = mne.read_epochs(f, preload=False, verbose=False)
    info = ep.info
    si = info.get("subject_info")
    print(f)
    print("  n_epochs", len(ep), "n_ch", len(info["ch_names"]), "sfreq", info["sfreq"], "tmin/tmax", ep.tmin, ep.tmax)
    print("  meas_date present:", info["meas_date"] is not None)
    print("  subject_info present:", si is not None, "keys:", sorted(si.keys()) if si else None)
    if si:
        for k in ("his_id","first_name","last_name","middle_name","birthday","sex","hand"):
            print(f"    {k}: {'SET' if si.get(k) not in (None, '', 0) else 'empty/none'}")
    print("  experimenter:", "SET" if info.get("experimenter") else "none", "| description:", "SET" if info.get("description") else "none", "| proj_name:", "SET" if info.get("proj_name") else "none")
    print("  device_info:", info.get("device_info"))
    print("  ch_names[:5]", info["ch_names"][:5])
    print("  file_id / meas_id present:", info.get("file_id") is not None, info.get("meas_id") is not None)
