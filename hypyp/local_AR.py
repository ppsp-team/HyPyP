import mne
from autoreject import AutoReject, set_matplotlib_defaults 

def AR_local(cleaned_epochs_ICA):
        """Apply local Autoreject
    Parameters
    ----------
    clean_epoch_concat : instance of Epochs after global Autoreject 
                         and ICA

    Returns
    -------
    cleaned_epochs_AR : instance of Epochs after local Autoreject
    """
    bad_epochs_AR = []

    ## defaults values for n_interpolates and consensus_percs
    n_interpolates = np.array([1, 4, 32])
    consensus_percs = np.linspace(0, 1.0, 11)

    for clean_epochs in cleaned_epochs_ICA:  # per subj

        picks = mne.pick_types(clean_epochs[0].info, meg=False, eeg=True, stim=False, eog=False, exclude = [])

        ar = AutoReject(n_interpolates, consensus_percs, picks=picks,
                        thresh_method='random_search', random_state=42)

        ## fitting AR to get bad epochs 
        ar.fit(clean_epochs)
        reject_log = ar.get_reject_log(clean_epochs,picks=picks) 
        bad_epochs_AR.append(reject_log)
    
    ## taking bad epochs for min 1 subj (dyad)
    log1=bad_epochs_AR[0]
    log2=bad_epochs_AR[1]
    
    bad1 = np.where(log1.bad_epochs==True)
    bad2 = np.where(log2.bad_epochs==True)
    
    bad = list(set(bad1[0].tolist()).intersection(bad2[0].tolist()))
    print('%s percent of bad epochs' % int(len(bad)/len(list(log1.bad_epochs))*100))
        
    ## picking good epochs for the two subj
    cleaned_epochs_AR = []
    for clean_epochs in cleaned_epochs_ICA:  # per subj
        clean_epochs_AR = clean_epochs.drop(indices=bad)
        cleaned_epochs_AR.append(clean_epochs_AR)

    ## Vizualisation before after AR
    evoked_before = []
    for clean_epochs in cleaned_epochs_ICA:  # per subj
            evoked_before.append(clean_epochs.average())
    evoked_after_AR = []
    for clean in cleaned_epochs_AR:
        evoked_after_AR.append(clean.average())

    for i,j in zip(evoked_before,evoked_after_AR):
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        for ax in axes:
            ax.tick_params(axis='x', which='both', bottom='off', top='off')
            ax.tick_params(axis='y', which='both', left='off', right='off')

        ylim = dict(grad=(-170, 200))
        i.pick_types(eeg=True, exclude=[])
        i.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
        axes[0].set_title('Before autoreject')
        j.pick_types(eeg=True, exclude=[])
        j.plot(exclude=[], axes=axes[1], ylim=ylim)
              # Problème titre ne s'affiche pas pour le deuxieme axe !!!
        axes[1].set_title('After autoreject')
        plt.tight_layout()


    return cleaned_epochs_AR