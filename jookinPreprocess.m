% Memphis Jookin - Final Pre-Processing
% This script was developed based on Makoto's suggested pre-processing
% pipeline. Note that in addition to using AMICA, it still attempts 
% to perform dipole fitting through the information gained through ICA.

rawDataFiles = dir('../data/eeg/experiences/*.set');

for dataset_index = 1:length(rawDataFiles) 
    loadName = rawDataFiles(dataset_index).name;
    dataName = loadName(1:end-4);

    % Step 1: Split components of file name
    parts = split(dataName, '_');
    id = parts{1};
    if length(parts) == 3
        experience = append(parts{2},'_',parts{3});
    else
        experience = parts{2};
    end

    % Step 2: Import data and remove ACC channels
    EEG = pop_loadset('filename',loadName,'filepath','../data/eeg/experiences/');
    EEG.setname = dataName;
    EEG = pop_select( EEG, 'nochannel',{'x_dir','y_dir','z_dir'}); % remove ACC channels

    % Step 3: Bandpass filter 1-45 Hz
    EEG = pop_eegfiltnew(EEG, 'locutoff',1,'hicutoff',45);

    % Step 4: Import Channel Locations
    EEG = pop_chanedit(EEG, 'lookup','C:\Users\ntasnim\Documents\eeglab/plugins/dipfit/standard_BEM/elec/standard_1005.elc','eval','chans = pop_chancenter( chans, [],[]);');
    chanlocs = EEG.chanlocs; % saving channel locations for later use
 
    % Step 5: Remove Bad Channels
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion',5,'ChannelCriterion',0.8,...
        'LineNoiseCriterion',4,'Highpass','off','BurstCriterion','off',...
        'WindowCriterion','off','BurstRejection','off','Distance','Euclidian');

    % Save list of removed channels to .csv file compiling removed channels for all datasets
    % Initialize removed_channels
    if length(EEG.chaninfo.removedchans) > 3
        removed_channels = cell(length(EEG.chaninfo.removedchans) - 3, 1);
        for i = 1:(length(EEG.chaninfo.removedchans) - 3) % Exclude 3 acc channels
            removed_channels{i} = EEG.chaninfo.removedchans(i + 3).labels;
        end
        removed_channels_str = strjoin(removed_channels, ';'); % Combine into a single string
    else
        disp('EEGLAB did not identify any bad channels.');
        removed_channels_str = '';
        removed_channels = {};
    end
    
    % Create a table for appending
    data_to_append = table({id}, {experience}, {removed_channels_str}, ...
        'VariableNames', {'id', 'experience', 'removed_channels'});
    
    % Define output file path
    output_file = '../docs/removed_channels.csv';
    
    % Check if the file exists
    if isfile(output_file)
        % Append data
        writetable(data_to_append, output_file, 'WriteMode', 'append', 'QuoteStrings', true);
    else
        % Create the file and write headers
        writetable(data_to_append, output_file, 'WriteMode', 'overwrite', 'QuoteStrings', true);
    end

    % Correct Data with ASR
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion','off','ChannelCriterion','off',...
        'LineNoiseCriterion','off','Highpass','off','BurstCriterion',20,...
        'WindowCriterion','off','BurstRejection','off','Distance','Euclidian');

    % Step 8: Interpolate all the removed channels if channel rejection is applied. Otherwise, this line does not do anything.
    EEG = pop_interp(EEG, chanlocs, 'spherical');
 
    % Step 9: Re-reference the data to average
    % We ideally want to keep as many ICs as possible for ICA
    % So we had a "zero" channel to represented FCz before re-referencing
    % This prevents us from losing 1 rank value

    EEG.nbchan = EEG.nbchan+1;
    EEG.data(end+1,:) = zeros(1, EEG.pnts);
    EEG.chanlocs(1,EEG.nbchan).labels = 'initialReference';
    EEG = pop_reref(EEG, []);
    EEG = pop_select( EEG,'nochannel',{'initialReference'}); % removes the zero channel we created
 
    % Step 10: Run AMICA using calculated data rank with 'pcakeep' option
    dataRank = sum(eig(cov(double(EEG.data'))) > 1E-6); % 1E-6 follows pop_runica() line 531, changed from 1E-7.
    numprocs = 1; % # of nodes
    max_threads = 8; % # of threads
    num_models = 1; % # of models of mixture ICA
    
    % run amica - will iterate 2000 times (default value)
    [weights,sphere,mods] = runamica15(EEG.data, 'num_models',num_models,...
        'numprocs', numprocs, 'max_threads',max_threads,'outdir',dataName,'pcakeep',dataRank,...
        'do_reject', 1, 'numrej', 15, 'rejsig', 3, 'rejint', 1);

    % Load Tmeporary AMICA Output to Save with Dataset
    EEG.etc.amica  = loadmodout15(dataName);
    EEG.icaweights = weights;
    EEG.icasphere  = sphere;
    EEG = eeg_checkset(EEG, 'ica');
 
    % Step 11: Estimate single equivalent current dipoles
    % Note: if ft_datatype_raw() complains about channel numbers, comment out (i.e. put % letter in the line top) line 88 as follows
    % assert(size(data.trial{i},1)==length(data.label), 'inconsistent number of channels in trial %d', i);
    templateChannelFilePath = 'C:\Users\ntasnim\Documents\eeglab\plugins\dipfit\standard_BEM\elec\standard_1005.elc';
    hdmFilePath             = 'C:\Users\ntasnim\Documents\eeglab\plugins\dipfit\standard_BEM\standard_vol.mat';
    EEG = pop_dipfit_settings( EEG, 'hdmfile', hdmFilePath, 'coordformat', 'MNI',...
        'mrifile', 'C:\Users\ntasnim\Documents\eeglab\plugins\dipfit\standard_BEM\standard_mri.mat',...
        'chanfile', templateChannelFilePath, 'coord_transform',[-5.7789e-06 -1.9739e-06 -8.6998e-06 1.3204e-07 2.2579e-07 -1.5708 1 1 1],'chansel', 1:EEG.nbchan);
    EEG = pop_multifit(EEG, 1:EEG.nbchan,'threshold', 100, 'dipplot','off','plotopt',{'normlen' 'on'});
 
    % Step 12: Search for and estimate symmetrically constrained bilateral dipoles
    % Note this is from Piazza et al. (2016), which has insights on the
    % value of using ICs for source localization
    EEG = fitTwoDipoles(EEG, 'LRR', 35);
 
    % Step 13: Run ICLabel (Pion-Tonachini et al., 2019)
    EEG = iclabel(EEG, 'default');

    % Step ##: Flag and remove ICs - WE WILL REMOVE ICs later!
    % EEG = pop_icflag(EEG,[NaN NaN;0.9 1;0.9 1;0.9 1;0.9 1;0.9 1;0.9 1]); % label >90% for muscle,eye,heart,line noise, other
    % EEG = pop_subcomp( EEG, [], 0); % remove components marked for rejection
 
    % Save the dataset
    EEG = pop_saveset(EEG, 'filename', dataName, 'filepath', '../data/eeg/preprocessed/');

end