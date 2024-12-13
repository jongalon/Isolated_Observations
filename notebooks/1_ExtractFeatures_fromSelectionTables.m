clc
clear all

% Directory where the data files are located
data_path = "../data/";
audio_path = "../audios/";

% Subset selection
subset = "Dataset_1/";
%subset = "Dataset_2/";
%subset = "Automatic_Labeling_Kyoogu/";

file = 'Labeled_Data.xlsx';
file_path = fullfile(data_path, subset, file);
disp(file_path)

if subset == "Dataset_2/"
    dataset = 'Dataset_2/';
else
    dataset = 'Dataset_1/';
end


dir_audio = fullfile(audio_path, dataset);
disp(dir_audio)

% Load information
table_data = readtable(file_path, 'ReadVariableNames', true);

%%

size(table_data);
files = table2cell(table_data(:, 'File'));
species = table2cell(table_data(:, 'SpecieID'));

% Preload spectrograms to avoid recomputation
unique_files = unique(files);
spectrogram_cache = containers.Map;

for i = 1:length(unique_files)
    file = unique_files{i};
    audio_path = fullfile(dir_audio, file); % Create the full path to the audio file
    
    % Check if the file exists
    if isfile(audio_path)
        % Read the audio file
        [y, fs] = audioread(audio_path);
        % Calculate spectrogram
        window_size = 1024; 
        overlap = window_size / 2;
        nfft = window_size * 2;
        [s, f, t, ~] = spectrogram(y(:,1), hann(window_size), overlap, nfft, fs, 'yaxis');
        
        s = abs(s);
        f1 = flip(f);
        s_flip = flip(s);
        spectrogram_cache(file) = struct('s_flip', s_flip, 'f1', f1, 't', t, 'fs', fs, 'u', size(s, 1));
    else
        disp(['File not found: ', file]);
    end
end

% Process each segment
data = [];
row = 1;
for i = 1:size(table_data, 1)
    file = table_data.File{i};
    if ~isKey(spectrogram_cache, file)
        continue;
    end

    % Retrieve cached spectrogram
    spec_data = spectrogram_cache(file);
    s_flip = spec_data.s_flip;
    f1 = spec_data.f1;
    t = spec_data.t;
    fs = spec_data.fs;
    u = spec_data.u;

    start_time = table_data.Start(i);
    end_time = table_data.End(i);
    fmax = table_data.FmaxVoc(i);
    fmin = table_data.FminVoc(i);

    specie_id = table_data.SpecieID(i);

    % Positions in the spectrogram
    [~, posX] = min(abs(t - start_time));
    [~, posXplusW] = min(abs(t - end_time));
    [~, posY] = min(abs(f1 - fmin));
    [~, posYplusH] = min(abs(f1 - fmax));

    % Extract segment from the spectrogram
    seg = s_flip(posYplusH:posY, posX:posXplusW);

    % Dominant frequency
    sum_domin = sum(seg, 2);
    [~, dom] = max(smooth(sum_domin));                                 
    dom = ((((fmin * u / (fs / 2)) + dom) / u) * fs / 2);

    % Calculate FCCs
    n_freq = 4;
    div = 4;
    n_filters = 14;

    features = FCC(seg, n_filters, n_freq, div);

    dfcc = diff(features, 1, 2);
    dfcc2 = diff(features, 2, 2);
    features = [features(:); mean(dfcc, 2); mean(dfcc2, 2)];

    % Save data if it meets the condition
    if end_time > start_time && fmax > fmin
        data(row, :) = [start_time, end_time, fmin, fmax, dom, features(2:end)']; %First FCC is not taken into account.
        row = row + 1;
    end
    disp(i);
end

result_data = [files, species, num2cell(data)];
column_names = {'File', 'Specie ID', 'Start', 'End', 'FminVoc', 'FmaxVoc', 'Fdom', 'FCC1', ...
    'FCC2', 'FCC3', 'FCC4', 'FCC5', 'FCC6', 'FCC7', 'FCC8', 'FCC9', 'FCC10', 'FCC11', 'FCC12', ...
    'FCC13', 'FCC14', 'FCC15', 'FCC16', 'FCC17', 'FCC18', 'FCC19', 'FCC20', 'FCC21', 'FCC22', 'FCC23'};
concat = [column_names; result_data];

%%
% Construct output file name
subset_clean = char(strrep(subset, '/', '')); % Ensure subset is a proper string
output_file = ['Labeled_Data_With_FCCs_' subset_clean '.xlsx'];
disp(['Output file: ' output_file]);

% Write to file
writecell(concat, output_file, 'Sheet', 1);
