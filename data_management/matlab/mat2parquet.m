% Define the directory containing the .mat files
%inputDir = '../../data/01_raw/abcd-sync/5.0/imaging_concat/vertexwise/tfmri'; % Update this path
inputDir = '../../data/01_raw/abcd-sync/6.0/imaging_concat/vertexwise/tfmri'; % Update this path


%outputDir = '../../data/02_intermediate/betas/r5'; % Update this path
outputDir = '../../data/02_intermediate/betas/r6'; % Update this path

% Create the output directory if it does not exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get a list of all .mat files in the directory
%matFiles = dir(fullfile(inputDir, '*.mat'));

cg = dir(fullfile(inputDir, 'sst_cg_*.mat'));
cs = dir(fullfile(inputDir, 'sst_cs_*.mat'));
is = dir(fullfile(inputDir, 'sst_is_*.mat'));
ig = dir(fullfile(inputDir, 'sst_ig_*.mat'));
vol_info = dir(fullfile(inputDir, 'vol_info.mat'));
matFiles = [cg; cs; is; ig];

%matFiles = vol_info;

% Loop through each .mat file
for i = 1:length(matFiles)
    % Load the .mat file
    matFileName = fullfile(inputDir, matFiles(i).name);
    data = load(matFileName);

    % Assuming the .mat file contains a single variable
    varName = fieldnames(data);
    tableData = data.(varName{1}); % Extract the variable
    tableData = array2table(tableData); % Convert to a table

    % Create the output .parquet file name
    [~, name, ~] = fileparts(matFileName);
    parquetFileName = fullfile(outputDir, [name, '.parquet']);

    % Write the data to a .parquet file
    %writetable(tableData, parquetFileName);
    parquetwrite(parquetFileName, tableData);

    % Display a message
    fprintf('Converted %s to %s\n', matFiles(i).name, [name, '.parquet']);
end


% write vol_info

%matFileName = fullfile(inputDir, 'vol_info.mat');
%data = load(matFileName);

%src_subject_id = data.subjidvec;
%eventname = data.eventvec;
%vol_info = table(src_subject_id, eventname);

 % Create the output .parquet file name
%[~, name, ~] = fileparts(matFileName);
%parquetFileName = fullfile(outputDir, [name, '.parquet']);

%parquetwrite(parquetFileName, vol_info);
