% Define the directory containing the .mat files
inputDir = '../../data/01_raw/abcd-sync/6.0/imaging_concat/vertexwise/tfmri'; % Update this path
outputDir = "../../data/01_raw/abcd-sync/6.0/imaging_concat/vertexwise/tfmri/parquet";


% Create the output directory if it does not exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get a list of all .mat files in the directory
%matFiles = dir(fullfile(inputDir, '*.mat'));
% sst = dir(fullfile(inputDir, 'SST_*.mat'));
% mid = dir(fullfile(inputDir, 'mid_*.mat'));
nback = dir(fullfile(inputDir, 'nback_*_sem*.mat'));
% nback = dir(fullfile(inputDir, 'nback_*.mat'))d;
% vol_info = dir(fullfile(inputDir, 'vol_info.mat'));
% matFiles = [sst; mid; nback; vol_info];
matFiles = [nback];

% Create a parallel pool
if isempty(gcp('nocreate'))
    parpool('local');
end

% Use parfor loop for parallel execution
parfor i = 1:length(matFiles)
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
    parquetwrite(parquetFileName, tableData);

    % Display a message
    fprintf('Converted %s to %s\n', matFiles(i).name, [name, '.parquet']);
end
