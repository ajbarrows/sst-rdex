% Define the directory containing the .mat files
inputDir = '../../data/01_raw/abcd-sync/6.0/imaging_concat/vertexwise/tfmri'; % Update this path
outputDir = '../../data/02_intermediate/'; % Update this path

% Create the output directory if it does not exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get a list of all .mat files in the directory
matFiles = dir(fullfile(inputDir, '*.mat'));

% Loop through each .mat file
for i = 1:length(matFiles)
    % Load the .mat file
    matFileName = fullfile(inputDir, matFiles(i).name);
    data = load(matFileName);

    % Assuming the .mat file contains a single variable
    varName = fieldnames(data);
    tableData = data.(varName{1}); % Extract the variable

    % Create the output .parquet file name
    [~, name, ~] = fileparts(matFileName);
    parquetFileName = fullfile(outputDir, [name, '.parquet']);

    % Write the data to a .parquet file
    writetable(tableData, parquetFileName);

    % Display a message
    fprintf('Converted %s to %s\n', matFiles(i).name, [name, '.parquet']);
end
