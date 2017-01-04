function fileLines = readTextFile(fileName)

fid = fopen(fileName, 'r');
fileLines = textscan(fid, '%s', 'delimiter', '\n', 'bufsize', 99900000);
fclose(fid);
fileLines = fileLines{1};

% % % %followed for instance by:
% for li = 1:length(fileLines)
%     [~, ~, ~, ~, ~, ~, splitLine] = regexp(fileLines{li}, '\t');
% %...
% end