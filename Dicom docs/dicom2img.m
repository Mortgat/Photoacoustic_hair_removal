% infolder = '.\IPh009_L\PA756\Dicom\';
infolder = '.\IPh009_L\PA797\Dicom\';

infiles = dir(fullfile(infolder,'*.dcm'));
Data = zeros([400 1120 1120]);
for fidx=1:length(infiles)
    infile = fullfile(infolder,infiles(fidx).name);
    [X, map] = dicomread(infile);
    Data(:,:,fidx) = double(X);
end

% out
% out = 'IPH009L_PA756.mat';
out = 'IPH009L_PA797.mat';
save(out,'Data','-v7.3');

% local MIP
% outfolder = fullfile(infolder,'MIP','xy');
% mkdir(outfolder);
% for ii=1:size(Data,3);
%     M = Data(:,:,ii:end);
%     M = max(M,[],3);
%     out = fullfile(outfolder,[num2str(ii,'%04d') '.tif']);
%     imwrite(M,out);
% end


1;

