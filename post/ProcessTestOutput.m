% Script to convert output of magic_test.py to final format
% Converting grayscale maps into colorized maps separated by type

clc; clear; close all;
set(0,"DefaultFigureVisible","off");

% Change the following folder name to the directory containing the output
% of magic_test.py
sample_test_folder = fullfile("../sample_test/test_imgs_expected_results");

% Load colormap
load("Rapid_Colormap.mat"); c_map = Rapid_U;

% Set and create save location
sample_test_output_dir = fullfile("../sample_test/test_imgs_expected_results_processed");
if ~exist(sample_test_output_dir,"dir"),mkdir(sample_test_output_dir);end

% Postprocessing method - Median was used to achieve final results reported in paper
method = "median"; 
unit = 256; % img px size

images = dir(sample_test_folder);

for i = 1:length(images)
    imgname = images(i).name;
    if strcmp(imgname(1),"."), continue; end
    
    full_ctp = imread(fullfile(images(i).folder, imgname));
    
    mtt = rgb2gray(full_ctp(:,unit*0+1:unit*1,:));
    ttp = rgb2gray(full_ctp(:,unit*1+1:unit*2,:));
    cbf = rgb2gray(full_ctp(:,unit*2+1:unit*3,:));
    cbv = rgb2gray(full_ctp(:,unit*3+1:unit*4,:));
    
    saveImageFinal(applyImageDenoising(mtt, method), c_map, imgname, sample_test_output_dir, "MTT");
    saveImageFinal(applyImageDenoising(ttp, method), c_map, imgname, sample_test_output_dir, "TTP");
    saveImageFinal(applyImageDenoising(cbf, method), c_map, imgname, sample_test_output_dir, "CBF");
    saveImageFinal(applyImageDenoising(cbv, method), c_map, imgname, sample_test_output_dir, "CBV");
end
fprintf("All outputs saved to %s", sample_test_output_dir);

function saveImageFinal(map_type, c_map, imgname, savepath, subfolder)
figure; imshow(map_type); colormap(c_map);
f = getframe;
map_type_savepath = makeSubfolder(savepath,subfolder);
savename = fullfile(map_type_savepath,imgname);
imwrite(f.cdata,savename);
close all;
end

function newfilepath = makeSubfolder(savepath,folder_name)
newfilepath = fullfile(savepath,folder_name);
if ~exist(newfilepath,'dir')
    mkdir(newfilepath);
end
end
