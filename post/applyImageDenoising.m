function outImage = applyImageDenoising(inImage, method)
% intended to work on grayscale
inImage(inImage == 0) = nan;
switch method
    case 'none'
        outImage = inImage;
    case 'median' % used for all images and results in paper
        outImage = medfilt2(inImage);
    case 'wavelet'
        outImage = wdenoise2(inImage,1);
    case 'max'
        outImage = ordfilt2(inImage,9,true(3));     
    case 'min'
        outImage = ordfilt2(inImage,1,true(3));  
    case 'max-min-avg'
        inImage = double(inImage);
        outImage = (ordfilt2(inImage,3,true(3)) + ordfilt2(inImage,7,true(3))) / 2;
        outImage = uint8(outImage);
    case 'average'
        f = ones(3) / 9;
        outImage = imfilter(inImage, f); 
    case 'disk'
        f = fspecial('disk',2);
        outImage = imfilter(inImage, f); 
    case 'gaussian'
        outImage = imgaussfilt(inImage, 0.75);
    case 'weighted-median' 
        a = double(inImage);
        W = [1, 1, 1; 1, 4, 1; 1, 1, 1] / 12;
        [row, col] = size(inImage);
        outImage = a;
        for x = 2:1:row-1
            for y = 2:1:col-1
                % To make a 3x3 weighted mask into a 1x9 mask
                a1 = [W(1)*a(x-1,y-1) W(2)*a(x-1,y) W(3)*a(x-1,y+1) ...
                      W(4)*a(x,y-1) W(5)*a(x,y) W(6)*a(x,y+1)...
                      W(7)*a(x+1,y-1) W(8)*a(x+1,y) W(9)*a(x+1,y+1)];
                a2 = sort(a1);
                med = a2(5); % the 5th value is the weighted median 
                outImage(x,y) = med;
            end
        end
        outImage = uint8(outImage);
    case 'wiener' 
        outImage = wiener2(inImage,[5 5]);
    case 'bilateral'
        DoS = 100;
        outImage = imbilatfilt(inImage, DoS, 10);
    case 'local-laplacian'
        outImage = locallapfilt(inImage, 0.2, 3);       
end
end