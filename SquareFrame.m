%----------------------------------------------------------
% Copyright (C) Etore Maloso Tronconi
% Embedded Systems & Instrumentation Department
% École Supérieure d'Ingénieurs - ESIGELEC, France
% This software is intended for research purposes only;
% its redistribution is forbidden under any circumstances.
%----------------------------------------------------------

%=====
% This is a script that creates a dataset with squares, with some covered
% parts, it is intend to mimimc a dataset where there is a target structure
% to be segmented, but there are parts of this structures that are covered
% oui oui baguette, 
%
%==PARAMETERS================================================

%Size of images
d = 512; 

imgIdx = 1;
%Change path here to train/test to save the files.
file_path  ="./dataset/test";
numImages = 20; 
numSquares = 5;
maxNumCircles = 3;
%==========================================================

while imgIdx <= numImages
    badImage = false;
    RGB = cell(1,4);
    for k = 1:4
        RGB{k} = zeros(d, d, 'uint8');
    end

    
    [RGB ,Label,nS] = DrawSquares(d,numSquares,RGB);
    [RGB] = DrawCircles(d,maxNumCircles,RGB);
    
    for i=1:4

            Img = rgb2gray(RGB{i});
            num = bwconncomp(Img,4).NumObjects;
            
            if num ~= maxNumCircles + 2
                            badImage = true;
                break
              
            end    
    end
    if ~badImage    
        imwrite(Label, sprintf(file_path +'/labels/label_%03d.png',imgIdx));
            
        for i=1:4
            imwrite(RGB{i}, sprintf(file_path +'/images/image_%03d_frame_%03d.png', imgIdx,i));
        end
        imgIdx = imgIdx + 1;
    end

end

function [RGB,Label,num_sq] =  DrawSquares(d,num_sq,RGB)
     
    [S] = SquareCoordinates(d,num_sq);
    c =randi([10, 255], 1, 3);
    fixedSquare = S(1,:);
    
    Label = insertShape(RGB{1}, "filled-rectangle", fixedSquare, 'Color', [255,255,255], 'Opacity', 1);
    RGB_in = RGB{1};
    for i=1:num_sq-1
        cover = SquareCover(fixedSquare);
        
        RGB{i} = insertShape(RGB_in, "filled-rectangle", fixedSquare, 'Color', c, 'Opacity', 1);
        
        if strcmp(cover.shape, 'filled-rectangle') 
            RGB{i} = insertShape(RGB{i}, 'filled-rectangle', [cover.xc, cover.yc, cover.wc, cover.hc], 'Color', [0 0 0], 'Opacity', 1);
        else
            RGB{i} = insertShape(RGB{i}, 'filled-circle', [cover.xc, cover.yc, cover.wc], 'Color', [0 0 0], 'Opacity', 1);
        end

        cover = SquareCover(S(i+1,:));

        RGB{i} = insertShape(RGB{i}, "filled-rectangle", S(i+1,:), 'Color', randi([10,255],1,3), 'Opacity', 1);   RGB{i} = insertShape(RGB{i}, "filled-rectangle", S(i+1,:), 'Color', randi([10, 255], 1, 3), 'Opacity', 1);
        if strcmp(cover.shape, 'filled-rectangle') 
            RGB{i} = insertShape(RGB{i}, 'filled-rectangle', [cover.xc, cover.yc, cover.wc, cover.hc], 'Color', [0 0 0], 'Opacity', 1);
        else
            RGB{i} = insertShape(RGB{i}, 'filled-circle', [cover.xc, cover.yc, cover.wc], 'Color', [0 0 0], 'Opacity', 1);
        end

    end

end


function cover = SquareCover(S)

    shape_list = {'filled-circle'};

    cover.shape = shape_list{randi(numel(shape_list))};

    x = S(1); 
    y = S(2); 
    width  = S(3); 
    height = S(4); 
    omega  = S(5);


    cover.xc = x + (2*randi([0 1]) - 1)*width/2;
    cover.yc = y + (2*randi([0 1]) - 1)*height/2;
    k = 3*rand() + 1;
    cover.wc = width/k;

    cover.hc = cover.wc ;

    cover.omega = omega;

    
end


function    [RGB, nC, C] = DrawCircles(d,nC,RGB)
    
    for j=1:4

        [C, ~] = CirclesCoordinates(d,nC); 
        for i = 1:nC
            RGB{j} = insertShape(RGB{j}, 'filled-circle', C(i,:), 'Color', randi([10, 255], 1, 3), 'Opacity', 1);
        end
        
    end
end

function [S_out] = SquareCoordinates(d,nS)
    
    Sq = zeros(1,5); 
    for i=1:nS
        
        a = (d/4 - d/8) * rand + d/8;
        omega =  360 * rand();
                
        x = a + ((d-a) - a)*rand;
        y = a + ((d-a) - a)*rand;
        
        Sq(i, :) = [x, y, a, a, omega];
    end
    S_out = Sq;
    
end


function [C, nC] = CirclesCoordinates(d,nC)
    
    C = zeros(nC, 3);
    for i = 1:nC
        
        radius = d/16 - d/32 * rand + d/32;
        
        x = radius + ((d-radius) - radius)*rand;
        y = radius + ((d-radius) - radius)*rand;

        C(i, :) = [x, y, radius];
    end

 end