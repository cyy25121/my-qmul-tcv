function hough_array = houghvoting(patches,position,spa_scale,tem_scale,frame_num,flag_mat,struct_cb)
% hough_array           : is a matrix with 7 rows, each column indicating the
%                       : predicted(votes) spatial location, predicted (voted) start and end frames,
%                       : the votes, bounding box values(scale compensated). Example: Let a descriptor matched with
%                       : the codeword (with one offset) cast votes at the spatial location [x,y],
%                       : temporal location [s,e] (predictions for start and end frames), bounding box values [b1 b2] (stored during training)
%                       : with value(weight) 'v', then corresponding column in the
%                       : matrix hough_array will be [x y s e v b1 b2]'.
% patches               : i/p descriptors
% position              : spatial location of the detected STIP                   
% spa_scale             : spatial scale                      
% tem_scale             : temporal scale
% frame_num             : frame number at which STIP was detected
% flag_mat              : refer to ism_test_voting.m

%-----------------------------------------------------------------------------------------------------
% Write your code here to compute the matrix hough_array
%-----------------------------------------------------------------------------------------------------
  
