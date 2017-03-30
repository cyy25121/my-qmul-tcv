function e = evalcf()
load('struct_TP_FP');
count = 0;
miss = 0;
for i = 1:3
    cl_sz = size(struct_TP_FP.class(i).seq,2);   
    for j = 1:cl_sz
        % Positive = overlap with ground true should be 1 and the predicted
        % class should be the same class as we look up
        temp_ind_pos = ((struct_TP_FP.class(i).seq(j).array(1,:)==1) & ...
            (struct_TP_FP.class(i).seq(j).array(3,:)==i));
        temp_class = struct_TP_FP.class(i).seq(j).array(3,temp_ind_pos);
        % noting found
        if cumsum(temp_ind_pos) == 0
            miss = miss + 1;
        % found, but miss the class
        elseif temp_class(1) ~= i 
            miss = miss + 1;
        end
        count = count + 1;
    end
end
e = miss/count;
fprintf('Missclassification rate = %f\n', e);