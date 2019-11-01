function FeasibleSet=exPostFeasible(oPref, match)
    
    utilityBySlot  = bsxfun(@eq, 1:size(oPref,2), match);
    utilityBySlot  = double(utilityBySlot);
    utilityBySlot  = oPref.*utilityBySlot;
    utilityBySlot(utilityBySlot == 0) = Inf;
    cutoffs = min(utilityBySlot,[],1);
    cutoffs(1,end) = -Inf; %Last option is always feasible for students
    
    FeasibleSet = NaN(size(oPref));
    
    for j = 1:size(oPref,2)
        minUtilityi = cutoffs(j);
        for i = 1:size(oPref,1)
            utilityOptioni = oPref(i,j);
            if utilityOptioni>=minUtilityi
                FeasibleSet(i,j) = true;
            else
                FeasibleSet(i,j) = false;
            end
        end
    end
    
    FeasibleSet = logical(FeasibleSet);

end