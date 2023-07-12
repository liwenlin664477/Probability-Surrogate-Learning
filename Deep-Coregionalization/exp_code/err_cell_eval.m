function err = err_cell_eval(YPred,yTrue)

    assert(iscell(YPred),'yPred is not a cel');
    err = [];
    
    for k = 1:length(YPred)
        err_k = err_eval(YPred{k},yTrue);
        err = cat(2,err,err_k);
    end

end