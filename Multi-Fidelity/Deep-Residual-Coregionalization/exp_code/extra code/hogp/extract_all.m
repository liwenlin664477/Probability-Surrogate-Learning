function [ker_params, U, bta] = extract_all(params, nvec, r, ker_type)
        nmod = length(nvec);
        U = cell(nmod, 1);
        ker_params = cell(nmod,1);
        %extract parameters
        [ker_params{1},idx] = load_kernel_parameter(params, r(1), ker_type, 0);
        for k=2:nmod
            U{k} = reshape(params(idx+1:idx+nvec(k)*r(k)),nvec(k), r(k));
            [ker_params{k},idx] = load_kernel_parameter(params, r(k), ker_type, idx+nvec(k)*r(k));
        end
        bta = exp(params(idx+1));
        assert(idx+1==length(params), 'inconsistent paramter number!');
end