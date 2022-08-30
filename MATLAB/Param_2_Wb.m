function [W,b] = Param_2_Wb(p,n)
   L = length(n);
   idx = 1;
   for l = 2:L
      W{l} = reshape(p(idx:idx+n(l)*n(l-1)-1) ,[n(l),n(l-1)]); idx = idx+n(l)*n(l-1);
      b{l} = reshape(p(idx:idx+n(l)-1), [n(l),1]); idx = idx+n(l);
   end
end