%% Stochastic gradient on least squares problems
clc; clear all; close all;
%% Global parameters
global n L xin yin act_fn;
%% Network parameters
n      = [2 5 1];                % dimension of each layer
m      = 98;                     % number of training data
L      = length(n);              % number of total layers
act_fn = "logsig";               % activation function
totWb  = dot(n(1:L-1)+1,n(2:L)); % total number of weights and biases
%% LM parameters
mu     = 1d+3;                   % initial damping parameter
maxit  = 3d3;                    % maximum number of epochs
TOL    = 1d-12;                  % tolerance
epoch  = 1;                      % initial step
p0     = randn(totWb,1);         % initial guess 
%% Problem setup
bdA = 0; bdB = 1;                % domain boundary
alpha = .5;                      % interface location
fi    = @(x) sin(2*pi*x);        % target function 1
fo    = @(x) cos(2*pi*x);        % target function 2
f     = @(x,z) (1-z)/2.*fi(x) + (1+z)/2.*fo(x);
%% Given training data
xin   = [(bdB-bdA)*rand(1,m)+bdA bdA bdB]; % input data
xin   = [xin; hardlims(xin-alpha)];        % input data augmentation
yin   = f(xin(1,:),xin(2,:));              % target output
%% Display network parameters
disp(['Training points : ', num2str(length(xin))]);
disp(['Parameters      : ', num2str(totWb)]);
%% Preallocation
a      = cell(L,1);              % activation
loss   = zeros(maxit,1);         % loss function
mu_rec = zeros(maxit,1);         % record damping parameter
%% Training
tic;
while ( epoch <= maxit )
   % ... Computation of Jacobian matrix ...
   J = Jacobian_cost(p0);
   % ... Computation of the vector loss function ...
   res = cost_vec(p0);
   % ... Update p_{k+1} using LM algorithm ...
   [U,S,V] = svd(J,"econ");
   p = p0 + V*(U'.*(diag(S)./(diag(S).^2+mu)))*res';
   % ... Compute the loss function ...
   loss(epoch) = sum(res.^2);
   % ... Damping parameter strategy ...
   mu_rec(epoch) = mu;
   if mod(epoch,2) == 0 && loss(epoch)<loss(epoch-1), mu = max(mu/3,1d-09); end
   if mod(epoch,2) == 0 && loss(epoch)>loss(epoch-1), mu = min(mu*2,1d+08); end
   % ... Break the loop if the certain condition is satisfied ...
   if loss(epoch) <= TOL, break; end
   % ... Next iteration loop ...
   epoch = epoch+1; p0 = p;
end
loss( loss == 0 ) = [];
disp(['LOSS            : ', num2str(loss(end),'%.3e'),' (epoch = ', num2str(epoch),')']);
toc;
%% Test & Output
% ... Output the training history ...
figure(1); loglog(loss,'linewidth',2); 
set(gca,'fontsize',20,'linewidth',1); title(['loss = ', num2str(loss(end),'%.3e')]); xlabel('training step'); grid on;
% ... Convert solution p to weights and biases ...
[W,b] = Param_2_Wb(p,n);
% ... Set test points & exact solution ...
x_test = bdA:(bdB-bdA)/999:bdB;
a{1}   = [x_test; hardlims(x_test-alpha)];
f_test = f( a{1}(1,:), a{1}(2,:) );
% ... Forward pass ...
y_test = W{3}*activation( W{2}*a{1}+b{2}, act_fn ) + b{3};
% ... Output ...
disp(['L_inf error     : ', num2str( norm(y_test-f_test,inf), '%.3e' )]);

figure(2);
subplot(1,2,1);
plot(x_test, y_test, '.', x_test, f_test, 'o');
set(gca,'linewidth',2,'fontsize',16); 
title('\phi_{S} and \phi'); xlabel('x'); ylim([-1.1 1.1]); legend('\phi_S','\phi');

subplot(1,2,2);
plot(x_test(a{1}(2,:) == -1),abs(y_test(a{1}(2,:) == -1)-f_test(a{1}(2,:) == -1)),...
     x_test(a{1}(2,:) ==  1),abs(y_test(a{1}(2,:) ==  1)-f_test(a{1}(2,:) ==  1)),'color',[0 .45 .74],'linewidth',2);
set(gca,'linewidth',2,'fontsize',16); 
title('|\phi_{S} - \phi|'); xlabel('x');
%% Cost vector function
function f = cost_vec(p)
   global n L xin yin act_fn;
   % ... preallocation ...
   a = cell(L,1); % activation
   % ... convert parameter to weights and biases
   [W,b] = Param_2_Wb(p,n);
   % ... forward pass ...
   a{1} = xin;
   a{L} = W{3}*activation( W{2}*a{1}+b{2}, act_fn ) + b{3};
   % ... output ...
   f = yin - a{L};
end
%% Jacobian of cost function
function J = Jacobian_cost(p)
   global n L xin act_fn
   % ... preallocation ...
   a     = cell(L,1); % activation
   z     = cell(L,1); % pre-activation
   % ... convert parameter to weights and biases ...
   [W,b] = Param_2_Wb(p,n);
   % ... forward pass ...
   a{1} = xin;
   z{2} = W{2}*a{1}+b{2};
   a{2} = activation( z{2}, act_fn );
   da   = dactivation( z{2}, act_fn );
   % ... derivatives w.r.t. parameters ...
   dUb_b{2} = W{3}.* da';
   dUb_b{3} = ones(length(a{1}),1);
   dUb_W{2} = repmat(dUb_b{2},1,1+1).*kron(a{1}',ones(1,n(2)));
   dUb_W{3} = a{2}';
   % ... output ...
   J = [dUb_W{2} dUb_b{2} dUb_W{3} dUb_b{3}];
end
%% Activation function
function y = activation(x,act_fn)
   if strcmp(act_fn,'logsig'),  y = logsig(x);  end
   if strcmp(act_fn,'tansig'),  y = tansig(x);  end
   if strcmp(act_fn,'poslin'),  y = poslin(x);  end
   if strcmp(act_fn,'purelin'), y = purelin(x); end
end
function y = dactivation(x,act_fn)
   if strcmp(act_fn,'logsig'),  y = logsig(x).*(1-logsig(x)); end
   if strcmp(act_fn,'tansig'),  y = 1-tansig(x).^2;           end
   if strcmp(act_fn,'poslin'),  y = ones(size(x));            end
   if strcmp(act_fn,'purelin'), y = ones(size(x));            end
end