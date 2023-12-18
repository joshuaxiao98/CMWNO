%%%%% coupled PDE: Gray-Scott equation %%%%%

% % % % Example 5: Gray-Scott equations (pulse splitting)
% % % %  
% % % %         u = spin('GS');
% % % %  
% % % %      solves the Gray-Scott equations
% % % %  
% % % %         u_t = diff(u,2) + 2e-2*(1-u) - u*v^2,
% % % %         v_t = 1e-2*diff(u,2) - 8.62e-2*v + u*v^2,
% % % %  
% % % %      on [-50 50] from t=0 to t=8000, with initial condition
% % % %  
% % % %         u0(x) = 1 - 1/2*sin(pi*(x-L)/(2*L))^100,
% % % %         v0(x) = 1/4*sin(pi*(x-L)/(2*L))^100, with L=50.
 
clear all; close all; clc;
gamma = 2.5;
tau = 7;
sigma = 7^(2);

% grid size - x
s = 1024;
dom = [0 10];
x = linspace(dom(1),dom(2),s+1);

% grid size - t
steps = 1;
tspan = linspace(0,0.1,steps+1);
dt = tspan(2) - tspan(1);
lambda1 = 2
lambda2 = 5

% Gray-Scott equations
S = spinop(dom, tspan);
S.lin = @(u,v)[diff(u,2);(1e-2)*diff(v,2)];
S.nonlin = @(u,v)[(2e-2)*(1-u)-lambda1*u.*v.^2;-(8.62e-2)*v+lambda2*u.*v.^2];

% initial condition: sin
%u0 = GRF1(s/2, 0, gamma, tau, sigma, "periodic");
u0 = 0.5*randnfun(0.5,[0,10],'trig');
v0 = GRF1(s/2, 0, gamma, tau, sigma, "periodic");
%v0 = 0.5*randnfun(0.2,[0,10],'trig');
% u0 = chebfun(@(x) 1 - 1/2*sin(pi*(x-L)/(2*L))^100, dom);
% v0 = chebfun(@(x)1/4*sin(pi*(x-L)/(2*L))^100, dom);

S.init = [u0;v0];

uv = spin(S, s, dt, 'plot', 'off');

input_u = u0(x);
input_v = v0(x);
output_u = uv{1}.values;
output_v = uv{2}.values;

figure; hold;
plot(input_u,LineWidth=2);
plot(input_v,LineWidth=2);
title('input')

figure; hold;
plot(output_u,LineWidth=2);
plot(output_v,LineWidth=2);
title('output')