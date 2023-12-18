%%%%% coupled PDE: Belousov-Zhabotinsky equation %%%%%

% % % % % Example 3: Belousov-Zhabotinsky (reaction-diffusion with three species)
% % % % %  
% % % % %         u = spin('BZ');
% % % % %  
% % % % %      solves the Belousov-Zhabotinsky equation
% % % % %  
% % % % %         u_t = 1e-5*diff(u,2) + u + v - u*v - u^2,
% % % % %         v_t = 2e-5*diff(v,2) + w - v - u*v,
% % % % %         w_t = 1e-5*diff(w,2) + u - w
% % % % %  
% % % % %      on [-1 1] from t=0 to t=30, with initial condition
% % % % %  
% % % % %         u0(x) = exp(-100*(x+.5)^2),
% % % % %         v0(x) = exp(-100*x^2),
% % % % %         w0(x) = exp(-100*(x-.5)^2).
 
clear all; close all; clc;

% grid size - x
s = 256;
dom = [-1 1];
x = linspace(dom(1),dom(2),s+1);

% grid size - t
steps = 1;
tspan = linspace(0,1,steps+1);
dt = tspan(2) - tspan(1);

% Belousov-Zhabotinsky equations
S = spinop(dom, tspan);
S.lin = @(u,v,w)[1e-5*diff(u,2);2e-5*diff(v,2);1e-5*diff(w,2)];
S.nonlin = @(u,v,w)[u+v-u.*v-u.^2;w-v-u.*v;u-w];

% initial condition
L = 50;
u0 = chebfun(@(x) exp(-100*(x+.5)^2), dom);
v0 = chebfun(@(x) exp(-100*x^2), dom);
w0 = chebfun(@(x) exp(-100*(x-.5)^2), dom);
S.init = [u0;v0;w0];

uvw = spin(S, s, dt, 'plot', 'off');

input_u = u0(x);
input_v = v0(x);
input_w = w0(x);
output_u = uvw{1}.values;
output_v = uvw{2}.values;
output_w = uvw{3}.values;

figure; hold;
plot(input_u);
plot(input_v);
plot(input_w);
title('input')

figure; hold;
plot(output_u);
plot(output_v);
plot(output_w);
title('output')