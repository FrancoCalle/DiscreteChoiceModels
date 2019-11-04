%Discrete Choice Model Lab, logit:
%--------------------------------

%1. Define Characteristics of the market:
global Data Pack;

%Set seed
seed = RandStream('mt19937ar','Seed',101);
RandStream.setGlobalStream(seed);
CC = parula(100);



%% 1. Generate Data and True Parameters for the DCM Lab:

% 1.1. Define number of students and options in the model:

nStudents = 10000;
nOptions = 20;
nXX = 15;
Data.nStudents = nStudents;
Data.nOptions = nOptions;

% 1.2. Generate random independent variables

Data.XX = {};
for ii = 1:nXX
    Data.XX{ii} = randn(nStudents,nOptions); %Variable 1
end

Data.FeasibleSet = true(nStudents,nOptions); %Variable 1
Data.sample = true(nStudents,1);

% 1.3. Define True Parameters
allParameters = [];

% a. Add parameters for X variables
allParameters  = [allParameters randn(1,numel(Data.XX))];

% b. Add parameters for sigma and rho:            
allParameters = [allParameters, ...
                0.75,   ... Parameter RC1
                0.75,   ... Parameter RC2
                0.15    ... Parameter RhoRC
                ];

Pack.allParameters = allParameters;

nXX = numel(Data.XX);
Pack.ThetaXj(1:nXX)=1;


Pack.Model = 'Model 1'; %This define the set of parameters we're gonna use and the Model to estimate

parameters = dcmLab.setupParameters();

% 1.5. Generate Choices
fakedata = true;
gradient = false;

dcmLab.generateFakeChoices(parameters, false); %Estimate fake data

[~, dL]=dcmLab.utilityFunction(parameters , true); % Four args if 'Model 1 Exploded Logit'

% Check if objective function is working:
passedUtilityFunction = @(theta) dcmLab.utilityFunction(theta, true);
[Q_total, Gradient]= dcmLab.objectiveFunction(parameters, passedUtilityFunction);



%% 2. Start model optimization:

%2.1. Check if gradients are well defined
passedUtilityFunction = @(parameters) dcmLab.utilityFunction(parameters,  true);
dcmLab.checkGradient(parameters,passedUtilityFunction)

%2.3. Check objective function
passedUtilityFunction = @(parameters) dcmLab.utilityFunction(parameters, true);
dcmLab.checkObjectiveFunction(parameters, passedUtilityFunction)

%2.3. Optimization, use multiple seeds to start optimization and look at
%parameter convergence
theta_init = randn(1,size(parameters,2));

%Run the model
passedUtilityFunction = @(theta) dcmLab.utilityFunction(theta, true);
[theta_quasinewton,fobj1,~,~]=dcmLab.estimationNLP(theta_init, passedUtilityFunction);

plot(min(parameters):0.01:max(parameters), min(parameters):0.01:max(parameters), 'Color', CC(7,:), 'LineWidth', 2)
hold on 
plot(theta_quasinewton, parameters(1:nXX), 'O', 'MarkerEdgeColor', CC(4,:), 'MarkerFaceColor', CC(4,:))
ylim([-2 2])
xlim([-2 2])
box on 
grid on

%% 4. Optimization (Adam) with different weights in the objective function:

%4.1. Define batch size:
batchSize = 2^10;
nIter = 120;
%4.2. What happens when we start from the true values?
initial_values  = theta_init(1,:); %initial_values  = theta_init(s,:);

passedUtilityFunction = @(theta) dcmLab.utilityFunction(theta, true);
[theta_estimated_Adam, allTheta]= dcmLab.optimizationAlgorithm(initial_values, batchSize, 'Adam', nIter, passedUtilityFunction);

figure(2)
kk = nIter; %Number of iterations
scatter(kk.*ones(1,nXX),parameters(1:nXX), 's', 'LineWidth', 1.8, 'MarkerEdgeColor', [.17 .17 .17], 'MarkerFaceColor', 'none')
hold on
HH = plot(allTheta(1:kk,1:nXX), 'LineWidth', 1.8);
for ii = 1:nXX 
    HH(ii,1).Color = [CC(ii*6,:) .8];
end
box on
grid on
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 8, 6], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])
legend({'True Parameters'},'Location','northeast')



%% Estimate changes in utility by eliminating options:

utilities0  = dcmLab.simulateUtilities(parameters);
[~,choice0] = max(utilities0,[],2);

avg_utility = nanmean(utilities0,1);
[~,p] = sort(avg_utility,'descend');
Ranking = 1:length(avg_utility);
Ranking(p) = Ranking;

TrueFeasible = Data.FeasibleSet;

%1. Policy: Remove worst 5 programs:
%===================================
nKilled = 5;
Policy = ~(Ranking > nOptions - nKilled);
Data.FeasibleSet = logical(double(Data.FeasibleSet).*double(Policy));

utilities1  = dcmLab.simulateUtilities(parameters);
[~,choice1] = max(utilities1,[],2);

histogram(utilities0(bsxfun(@eq, 1:size(utilities0,2), choice0)), 'EdgeColor', CC(10,:), 'FaceColor', CC(10,:), 'FaceAlpha', 0.4)
hold on
histogram(utilities1(bsxfun(@eq, 1:size(utilities1,2), choice1)), 'EdgeColor', CC(50,:), 'FaceColor', CC(50,:), 'FaceAlpha', 0.4)
legend('Before Policy','After Policy')
grid on
box on
