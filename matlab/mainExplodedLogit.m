%Discrete Choice Model Lab:
%--------------------------

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

Data.ScoreOption = abs(randn(nOptions,2)); %This will work for the quadrature
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
Pack.ThetaXj=zeros(size(allParameters,2),1);
Pack.ThetaSigma=zeros(size(allParameters,2),1);
Pack.ThetaRho=zeros(size(allParameters,2),1);

nXX = numel(Data.XX);
Pack.ThetaXj(1:nXX)=1;
Pack.ThetaSigma(nXX+1:nXX+2)=1;
Pack.ThetaRho(nXX+3)=1;


Pack.Model = 'Model 1 Exploded Logit'; %This define the set of parameters we're gonna use and the Model to estimate
Pack.w_choice = .5;
Pack.w_joint = .5;

parameters = dcmLab.setupParameters();

% 1.4. Quadrature Points
K=6;
[Data.q_Nodes, Data.q_Weights ] = GHQuadInit(2, K );

nVar   = 2;
nNodes = 36;
All_Nodes ={};
for s = 1:nStudents
    All_Nodes{s} = randn(nVar,nNodes);
end

Data.q_Nodes = All_Nodes;
Data.q_Weights = ones(size(All_Nodes{1},2),1)./size(Data.q_Weights,1);

% 1.5. Generate Choices
fakedata = true;
gradient = false;

dcmLab.generateFakeChoices(parameters, false); %Estimate fake data

[~, dL, ~, dL_Joint]=dcmLab.utilityFunctionRC(parameters , true); % Four args if 'Model 1 Exploded Logit'

% Check if objective function is working:
passedUtilityFunction = @(theta) dcmLab.utilityFunctionRC(theta, true);
[Q_total, Gradient, Q_Choice, GradientChoice, Q_Joint, GradientJoint]= dcmLab.objectiveFunction(parameters, passedUtilityFunction);



%% 2. Start model optimization:

%2.1. Check if gradients are well defined
passedUtilityFunction = @(parameters) dcmLab.utilityFunctionRC(parameters,  true);
dcmLab.checkGradient(parameters,passedUtilityFunction)

%2.2. Check if gradients of joint probabilities are well defined:
passedUtilityFunction = @(parameters) dcmLab.utilityFunctionRC(parameters,  true);
dcmLab.checkGradientJoint(parameters,passedUtilityFunction)

%2.3. Check objective function
passedUtilityFunction = @(parameters) dcmLab.utilityFunctionRC(parameters, true);
dcmLab.checkObjectiveFunction(parameters, passedUtilityFunction)

%2.3. Optimization, use multiple seeds to start optimization and look at
%parameter convergence
nsim=10;
theta_init = randn(nsim,size(parameters,2));
theta_init(:,size(theta_init,2)-2:size(theta_init,2)) = abs(theta_init(:,size(theta_init,2)-2:size(theta_init,2))); %Parameters in P are always positive

theta_quasinewton = NaN(nsim, size(parameters,2));
fobj1 = NaN(nsim, 1);
time = NaN(nsim, 1);

for s=1:nsim
tic
passedUtilityFunction = @(theta) dcmLab.utilityFunctionRC(theta, true);
[theta_quasinewton(s,:),fobj1(s,:),~,~]=dcmLab.estimationNLP(theta_init(s,:), passedUtilityFunction);
time(s,1)=toc;
end

plot(min(parameters):0.01:max(parameters), min(parameters):0.01:max(parameters), 'Color', CC(7,:), 'LineWidth', 2)
hold on 
plot(theta_quasinewton(s,1:nXX), parameters(1:nXX), 'O', 'MarkerEdgeColor', CC(4,:), 'MarkerFaceColor', CC(4,:))
ylim([-2 2])
xlim([-2 2])
box on 
grid on

%% 3. Optimization with different weights in the objective function:

% What happens when we start from the true values?
initial_values  = parameters; %initial_values  = theta_init(s,:);

%3.1. Only Choice Pr
Pack.w_choice = 1;
Pack.w_joint = 0;

passedUtilityFunction = @(theta) dcmLab.utilityFunctionRC(theta, true);
[theta_estimated_1,~,~,~]=dcmLab.estimationNLP(initial_values, passedUtilityFunction);


figure(1) %Plot All Estimates
subplot(1,3,1)
scatter(parameters(1,1:4),theta_estimated_1(1,1:4), 'b')
refline
hold on
scatter(parameters(1,5:7),theta_estimated_1(1,5:7), 'r') 
hold on
%plot(x,y)
title(['Fit: Quasi Newton with wChoice = ' num2str(1) ' and wJoint = ' num2str(0)])
xlim([-0.2 1])
ylim([-0.2 1])
grid on


%% 4. Optimization (Adam) with different weights in the objective function:

%4.1. Define batch size:
batchSize = 2^10;

%4.2. What happens when we start from the true values?
initial_values  = theta_init(1,:); %initial_values  = theta_init(s,:);

%4.3. Only Choice Pr
Pack.w_choice = 0.5;
Pack.w_joint = 0.5;

passedUtilityFunction = @(theta) dcmLab.utilityFunctionRC(theta, true);
[theta_estimated_Adam, allTheta]= dcmLab.optimizationAlgorithm(initial_values, batchSize, 'Adam', 200, passedUtilityFunction);

figure(2)
kk = 200; %Number of iterations
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

%2. Recursively remove best 40 programs:
%=======================================

Data.FeasibleSet = TrueFeasible;

maxRemove = 10;
overallUtility = NaN(1,maxRemove);
overallDropout = NaN(1,maxRemove);

for i = 1:maxRemove
    nKilled = i;
    Policy = ~(Ranking > nOptions - nKilled);
    Data.FeasibleSet = logical(double(Data.FeasibleSet).*double(Policy));
    Data.FeasibleSet(:,end) = true;
    Data.FeasibleSet = logical(Data.FeasibleSet);
    utilitiesi  = dcmLab.simulateUtilities(parameters);
    [~,choicei] = max(utilitiesi,[],2);

    utilityPick = utilitiesi(bsxfun(@eq, 1:size(utilitiesi,2), choicei));
    overallUtility(i) = mean(utilityPick);
    overallDropout(i) = sum(choicei == nOptions)/nStudents;
end

UtilityDecrease = (overallUtility - overallUtility(1))./overallUtility(1)*100;

figure(1)
subplot(1,2,1)
plot(UtilityDecrease)
ylabel('Utility Percentage Decrease')
xlabel('Number of Programs Removed')


subplot(1,2,2)
plot(overallDropout)
ylabel('Drop-out Rate')
xlabel('Number of Programs Removed')


