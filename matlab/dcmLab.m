%{
Author: Franco Calle
%}

classdef dcmLab
    
    properties
    end
    
    methods (Static)
        
        % 1. Set model Parameters:
        
        function parameters = setupParameters()
            
            global Pack
            
            parameters = Pack.allParameters(Pack.ThetaXj==1);
            
            if contains(Pack.Model,'RC')
                
                parameters = [parameters, ...
                    Pack.allParameters(Pack.ThetaSigma==1) ...
                    Pack.allParameters(Pack.ThetaRho==1) ...
                    ];
            end
        end
        
        % 1. Generate Fake Data:
        
        function generateFakeChoices(parameters, da)
            
            global Data Pack;
            
            sample = Data.sample;
            nObs = sum(sample);
            nOptions = size(Data.XX{1},2);
            NXX = numel(Data.XX);
            %Unpack Data:
            
            XX = {};
            for nx = 1:NXX
                XX{nx} = Data.XX{nx}(sample,:);
            end
            
            FeasibleSet = Data.FeasibleSet(sample,:);
            
            switch Pack.Model
                
                %Model 1:
                %--------
                case {"Model 1","Model 1 Exploded Logit"}
                    
                    probabilities = NaN(nObs, nOptions);
                    
                    parfor s = 1:nObs
                        
                        f = FeasibleSet(s,:);
                        utilities = zeros(1,nOptions);
                        for nx = 1:NXX
                            %if nx ~= Data.nXXcontinuous+1
                                utilities = utilities + XX{nx}(s,:) * parameters(nx) ;
                            %end
                        end
                        
                        %Estimate Utilities and Probabilities considering
                        uni = rand(1,nOptions);
                        E   = gevinv(uni);
                        utilities = (utilities + E);
                        utilitiesVi = utilities;
                        probabilitiesViAll = zeros(1,nOptions);
                        probabilitiesViAll(:,f) = exp(utilitiesVi(:,f) - max(utilitiesVi(:,f), [],2))./sum(exp(utilitiesVi(:,f) - max(utilitiesVi(:,f), [],2)),2);
                        
                        probabilities(s,:) = probabilitiesViAll;
                    end
                    
                case {"Model 1 - RC", "Model 1 Exploded Logit - RC"}
                    
                    ScoresOption = Data.ScoreOption;
                    nNodes = size(Data.q_Weights,1);
                    probabilities = NaN(nObs, nOptions);
                    q_NodesList = Data.q_Nodes;
                    q_Weights = Data.q_Weights;
                    
                    % Gradient transformation:
                    xSigma = parameters(Pack.ThetaSigma==1);
                    ThetaSigma= log(1+exp(xSigma));

                    xRho = parameters(Pack.ThetaRho==1);
                    ThetaRho=tanh(xRho); 

                    RC1 = ThetaSigma(1);
                    RC2 = ThetaSigma(2);
                    rhoRC = ThetaRho(1);

                    P = [RC1 0;  ...
                        RC2*rhoRC RC2*sqrt(1-rhoRC^2)]';
                    
                    ViWeights=q_Weights;
                    
                    parfor s = 1:nObs
                        
                        f = FeasibleSet(s,:);
                        utilities = zeros(1,nOptions);
                        for nx = 1:NXX
                            utilities = utilities + XX{nx}(s,:) * parameters(nx) ;
                        end
                        
                        q_Nodes = q_NodesList{s};
                        ViNodes = P*q_Nodes;  %q_Nodes change by student
                        
                        %Estimate Utilities and Probabilities considering
                        %random coeficients or not:
                        talentNode = ScoresOption * ViNodes(1:2,:);
                        
                        uni = rand(1,nOptions);
                        E   = gevinv(uni);
                        utilities = (utilities + E);
                        utilitiesVi = utilities + talentNode';
                        probabilitiesViAll = zeros(nNodes,nOptions);
                        probabilitiesViAll(:,f) = exp(utilitiesVi(:,f) - max(utilitiesVi(:,f), [],2))./sum(exp(utilitiesVi(:,f) - max(utilitiesVi(:,f), [],2)),2);
                        
                        probabilities(s,:) = probabilitiesViAll'*ViWeights;
                    end
                    
                otherwise
                    warning('Model was not specified or does not exist.')
            end
            
            if da == true
                oPref = rand(nObs,nOptions);
                
                %Slots by option
                share = rand(nOptions,1);
                share = share/sum(share)*round(nObs*0.7);
                slots = round(share);
                slots(end) = nObs;
                
                ChoiceIndex = DA(probabilities, oPref, slots);
                Data.ChoiceIndex = ChoiceIndex;
                Data.FeasibleSet = exPostFeasible(oPref,ChoiceIndex);
                
                %Get Next Option:
                probabilities = probabilities.*double(Data.FeasibleSet);
                probabilities(bsxfun(@eq, 1:nOptions,Data.ChoiceIndex)) = 0;
                [~,NextChoiceIndex] = max(probabilities,[],2);
                NextChoiceIndex(sum(probabilities,2) == 0) = NaN;
                Data.NextChoiceIndex = NextChoiceIndex;
                
            elseif da == false
                
                [~,ChoiceIndex] = max(probabilities,[],2);
                Data.ChoiceIndex=ChoiceIndex;
                probabilities(bsxfun(@eq, 1:nOptions,Data.ChoiceIndex)) = 0;
                [~,NextChoiceIndex] = max(probabilities,[],2);
                Data.NextChoiceIndex = NextChoiceIndex;
                
            end
        end
        
        % 3. Defining Utility Function:
        
        function [probabilityChoice, dL]=utilityFunction(parameters, gradient)
            
            global Data Pack;
            
            sample = Data.sample;
            nObs = sum(sample);
            nOptions = size(Data.XX{1},2);
            NXX = numel(Data.XX);
            
            FeasibleSet = Data.FeasibleSet(sample,:);
            ChoiceIndex = Data.ChoiceIndex(sample);
            
            %Unpack Data:
            XX = {};
            for nx = 1:NXX
                XX{nx} = Data.XX{nx}(sample, :); % Cell array containing X characteristics
            end
            
            utilities = zeros(nObs, nOptions);
            for ii = 1:NXX
                utilities = utilities + XX{ii} * parameters(ii);
            end
            
            %Estimate Utilities and Probabilities
            utilities = utilities.*FeasibleSet;
            utilities(utilities==0) = -Inf;
            probabilities = bsxfun(@rdivide, exp(utilities), sum(exp(utilities),2));
            
            idx = sub2ind(size(probabilities), 1:nObs, ChoiceIndex');
            probabilityChoice = probabilities(idx)';
            
            dL = NaN(nObs,NXX);
            
            if gradient == true
                for nx = 1:NXX
                    dLi = ((XX{nx} - sum(XX{nx}.*probabilities,2)));
                    dL(:,nx) = dLi(idx);
                end
            else
                dL = nan;
            end
            
            
        end
        
        
        function [probabilityChoice, dL, probabilityJoint, dL_Joint]=utilityFunctionRC(parameters, gradient)
            
            global Data Pack;
            
            sample = Data.sample;
            nObs = sum(sample);
            nOptions = size(Data.XX{1},2);
            nNodes = size(Data.q_Weights,1);
            nParameters  = size(parameters,2);
            NXX = numel(Data.XX);
            %Unpack Data:
            XX = {};
            for nx = 1:NXX
                XX{nx} = Data.XX{nx}(sample, :); % Cell array containing X characteristics
            end
            
            FeasibleSet = Data.FeasibleSet(sample,:);
            ScoresOption = Data.ScoreOption;
            
            q_NodesList = Data.q_Nodes;
            q_Weights = Data.q_Weights;
            
            % Transforming elements for sigma and rho:
            transformGradient=ones(nParameters,1);

            xSigma = parameters(Pack.ThetaSigma==1);
            ThetaSigma= log(1+exp(xSigma));
            transformGradient(Pack.ThetaSigma==1)=(exp(xSigma))./(1+exp(xSigma));

            xRho = parameters(Pack.ThetaRho==1);
            ThetaRho=tanh(xRho); 
            transformGradient(Pack.ThetaRho==1)=1-tanh(xRho)^2;
            
            RC1 = ThetaSigma(1);
            RC2 = ThetaSigma(2);
            rhoRC = ThetaRho(1);
            
            P = [RC1 0;  ...
                RC2*rhoRC RC2*sqrt(1-rhoRC^2)]';
            
            % Gradients for Cholesky Decomposition
            dP = {};
            dP{1}=[1 0;
                0 0]';
            dP{2}=[0 0
                rhoRC sqrt(1-rhoRC^2)]';
            dP{3} =[0 0;
                RC2 -RC2*rhoRC/sqrt(1-rhoRC^2)]';
            
            dtemp_rc = {};
            
            ViWeights=q_Weights;
            
            ChoiceIndex = Data.ChoiceIndex(sample);
            NextChoiceIndex = Data.NextChoiceIndex(sample);
            
            switch Pack.Model
                
                %Model 1:
                %--------
                case "Model 1 - RC"
                    
                    dL_temp = NaN(nObs,size(parameters,2));
                    
                    probabilityChoice = NaN(nObs, 1);
                    
                    parfor s = 1:nObs
                        
                        f = FeasibleSet(s,:);
                        utilities =  XX{1}(s,:) * parameters(1) + ...
                            XX{2}(s,:) * parameters(2)  + ...
                            XX{3}(s,:) * parameters(3) + ...
                            XX{4}(s,:) * parameters(4) ;
                        
                        q_Nodes = q_NodesList{s};
                        ViNodes=P*q_Nodes;
                        
                        %Estimate Utilities and Probabilities considering
                        %talents or not:
                        talentNode = ScoresOption * ViNodes(1:2,:);
                        
                        utilitiesVi = utilities + talentNode';
                        
                        probabilitiesViAll = zeros(nNodes,nOptions);
                        probabilitiesViAll(:,f) = exp(utilitiesVi(:,f) - max(utilitiesVi(:,f), [],2))./sum(exp(utilitiesVi(:,f) - max(utilitiesVi(:,f), [],2)),2);
                        
                        
                        pickIndex = ChoiceIndex(s);
                        prChoiceVi = probabilitiesViAll(:,pickIndex);
                        probabilityChoice(s,1)= prChoiceVi'*ViWeights;
                        
                        
                        % Compute the gradient
                        if gradient == true
                            
                            dLi_temp = NaN(1,size(parameters,2));
                            
                            dLi = (XX{1}(s,pickIndex).*prChoiceVi - sum(XX{1}(s,:).*probabilitiesViAll,2).*prChoiceVi)'*ViWeights;
                            dLi_temp(1) = dLi;
                            dLi = (XX{2}(s,pickIndex).*prChoiceVi - sum(XX{2}(s,:).*probabilitiesViAll,2).*prChoiceVi)'*ViWeights;
                            dLi_temp(2) = dLi;
                            dLi = (XX{3}(s,pickIndex).*prChoiceVi - sum(XX{3}(s,:).*probabilitiesViAll,2).*prChoiceVi)'*ViWeights;
                            dLi_temp(3) = dLi;
                            dLi = (XX{4}(s,pickIndex).*prChoiceVi - sum(XX{4}(s,:).*probabilitiesViAll,2).*prChoiceVi)'*ViWeights;
                            dLi_temp(4) = dLi;
                            
                            dxtemp=ScoresOption(:,1:2);
                            dNodes=dP_1*q_Nodes;
                            dtemp=dxtemp*dNodes ;
                            dPrVi=(prChoiceVi.*(dtemp(pickIndex,:)'-sum(dtemp'.*probabilitiesViAll,2)));
                            dLi_temp(5)=dPrVi'*ViWeights;
                            
                            dxtemp=ScoresOption(:,1:2);
                            dNodes=dP_2*q_Nodes;
                            dtemp=dxtemp*dNodes ;
                            dPrVi=(prChoiceVi.*(dtemp(pickIndex,:)'-sum(dtemp'.*probabilitiesViAll,2)));
                            dLi_temp(6)=dPrVi'*ViWeights;
                            
                            dxtemp=ScoresOption(:,1:2);
                            dNodes=dP_3*q_Nodes;
                            dtemp=dxtemp*dNodes ;
                            dPrVi=(prChoiceVi.*(dtemp(pickIndex,:)'-sum(dtemp'.*probabilitiesViAll,2)));
                            dLi_temp(7)=dPrVi'*ViWeights;
                            
                            dL_temp(s,:) = dLi_temp;
                            
                        end
                    end
                    
                    dL_temp = dL_temp./probabilityChoice;
                    dL=dL_temp;
                    
                    
                    
                    %Model 1 Exploded Logit:
                    %-----------------------
                case "Model 1 Exploded Logit - RC"
                    
                    dL_temp = NaN(nObs,size(parameters,2));
                    dL_Joint_temp = NaN(nObs,size(parameters,2));
                    probabilityChoice = NaN(nObs, 1);
                    probabilityJoint = NaN(nObs, 1);
                    
                    parfor s = 1:nObs
                        
                        pickIndex = ChoiceIndex(s);
                        pickIndexNext = NextChoiceIndex(s);
                        
                        pickIndexNext_nan = false;
                        
                        if isnan(pickIndexNext) % Workarround to continue with parallel iteration:
                            pickIndexNext_nan = true;
                            pickIndexNext = nOptions;
                        end
                        
                        f = FeasibleSet(s,:);
                        f_next = f;
                        f_next(pickIndex) = false;
                        
                        utilities = zeros(1, nOptions);
                        for nx = 1:NXX
                            %if nx ~= Data.nXXcontinuous+1
                            utilities = utilities + XX{nx}(s,:) * parameters(nx);
                            %end
                        end
                        
                        q_Nodes = q_NodesList{s};
                        ViNodes=P*q_Nodes;
                        
                        %Estimate Utilities and Probabilities considering
                        %talents or not:
                        talentNode = ScoresOption * ViNodes(1:2,:);
                        
                        utilitiesVi = utilities + talentNode';
                        
                        probabilitiesViAll = zeros(nNodes,nOptions);
                        probabilitiesViAll(:,f) = exp(utilitiesVi(:,f) - max(utilitiesVi(:,f), [],2))./sum(exp(utilitiesVi(:,f) - max(utilitiesVi(:,f), [],2)),2);
                        
                        probabilitiesViAllNext = zeros(nNodes,nOptions);
                        probabilitiesViAllNext(:,f_next) = exp(utilitiesVi(:,f_next) - max(utilitiesVi(:,f_next), [],2))./sum(exp(utilitiesVi(:,f_next) - max(utilitiesVi(:,f_next), [],2)),2);
                        
                        prChoiceVi      = probabilitiesViAll(:,pickIndex);
                        prChoiceViNext  = probabilitiesViAllNext(:,pickIndexNext);
                        
                        probabilityChoice(s,1)= prChoiceVi'*ViWeights;
                        %probabilityChoiceNext(s,1)= prChoiceViNext'*ViWeights;
                        
                        probabilityJoint(s,1)=(prChoiceVi.*prChoiceViNext)'*ViWeights;
                        
                        
                        % Compute the gradient
                        if gradient == true
                            
                            dLi  = NaN(nNodes, nParameters);
                            dLiNext  = NaN(nNodes, nParameters);
                            dLiJoint  = NaN(nNodes, nParameters);
                            
                            dLi_temp = NaN(1,size(parameters,2));
                            dLiJoint_temp = NaN(1,size(parameters,2));
                            
                            %1. Derivative of first option Pr
                            % Compute the derivative across all XX associated to beta:
                            for nx = 1:NXX
                                %if nx ~= Data.nXXcontinuous+1
                                    jj = nx;
                                    dLi(:,jj) = (XX{nx}(s,pickIndex).*prChoiceVi - sum(XX{nx}(s,:).*probabilitiesViAll,2).*prChoiceVi);
                                    dLi_temp(jj) = dLi(:,jj)'*ViWeights; % Parameters Beta
                                    dLiNext(:,jj) = (XX{nx}(s,pickIndexNext).*prChoiceViNext - sum(XX{nx}(s,:).*probabilitiesViAllNext,2).*prChoiceViNext);
                                    dLiJoint(:,jj) = (dLi(:,jj).*prChoiceViNext + dLiNext(:,jj).*prChoiceVi); % Beta parameters
                                    dLiJoint_temp(jj) = dLiJoint(:,jj)'*ViWeights; % Average joint probability:
                                %end
                            end
                            
                            %Derivative for random coefficients
                            for nrc = 1:numel(dP)
                                jj = NXX + nrc;
                                dtemp_rc=ScoresOption*dP{nrc}*q_Nodes;
                                dLi(:,jj)=(prChoiceVi.*(dtemp_rc(pickIndex,:)'-sum(dtemp_rc'.*probabilitiesViAll,2)));
                                dLi_temp(jj) = dLi(:,jj)'*ViWeights; % Random Coefficients
                                dLiNext(:,jj)=(prChoiceViNext.*(dtemp_rc(pickIndexNext,:)'-sum(dtemp_rc'.*probabilitiesViAllNext,2)));
                                dLiJoint(:,jj) = (dLi(:,jj).*prChoiceViNext + dLiNext(:,jj).*prChoiceVi); % Random Coef
                                dLiJoint_temp(jj) = dLiJoint(:,jj)'*ViWeights; % Random coefficients
                            end
                            
                            dL_temp(s,:) = dLi_temp;
                            dL_Joint_temp(s,:) = dLiJoint_temp;
                            
                            if pickIndexNext_nan == true % Fill Joint probability and derivative with:
                                dL_Joint_temp(s,:) = 0;
                                probabilityJoint(s,1) = NaN;
                            end
                            
                        end
                    end
                    
                    dL_temp = repmat(transformGradient',nObs,1).*dL_temp./probabilityChoice;
                    dL=dL_temp;
                    
                    dL_Joint_temp = dL_Joint_temp./probabilityJoint;
                    dL_Joint = dL_Joint_temp;
                    dL_Joint = repmat(transformGradient',nObs,1).*dL_Joint;
                    
                otherwise
                    warning('Model was not specified or does not exist.')
            end
            
        end
        
        
        % 4. Defining Objective Function
        function [Q_total, Gradient, Q_Choice, GradientChoice, Q_Joint, GradientJoint]=objectiveFunction(parameters, passedFunction)
            
            global Data Pack;
            
            if ~contains(Pack.Model, "Exploded")
                
                sample = Data.sample;
                nObs = size(sample,1);
                
                [probabilityChoice, dL]=passedFunction(parameters);
                
                Q_Choice= -sum(log(probabilityChoice))/nObs;
                
                if nargout > 1
                    GradientChoice= -sum(dL)/nObs;
                end
                
                Q_total = Q_Choice;
                Gradient = GradientChoice;
                
            elseif contains(Pack.Model, "Exploded")
                
                sample = Data.sample;
                nObs = size(sample,1);
                nObsJoint = sum(~isnan(Data.NextChoiceIndex(Data.sample)));
                
                [probabilityChoice, dL, probabilityJoint, dL_Joint]=passedFunction(parameters);
                
                Q_Choice= -nansum(log(probabilityChoice))/nObs;
                Q_Joint = -nansum(log(probabilityJoint))/nObsJoint;
                
                if nargout > 1
                    GradientChoice= -nansum(dL)/nObs;
                    GradientJoint = -nansum(dL_Joint)/nObsJoint;
                end
                
                w_choice = Pack.w_choice;
                w_joint = Pack.w_joint;
                
                Q_total = w_choice*Q_Choice + w_joint*Q_Joint;
                Gradient = w_choice*GradientChoice + w_joint*GradientJoint;
            end
            
        end
        
        % 5. Check gradients of utility function:
        function checkGradient(parameters, passedUtilityFunction)
            
            global Data Pack;
            
            nStudents = Data.nStudents;
            
            theta=parameters;step=10^-6;
            NUM_dPrChoice=NaN(1,length(theta));
            
            for i=1:length(theta)
                theta1=theta;
                theta2=theta;
                
                theta1(i)=theta(i)-step/2;
                theta2(i)=theta(i)+step/2;
                
                [PrChoice1,~]=passedUtilityFunction(theta1);
                [PrChoice2,~]=passedUtilityFunction(theta2);
                
                Q_Choice1 = -nansum(log(PrChoice1))/nStudents;
                Q_Choice2 = -nansum(log(PrChoice2))/nStudents;
                
                NUM_dPrChoice(:,i)=(Q_Choice2-Q_Choice1)/step;
                
                fprintf('Probability change:  Parameter %3.0f \n',i )
            end
            
            if contains(Pack.Model, "Exploded")
                
                
                [~, ~, ~, dPrChoice, ~, ~]=dcmLab.objectiveFunction(theta, passedUtilityFunction);
                dif=abs((NUM_dPrChoice(:,1:i)-dPrChoice(:,1:i)));
                difp=abs((NUM_dPrChoice(:,1:i)-dPrChoice(:,1:i)))./dPrChoice(:,1:i);
                
            else
                
                [~, dPrChoice]=dcmLab.objectiveFunction(theta, passedUtilityFunction);
                dif=abs((NUM_dPrChoice(:,1:i)-dPrChoice(:,1:i)));
                difp=abs((NUM_dPrChoice(:,1:i)-dPrChoice(:,1:i)))./dPrChoice(:,1:i);
                
            end
            
            for ii=1:i
                fprintf('%3.0f -  : Levels %13.10f Percent %13.10f \n',[ii max(dif(:,ii)) max(difp(:,ii))])
            end
        end
        
        function checkGradientJoint(parameters, passedUtilityFunction)
            
            global Data;
            
            %nStudents = Data.nStudents;
            nStudentsJoint = sum(~isnan(Data.NextChoiceIndex(Data.sample)));
            
            theta=parameters;step=10^-6;
            NUM_dPrJoint=NaN(1,length(theta));
            
            for i=1:length(theta)
                theta1=theta;
                theta2=theta;
                
                theta1(i)=theta(i)-step/2;
                theta2(i)=theta(i)+step/2;
                
                [~,~,PrJoint1,~]=passedUtilityFunction(theta1);
                [~,~,PrJoint2,~]=passedUtilityFunction(theta2);
                
                Q_Joint1 = -nansum(log(PrJoint1))/nStudentsJoint;
                Q_Joint2 = -nansum(log(PrJoint2))/nStudentsJoint;
                
                NUM_dPrJoint(:,i)=(Q_Joint2-Q_Joint1)/step;
                
                fprintf('Joint probability change: Parameter %3.0f \n',i )
            end
            [~, ~, ~, ~, ~, dPrJoint]=dcmLab.objectiveFunction(theta, passedUtilityFunction);
            dif=abs((NUM_dPrJoint(:,1:i)-dPrJoint(:,1:i)));
            difp=abs((NUM_dPrJoint(:,1:i)-dPrJoint(:,1:i)))./dPrJoint(:,1:i);
            for ii=1:i
                fprintf('%3.0f -  : Levels %13.10f Percent %13.10f \n',[ii max(dif(:,ii)) max(difp(:,ii))])
            end
        end
        
        % 6. Check objective function:
        function checkObjectiveFunction(parameters,passedFunction)
            
            global Pack
            
            theta=parameters;step=10^-7;
            NUM_Grad = NaN(size(theta,2),1);
            
            for i=1:length(theta) %80:90 %
                theta0=theta;
                theta1=theta;
                
                theta0(i)=theta(i)-step/2;
                theta1(i)=theta(i)+step/2;
                
                if contains(Pack.Model, "Exploded")
                    [~, ~, ~, ~, Q0, ~]=dcmLab.objectiveFunction(theta0,passedFunction);
                    [~, ~, ~, ~, Q1, ~]=dcmLab.objectiveFunction(theta1,passedFunction);
                else
                    [Q0, ~]=dcmLab.objectiveFunction(theta0,passedFunction);
                    [Q1, ~]=dcmLab.objectiveFunction(theta1,passedFunction);
                end
                
                NUM_Grad(i,1)=(Q1-Q0)/step;
                
                fprintf('Overall change in objective function: Parameter %3.0f \n',i )
            end
            
            if contains(Pack.Model, "Exploded")
                [~, ~, ~, ~, ~, Gradient]=dcmLab.objectiveFunction(theta,passedFunction);
            else
                [~,Gradient]=dcmLab.objectiveFunction(theta,passedFunction);
            end
            
            
            dif=abs((NUM_Grad(1:i)-Gradient(1:i)'));
            difp=abs((NUM_Grad(1:i)-Gradient(1:i)'))./Gradient(1:i)';
            
            for ii=1:i
                fprintf('%3.0f -  : Levels %13.10f Percent %13.10f \n',[ii max(dif(ii)) max(difp(ii))])
            end
        end
        
        % 7. Parameter Estimation using matlab built-in optimizers
        
        function [x1,fobj1,flag1,output1] = estimationNLP(parameters, passedUtilityFunction)
            
            % Define Optimization Options:
            
            options = optimoptions(@fminunc,    'MaxIter',2000,...
                'MaxFunEvals', 2000, ...
                'HessUpdate', 'bfgs', ...
                'Display','iter', ...
                'SpecifyObjectiveGradient',true, ...
                'FiniteDifferenceType','central', ...
                'TolFun',10^-8, ...
                'StepTolerance',10^-8,...
                'FunctionTolerance',10^-8, ...
                'OptimalityTolerance',10^-8,...
                'TolX', 10^-8, ...
                'UseParallel', true, ... %'CheckGradients',true, ...
                'Algorithm','quasi-newton');
            
            passedObjectiveFunction = @(parameters) dcmLab.objectiveFunction(parameters, passedUtilityFunction);
            [x1,fobj1,flag1,output1]=fminunc(passedObjectiveFunction,parameters,options);
            
        end
        
        % 8. Gradient Descent Optimization:
        
        % 8.1. Assign Minibatches:
        
        function batchAssigned = assignMinibatch(batchSize)
            
            global Data;
            
            nObs = Data.nStudents;
            randomOrder = randperm(nObs);
            numCompleteMinibatches = floor(nObs/batchSize);
            batchAssigned = NaN(nObs, 1);
            for k = 1:numCompleteMinibatches
                lo = (k-1) * batchSize + 1;
                up = (k * batchSize) ;
                studentMiniBatchIndex = randomOrder(lo:up);
                batchAssigned(studentMiniBatchIndex) = k;
                if rem(nObs,batchSize) ~= 0
                    lo = numCompleteMinibatches * batchSize;
                    studentMiniBatchIndex = randomOrder(lo:end);
                    batchAssigned(studentMiniBatchIndex) = k;
                end
            end
        end
        
        % 9.4. Optimization algorithm
        
        function [Theta, thetaAll]= optimizationAlgorithm(initial_seed, batchSize, step, maxiter, passedFunction)
            
            global Data;
            
            nObs = Data.nStudents;
            Theta = initial_seed;
            
            if isempty(batchSize)
                batchID = true(nObs);
            else
                batchID = dcmLab.assignMinibatch(batchSize);
            end
            
            nBatches = max(batchID);
            
            switch step
                case 'Adam'
                    
                    alpha=0.1;
                    beta1=0.90;
                    beta2=0.99;
                    eps=10^-4;
                    decay=0.999999; % No decay
                    
                    % Initialize inputs for Adam (momentum+RMSprop)
                    VdL=0;
                    SdL=0;
                    
                    itt=1; epoc=1;Qo=1000;
                    Traj=NaN(nBatches*100,4);
                    Gradient = 1;
                    dif = 1;
                    thetaAll = [];
                    while max(abs(Gradient))>10^-8 && itt < maxiter %&& dif>10^-100 %&& alpha>10^-100 %&& epoc < 1000
                        for b=1:nBatches
                            if nBatches>1
                                Data.sample=(batchID==b);
                            end
                            
                            [Q,Gradient]=dcmLab.objectiveFunction(Theta, passedFunction);
                            
                            VdL=beta1*VdL+(1-beta1)*Gradient;
                            SdL=beta2*SdL+(1-beta2)*(Gradient.^2);
                            Theta = Theta-alpha*(VdL./(sqrt(SdL)+eps));
                            thetaAll = [thetaAll; Theta];
                            if sum(Data.sample) == nObs
                                dif=abs((Qo-Q)/Qo);
                            else
                                dif = Qo-Q;
                            end
                            Qo=Q;
                            %fprintf(' %6.0f :  the objetive function is %12.9f and max dL is %12.9f  and difference in parameters %11.9f\n',[itt Q max(abs(Gradient)) max(abs(dif)) ])
                            Traj(itt,:)=[itt Q max(abs(Gradient)) max(abs(dif))];
                            if itt==round(itt/100)*100
                                figure(1)
                                hold on
                                pause(.1)
                                plot(1:itt-1,Traj(1:itt-1,2),'b','LineWidth',1)
                            end
                            itt=itt+1;
                        end
                        fprintf(' %6.0f :  the objetive function is %12.9f and max dL is %12.9f  and difference in parameters %11.9f\n',[itt Q max(abs(Gradient)) max(abs(dif)) ])
                        epoc=epoc+1;
                        alpha=alpha*decay^epoc;
                        if sum(isnan(Gradient))>1
                            Gradient=1;
                            continue
                        end
                    end
                    
                    figure(1)
                    plot(1:itt-1,Traj(1:itt-1,2),'b','LineWidth',1)
                    
                    hold on
                    plot(1:itt-1,Traj(1,2)*ones(itt-1,1),':k','LineWidth',1)
                    plot(1:itt-1,Traj(itt-1,2)*ones(itt-1,1),':k','LineWidth',1)
                    plot(1:itt-1,mean(Traj(1:itt-1,2))*ones(itt-1,1),':r','LineWidth',1)
                    title("Adam Optimizer with " + num2str(nBatches) + " mini-batches")
                    ylim([0 0.1])
                    xlim([0 1000])
                    
                    
                case {'Gradient Descent',[]}
                    
                    alpha=0.01;
                    
                    itt=1; epoc=1;Qo=1000;
                    Traj=NaN(nBatches*100,4);
                    Gradient = 1;
                    dif = 1;
                    while max(abs(Gradient))>0.00001 && dif>0.0001 && epoc < 100
                        for b=1:nBatches
                            if nBatches>1
                                Data.sample=(batchID==b);
                            end
                            
                            [Q,Gradient]=dcmLab.objectiveFunction(Theta, passedFunction);
                            
                            Theta=Theta - alpha*Gradient;
                            if sum(Data.sample) == nObs
                                dif=abs((Qo-Q)/Qo);
                            else
                                dif = Qo-Q;
                            end
                            Qo=Q;
                            Traj(itt,:)=[itt Q max(abs(Gradient)) max(abs(dif))];
                            if itt==round(itt/100)*100
                                figure(1)
                                hold on
                                pause(.1)
                                plot(1:itt-1,Traj(1:itt-1,2),'b','LineWidth',1)
                            end
                            itt=itt+1;
                        end
                        fprintf(' %6.0f :  the objetive function is %12.9f and max dL is %12.9f  and difference in parameters %11.9f\n',[itt Q max(abs(Gradient)) max(abs(dif)) ])
                        epoc=epoc+1;
                        if sum(isnan(Gradient))>1
                            Gradient=1;
                            continue
                        end
                    end
                    
                    figure(1)
                    title("Gradient Descent with " + num2str(nBatches) + " mini-batches")
                    plot(1:itt-1,Traj(1:itt-1,2),'b','LineWidth',1)
                    
                    hold on
                    plot(1:itt-1,Traj(1,2)*ones(itt-1,1),':k','LineWidth',1)
                    plot(1:itt-1,Traj(itt-1,2)*ones(itt-1,1),':k','LineWidth',1)
                    plot(1:itt-1,mean(Traj(1:itt-1,2))*ones(itt-1,1),':r','LineWidth',1)
                    
            end
            
            
        end
        
        % 10. Simulate Utilities and probabilities:
        
        function Utilities = simulateUtilities(parameters)
            
            global Data Pack;
            
            NXX = numel(Data.XX);
            nObs = size(Data.XX{1},1);
            nOptions = size(Data.XX{1},2);
            
            %Unpack Data:
            
            XX = {};
            for nx = 1:NXX
                XX{nx} = Data.XX{nx}; % Cell array containing X characteristics
            end
            
            FeasibleSet = Data.FeasibleSet;
            ScoresOption = Data.ScoreOption;
            
            Utilities = NaN(nObs, nOptions);
            
            if contains(Pack.Model,'RC') %this is for the Random Coefficients
                
                q_NodeList = Data.q_Nodes;
                q_Weights = Data.q_Weights;
                
                RC1 = parameters(end-2);
                RC2 = parameters(end-1);
                rhoRC = parameters(end);
                
                P=[RC1 0;
                    rhoRC RC2]';
                
                ViWeights=q_Weights;
                
                for s = 1:nObs
                    
                    q_Nodes = q_NodeList{s};
                    ViNodes=P*q_Nodes;
                    
                    f = FeasibleSet(s,:);
                    utilities = zeros(1,nOptions);
                    for nx = 1:NXX
                        utilities = utilities + XX{nx}(s,:) * parameters(nx);
                    end
                    
                    %Estimate Utilities and Probabilities considering
                    %talents or not:
                    talentNode = ScoresOption * ViNodes(1:2,:);
                    
                    utilitiesVi = utilities(1,f) + talentNode(f,:)';
                    Utilities(s,f) = utilitiesVi'*ViWeights;
                    
                end
                
            else %Not random coefficients model
                
                for s = 1:nObs
                    
                    f = FeasibleSet(s,:);
                    utilities = zeros(1,nOptions);
                    for nx = 1:NXX
                        utilities = utilities + XX{nx}(s,:) * parameters(nx) ;
                    end
                    
                    %Estimate Utilities and Probabilities considering
                    utilitiesVi = utilities(1,f);
                    Utilities(s,f) = utilitiesVi;
                    
                end
                
            end
            
            
        end
        
        
    end
end

