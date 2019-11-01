classdef dcmLab
    
    properties
    end
    
    methods (Static)
        
        % 1. Set model Parameters:    
        
        function parameters = setupParameters()
            
            global Pack
            
            if Pack.Model == "Model 1" || Pack.Model == "Model 1 Exploded Logit"
                
                parameters = Pack.allParameters(Pack.ThetaXj==1);
                
            end
            
            talents = true;
            if talents == true
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
            nNodes = size(Data.q_Weights,1);
            NXX = numel(Data.XX);
            %Unpack Data:
            
            XX = {};
            for nx = 1:NXX
                XX{nx} = Data.XX{nx}(sample,:);
            end
            
            FeasibleSet = Data.FeasibleSet(sample,:);
            ScoresOption = Data.ScoreOption;

            q_NodesList = Data.q_Nodes;
            q_Weights = Data.q_Weights;

            RC1 = parameters(end-2);
            RC2 = parameters(end-1);
            rhoRC = parameters(end);

            P=[RC1 rhoRC;
                0 RC2];

            ViWeights=q_Weights;            
            
            switch Pack.Model

                %Model 1:
                %--------                
                case {"Model 1","Model 1 Exploded Logit"}
                    
                    probabilities = NaN(nObs, nOptions);

                    parfor s = 1:nObs
                        
                        f = FeasibleSet(s,:);
                        utilities = zeros(1,nOptions);
                        for nx = 1:NXX
                            utilities = utilities + XX{nx}(s,:) * parameters(nx) ;
                        end

                        q_Nodes = q_NodesList{s};
                        ViNodes=P*q_Nodes;  %q_Nodes change by student
                        
                        %Estimate Utilities and Probabilities considering
                        %talents or not:
                        talentNode = ScoresOption * ViNodes(1:2,:);

                        uni = rand(1,nOptions);
                        E   = gevinv(uni);
                        utilities = (utilities + E);
                        utilitiesVi = utilities + talentNode';
                        probabilitiesViAll = zeros(nNodes,nOptions);
                        probabilitiesViAll(:,f) = exp(utilitiesVi(:,f) - max(utilitiesVi(:,f), [],2))./sum(exp(utilitiesVi(:,f) - max(utilitiesVi(:,f), [],2)),2);

                        probabilities(s,:) = probabilitiesViAll'*ViWeights;
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
                    
                                        
                otherwise
                    warning('Model was not specified or does not exist.')
            end
            
        end
                                
        % 3. Defining Utility Function:

        function [probabilityChoice, dL]=utilityFunction(parameters, fakedata, gradient)

            global Data Pack;

            sample = Data.sample;
            nObs = sum(sample);
            nOptions = size(Data.x1,2); 

            %Unpack Data:
            
            x1 = Data.x1(sample,:);
            x2 = Data.x2(sample,:);
            x3 = Data.x3(sample,:);
            x4 = Data.x4(sample,:);
            Type = Data.Type(sample,:);
            FeasibleSet = Data.FeasibleSet(sample,:);


            if isempty(fakedata) || fakedata == false
                ChoiceIndex = Data.ChoiceIndex(sample);
            end
            
            switch Pack.Model

                %Model 1:
                %--------                
                case "Model 1"
                    
                    utilities =  x1 * parameters(1) + ...
                        x2 * parameters(2)  + ...
                        x3 * parameters(3) + ...
                        x4 * parameters(4) ;
                                        
                    %Estimate Utilities and Probabilities considering
                    %talents or not:
                    utilities = utilities.*FeasibleSet;
                    utilities(utilities==0) = -Inf;
                    probabilities = bsxfun(@rdivide, exp(utilities - max(utilities)), sum(exp(utilities),2));
                    
                    if isempty(fakedata) || fakedata == false
                        idx = sub2ind(size(probabilities), 1:nObs, ChoiceIndex');
                        probabilityChoice = probabilities(idx)';
                        
                    elseif fakedata == true  %Generate Fake Data:
                        uni = rand(nObs,nOptions);
                        E   = gevinv(uni);
                        utilities = (utilities + E);
                        probabilities = bsxfun(@rdivide, exp(utilities), sum(exp(utilities),2));
                        da = false;
                        if da == true
                            oPref = randn(nObs,nOptions);
                            slots = [120;35;45;55;20;70;30;40;120;1000];
                            choiceIndex = DA(probabilities, oPref, slots);
                            Data.ChoiceIndex = choiceIndex;
                            Data.FeasibleSet = exPostFeasible(oPref,choiceIndex);
                            probabilityChoice = NaN;
                        else
                            [~,choiceIndex] = max(probabilities,[],2);
                            Data.ChoiceIndex=choiceIndex;
                            probabilityChoice = NaN;
                        end
                    end
               
                    
                    if gradient == true
                        dL = NaN(nObs,size(parameters,2));
                        dLi = ((x1 - sum(x1.*probabilities,2)));
                        dL(:,1) = dLi(idx);
                        dLi = ((x2 - sum(x2.*probabilities,2)));
                        dL(:,2) = dLi(idx);
                        dLi = ((x3 - sum(x3.*probabilities,2)));
                        dL(:,3) = dLi(idx);
                        dLi = ((x4 - sum(x4.*probabilities,2)));
                        dL(:,4) = dLi(idx);                    
                    else
                        dL = nan;
                    end
                    
                %Model 2:
                %--------
                case "Model 2"
                    
                    utilities =  x1 * parameters(1) + ...
                        x2 * parameters(2)  + ...
                        x3 * parameters(3) + ...
                        x4 * parameters(4) + ...
                        x1.*x2 * parameters(5) + ...
                        x1.*x3 * parameters(6) + ...
                        x1.*x4 * parameters(7) + ...
                        x2.*x3 * parameters(8) + ...
                        x2.*x4 * parameters(9) + ...
                        x3.*x4 * parameters(10) ;
                    
                    %Estimate Utilities and Probabilities considering
                    %talents or not:

                    utilities = utilities.*FeasibleSet;
                    utilities(utilities==0) = -Inf;
                    probabilities = bsxfun(@rdivide, exp(utilities), sum(exp(utilities),2));
                    
                    if isempty(fakedata) || fakedata == false
                        idx = sub2ind(size(probabilities), 1:nObs, ChoiceIndex');
                        probabilityChoice = probabilities(idx)';                        
                    elseif fakedata == true
                        uni = rand(nObs,nOptions);
                        E   = gevinv(uni);
                        utilities = (utilities + E);
                        probabilities = bsxfun(@rdivide, exp(utilities), sum(exp(utilities),2));

                        [~,choiceIndex] = max(probabilities,[],2);
                        Data.ChoiceIndex=choiceIndex;
                        probabilityChoice = NaN;
                    end
                    
                    
                    if gradient == true
                        dL = NaN(nObs,size(parameters,2));
                        dLi = ((x1 - sum(Data.x1.*probabilities,2)));
                        dL(:,1) = dLi(idx);
                        dLi = ((x2 - sum(Data.x2.*probabilities,2)));
                        dL(:,2) = dLi(idx);
                        dLi = ((x3 - sum(Data.x3.*probabilities,2)));
                        dL(:,3) = dLi(idx);
                        dLi = ((x4 - sum(Data.x4.*probabilities,2)));
                        dL(:,4) = dLi(idx);
                        
                        %Interactions
                        dLi = ((x1.*x2 - sum(x1.*x2.*probabilities,2)));
                        dL(:,5) = dLi(idx);
                        dLi = ((x1.*x3 - sum(x1.*x3.*probabilities,2)));
                        dL(:,6) = dLi(idx);
                        dLi = ((x1.*x4 - sum(x1.*x4.*probabilities,2)));
                        dL(:,7) = dLi(idx);
                        dLi = ((x2.*x3 - sum(x2.*x3.*probabilities,2)));
                        dL(:,8) = dLi(idx);
                        dLi = ((x2.*x4 - sum(x2.*x4.*probabilities,2)));
                        dL(:,9) = dLi(idx);
                        dLi = ((x3.*x4 - sum(x3.*x4.*probabilities,2)));
                        dL(:,10) = dLi(idx);
                        
                    else
                        dL = nan;
                    end
                    
                    case  "Model 3"
                    
                    utilities =  x1.*x2 * parameters(1) + ...
                        x1.*x3 * parameters(2) + ...
                        x1.*x4 * parameters(3) + ...
                        x2.*x3 * parameters(4) + ...
                        x2.*x4 * parameters(5) + ...
                        x3.*x4 * parameters(6)+ ... 
                        x1.*(Type == 0) * parameters(7) + ...
                        x2.*(Type == 0) * parameters(8) + ...
                        x3.*(Type == 0) * parameters(9) + ...
                        x4.*(Type == 0) * parameters(10) + ...
                        x1.*(Type == 1) * parameters(11) + ...
                        x2.*(Type == 1) * parameters(12) + ...
                        x3.*(Type == 1) * parameters(13) + ...
                        x4.*(Type == 1) * parameters(14) ;
                    
                    %Estimate Utilities and Probabilities considering
                    %talents or not:
                    utilities = utilities.*FeasibleSet;
                    utilities(utilities==0) = -Inf;
                    probabilities = bsxfun(@rdivide, exp(utilities), sum(exp(utilities),2));
                    
                    if isempty(fakedata) || fakedata == false
                        idx = sub2ind(size(probabilities), 1:nObs, ChoiceIndex');
                        probabilityChoice = probabilities(idx)';                        
                    elseif fakedata == true
                        uni = rand(nObs,nOptions);
                        E   = gevinv(uni);
                        utilities = (utilities + E);
                        probabilities = bsxfun(@rdivide, exp(utilities), sum(exp(utilities),2));

                        [~,choiceIndex] = max(probabilities,[],2);
                        Data.ChoiceIndex=choiceIndex;
                        probabilityChoice = NaN;
                    end
                    
                    
                    if gradient == true
                        dL = NaN(nObs,size(parameters,2));
                        %Interactions
                        dLi = ((x1.*x2 - sum(x1.*x2.*probabilities,2)));
                        dL(:,1) = dLi(idx);
                        dLi = ((x1.*x3 - sum(x1.*x3.*probabilities,2)));
                        dL(:,2) = dLi(idx);
                        dLi = ((x1.*x4 - sum(x1.*x4.*probabilities,2)));
                        dL(:,3) = dLi(idx);
                        dLi = ((x2.*x3 - sum(x2.*x3.*probabilities,2)));
                        dL(:,4) = dLi(idx);
                        dLi = ((x2.*x4 - sum(x2.*x4.*probabilities,2)));
                        dL(:,5) = dLi(idx);
                        dLi = ((x3.*x4 - sum(x3.*x4.*probabilities,2)));
                        dL(:,6) = dLi(idx);
                        
                        % Heterogeneous effects by type:
                        dLi = ((x1.*(Type==0) - sum(x1.*(Type==0).*probabilities,2)));
                        dL(:,7) = dLi(idx);
                        dLi = ((x2.*(Type==0) - sum(x2.*(Type==0).*probabilities,2)));
                        dL(:,8) = dLi(idx);
                        dLi = ((x3.*(Type==0) - sum(x3.*(Type==0).*probabilities,2)));
                        dL(:,9) = dLi(idx);
                        dLi = ((x4.*(Type==0) - sum(x4.*(Type==0).*probabilities,2)));
                        dL(:,10) = dLi(idx);

                        dLi = ((x1.*(Type==1) - sum(x1.*(Type==1).*probabilities,2)));
                        dL(:,11) = dLi(idx);
                        dLi = ((x2.*(Type==1) - sum(x2.*(Type==1).*probabilities,2)));
                        dL(:,12) = dLi(idx);
                        dLi = ((x3.*(Type==1) - sum(x3.*(Type==1).*probabilities,2)));
                        dL(:,13) = dLi(idx);
                        dLi = ((x4.*(Type==1) - sum(x4.*(Type==1).*probabilities,2)));
                        dL(:,14) = dLi(idx);                        
                        
                    else
                        dL = nan;
                    end
                    
                otherwise
                    warning('Model was not specified or does not exist.')
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

            RC1 = parameters(end-2);
            RC2 = parameters(end-1);
            rhoRC = parameters(end);

            P=[RC1 rhoRC;
                0 RC2];

            % Gradients for Cholesky Decomposition
            dP = {};
            dP{1}=[1 0;
                0 0];
            dP{2}=[0 0
                0 1];
            dP{3} =[0 1;
                0 0];                
            
            dtemp_rc = {};

            ViWeights=q_Weights;

            ChoiceIndex = Data.ChoiceIndex(sample);
            NextChoiceIndex = Data.NextChoiceIndex(sample);
            
            switch Pack.Model

                %Model 1:
                %--------                
                case "Model 1"
                    
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
                case "Model 1 Exploded Logit"
                    
                    dL_temp = NaN(nObs,size(parameters,2));
                    dL_Joint_temp = NaN(nObs,size(parameters,2));
                    probabilityChoice = NaN(nObs, 1);
                    probabilityJoint = NaN(nObs, 1);
                    
                    for s = 1:nObs
                    
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
                        for ii = 1:NXX
                            utilities = utilities + XX{ii}(s,:) * parameters(ii);
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
                                jj = nx;
                                dLi(:,jj) = (XX{nx}(s,pickIndex).*prChoiceVi - sum(XX{nx}(s,:).*probabilitiesViAll,2).*prChoiceVi);
                                dLi_temp(jj) = dLi(:,jj)'*ViWeights; % Parameters Beta
                                dLiNext(:,jj) = (XX{nx}(s,pickIndexNext).*prChoiceViNext - sum(XX{nx}(s,:).*probabilitiesViAllNext,2).*prChoiceViNext);
                                dLiJoint(:,jj) = (dLi(:,jj).*prChoiceViNext + dLiNext(:,jj).*prChoiceVi); % Beta parameters
                                dLiJoint_temp(jj) = dLiJoint(:,jj)'*ViWeights; % Average joint probability:
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
                    
                    dL_temp = dL_temp./probabilityChoice;
                    dL=dL_temp;

                    dL_Joint_temp = dL_Joint_temp./probabilityJoint;
                    dL_Joint = dL_Joint_temp;
                    
                otherwise
                    warning('Model was not specified or does not exist.')
            end
            
        end


        % 4. Defining Objective Function
        function [Q_total, Gradient, Q_Choice, GradientChoice, Q_Joint, GradientJoint]=objectiveFunction(parameters, passedFunction)

            global Data Pack;
            
            switch Pack.Model
                
                case 'Model 1'

                    sample = Data.sample;
                    nObs = size(sample,1);

                    [probabilityChoice, dL]=passedFunction(parameters);

                    Q_Choice= -sum(log(probabilityChoice))/nObs;

                    if nargout > 1 
                        GradientChoice= -sum(dL)/nObs;
                    end
                    
                    Q_total = Q_Choice;
                    Gradient = GradientChoice;
                    
                case 'Model 1 Exploded Logit'

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
                    
                otherwise
                    warning('Model was not specified or does not exist.')
            end
                    
        end
        
        % 5. Check gradients of utility function:
        function checkGradient(parameters, passedUtilityFunction)
            
            global Data;
            
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
            [~, ~, ~, dPrChoice, ~, ~]=dcmLab.objectiveFunction(theta, passedUtilityFunction); 
            dif=abs((NUM_dPrChoice(:,1:i)-dPrChoice(:,1:i)));
            difp=abs((NUM_dPrChoice(:,1:i)-dPrChoice(:,1:i)))./dPrChoice(:,1:i);
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
            
            theta=parameters;step=10^-7;
            NUM_Grad = NaN(size(theta,2),1);
            
            for i=1:length(theta) %80:90 %
                theta0=theta;
                theta1=theta;

                theta0(i)=theta(i)-step/2;
                theta1(i)=theta(i)+step/2;
                
                [~, ~, ~, ~, Q0, ~]=dcmLab.objectiveFunction(theta0,passedFunction);
                [~, ~, ~, ~, Q1, ~]=dcmLab.objectiveFunction(theta1,passedFunction);

                NUM_Grad(i,1)=(Q1-Q0)/step;

                fprintf('Overall change in objective function: Parameter %3.0f \n',i )
            end

            [~, ~, ~, ~, ~, Gradient]=dcmLab.objectiveFunction(theta,passedFunction);

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
            
            global Data;
            
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

            q_NodeList = Data.q_Nodes;
            q_Weights = Data.q_Weights;

            RC1 = parameters(end-2);
            RC2 = parameters(end-1);
            rhoRC = parameters(end);

            P=[RC1 0;
                rhoRC RC2]';

            ViWeights=q_Weights;

            %Model 1:
            %--------                
            probabilities = NaN(nObs, nOptions);
            Utilities = NaN(nObs, nOptions);

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
        end
      
        
    end
end

    