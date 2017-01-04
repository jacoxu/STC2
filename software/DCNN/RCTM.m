classdef RCTM < handle

    properties
        E;
        one; 
        one_b;
        two;
        two_b;
        three;
        three_b;
        Z;       
    end
    
    methods
        function CR = RCTM(p)
            factor_b = p(65);
            %In case left initialized
            CR.one = 0;
            CR.one_b = 0;
            CR.two = 0;
            CR.two_b = 0;
            CR.three = 0;
            CR.three_b = 0;            
            % R = NORMRND(MU,SIGMA,[M,N,...])£¬returns an M-by-N-by-... array.
            %CR.E = normrnd(0,0.05,[p(1),p(30)]); 
            CR.E = randi([-25,25],p(1),p(30))/100; 
            disp(strcat('Number of weights E:',num2str(size(CR.E))));
            disp(strcat('Size (max) sentence matrix: ', num2str(p(2)*p(1)))); 

            if(p(68))
                p(1)=p(1)+p(69);
            end
 
            CR.one = normrnd(0,0.05,[p(1)*p(3),p(4)]);
            disp(strcat('Number of weights 1:',num2str(size(CR.one))));
            
            if p(12) 
              p(1) = p(1)/2; 
            end
            CR.one_b = factor_b.*rand(p(1)*p(3),1);
            disp(strcat('Number of weights 1 bias:',num2str(size(CR.one_b))));
            %disp(strcat('Size representation 1: ', num2str([p(3)*p(1),p(7)])));
            
            if p(10) >= 2 
                CR.two = normrnd(0,0.05,[p(1)*p(3)*p(5),p(6)]);
                disp(strcat('Number of weights 2:',num2str(size(CR.two))));

                if p(13) 
                   p(1) = p(1)/2; 
                end
                
                CR.two_b = factor_b.*rand(p(1)*p(5),1);
                disp(strcat('Number of weights 2 bias:',num2str(size(CR.two_b))));
                %disp(strcat('Size representation 2: ', num2str([p(5)*p(1),p(11)])));
            end
            
            if p(10) == 3 
                CR.three = normrnd(0,0.05,[p(1)*p(5)*p(37),p(36)]);
                disp(strcat('Number of weights 3:',num2str(size(CR.three))));
                
                if p(35)
                    p(1) = p(1)/2;
                end

                CR.three_b = factor_b.*rand(p(1)*p(37),1);
                disp(strcat('Number of weights 3 bias:',num2str(size(CR.three_b))));
                %disp(strcat('Size representation 3: ', num2str([p(37)*p(1),p(11)])));
                
            end

            if p(10) == 1
                 CR.Z = normrnd(0,0.05,[p(9),p(1)*p(7)*p(3)+1]);
            elseif p(10) == 2
                 CR.Z = normrnd(0,0.05,[p(9),p(1)*p(7)*p(5)+1]);             
            elseif p(10) == 3
                 CR.Z = normrnd(0,0.05,[p(9),p(1)*p(7)*p(37)+1]);            
            end
            disp(strcat('Number of weights in classification:',num2str(size(CR.Z))));
            
            if p(31)
               CR.E = single(CR.E);
               CR.one = single(CR.one);
               CR.one_b = single(CR.one_b);
               CR.two = single(CR.two);
               CR.two_b = single(CR.two_b);
               CR.three = single(CR.three);
               CR.three_b = single(CR.three_b);
               CR.Z = single(CR.Z);
            end
        end
    end  
end