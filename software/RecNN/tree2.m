classdef tree2
    
    properties
        % parent pointers
        pp = [];
        nodeNames;
        nodeFeatures_unnormalized =[];
        nodeFeatures = [];
        leafFeatures=[];
        % the parent pointers do not save which is the left and right child of each node, hence:
        % numNodes x 2 matrix of kids, [0 0] for leaf nodes
        kids = [];
        numkids = [];
        freq = [];
        nodeLabels=[];
        score=0;
        nodeScores=[];
        nodeZ = [];
        nodeDelta_out1 = [];
        nodeDelta_out2 = [];
        parentDelta = [];
        catDelta = [];
        catDelta_out = [];
        node_dW1 = [];
        node_dW2 = [];
        node_dW3 = [];
        node_dW4 = [];
        node_dL = [];
        node_y1c1 = [];
        node_y2c2 = [];
        
    end
    
    
    methods
        function id = getTopNode(obj)
            id = find(obj.pp==0);
        end
        
        function kids = getKids(obj,node)
            %kids = find(obj.pp==node);
            kids = obj.kids(node,:);
        end

        %TODO: maybe compute leaf-node-ness once and then just check for it
        function l = isLeaf(obj,node)
            l = ~any(obj.pp==node);
        end        
        
        
        
function plotTree(obj)
            %TREEPLOT Plot picture of tree.
            %   TREEPLOT(p) plots a picture of a tree given a row vector of
            %   parent pointers, with p(i) == 0 for a root and labels on each node.
            %
            %   Example:
            %      myTreeplot([2 4 2 0 6 4 6],{'i' 'like' 'labels' 'on' 'pretty' 'trees' '.'})
            %   returns a binary tree with labels.
            %
            %   Copyright 1984-2004 The MathWorks, Inc.
            %   $Revision: 5.12.4.2 $  $Date: 2004/06/25 18:52:28 $
            %   Modified by Richard @ Socher . org to display text labels
            
            p = obj.pp';
            [x,y,h]=treelayout(p);
            f = find(p~=0);
            pp = p(f);
            X = [x(f); x(pp); NaN(size(f))];
            Y = [y(f); y(pp); NaN(size(f))];
            X = X(:);
            Y = Y(:);
            
            n = length(p);
            if n < 500,
                plot (x, y, 'wo', X, Y, 'b-');
            else
                plot (X, Y, 'r-');
            end;
            xlabel(['height = ' int2str(h)]);
            axis([0 1 0 1]);
            
            if ~isempty(obj.nodeNames)
                for l=1:length(obj.nodeNames)
                        if isnumeric(obj.nodeNames(l))
                            text(x(l),y(l),num2str(obj.nodeNames(l)),'Interpreter','none',...
                                'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
                        else
                            text(x(l),y(l),obj.nodeNames{l},'Interpreter','none',...
                                'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
                        end
%                     if ~isempty(obj.nodeLabels)
%                         if iscell(obj.nodeNames)
%                             text(x(l),y(l),[labels{l} '(' obj.nodeLabels{l} ')'],'Interpreter','none',...
%                                 'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
%                         else
%                             % for numbers
%                             if isnumeric(obj.nodeLabels(l))
% %                                 if isinteger(obj.nodeLabels(l))
%                                     allL = obj.nodeLabels(:,l);
%                                     allL = find(allL);
%                                     if isempty(allL)
%                                         text(x(l),y(l),[num2str(obj.nodeNames(l))],'Interpreter','none',...
%                                         'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
%                                     else
%                                         text(x(l),y(l),[num2str(obj.nodeNames(l)) ' (' mat2str(allL) ')'],'Interpreter','none',...
%                                             'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
%                                     end
%                                     
% %                                 else
% %                                     text(x(l),y(l),[obj.nodeLabels(l) ' ' num2str(obj.nodeLabels(l),'%.1f') ],'Interpreter','none',...
% %                                         'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
% %                                 end
%                                 % change to font size 6 for nicer tree prints
%                             else
%                                 text(x(l),y(l),[obj.nodeNames{l}],'Interpreter','none',...
%                                     'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
%                             end
%                         end
%                     end
                end
            end
            
            
        end
        
        
    end
end