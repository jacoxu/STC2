function [prec, recall, acc, f1] = getAccuracy(predicted, gold)

cm = confusionmat(gold,predicted);

prec = mean(diag(cm)'./max(sum(cm),1e-5));
recall = mean(diag(cm)'./max(sum(cm'),1e-5));
acc = sum(diag(cm))/sum(sum(cm));
f1 = 2*prec*recall/(prec+recall);

end