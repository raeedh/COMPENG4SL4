x = [0:1/9:1];
t = transpose([-0.11234088  1.33386848  0.69332864 -0.63718403 -1.0625367 0.40120175 0.69581025 -0.34107385 -0.70460877 -0.36980081]);

% plot(x,t)

X = repmat(transpose(x),1,10);
a = repmat(0:9,10,1);

X = X.^a;

lambda = exp(-24);
B = diag(lambda * ones(1,10));

close all

% for D = 10
%     X_calc = X(:,1:D);
% %     size(transpose(X_calc))
% %     size(t)
%     size(X_calc)
%     w = inv(transpose(X_calc)*X_calc + B)*(transpose(X_calc)*t);
%     w = flip(w)
%     
%     if D > 9
%         w_poly = polyfit(x,t,D-1)
% %         w_poly(10)
%         figure;
%         x_plot = linspace(0,1,100);
%         plot(x_plot,polyval(w,x_plot)); title("M =", D-1);
%         hold on;
%         scatter(x, t);
%         plot(x_plot,sin(4*pi*x_plot));
%         
%         figure; 
%         plot(x_plot,polyval(w_poly,x_plot)); title("polyfit, M =", D-1);
%         hold on;
%         scatter(x, t);
%         plot(x_plot,sin(4*pi*x_plot));
%     end
% end

XX_train = zeros(10,9);
for i = 1:9
    XX_train(:,i) = x;
    XX_train(:,i) = XX_train(:,i).^i;
end
XX_train

