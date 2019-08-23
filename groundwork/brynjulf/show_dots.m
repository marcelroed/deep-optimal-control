function show_dots(x, y, is_test)
    
if nargin < 3
    is_test = false;
end

if is_test
%     scatter(x(1,:), x(2,:), 10, y, 'filled', 'MarkerEdgeColor', 'black')
    scatter(x(1,:), x(2,:), 10, y, 'filled')
else
%     scatter(x(1,:), x(2,:), 10, y, 'filled', 'MarkerEdgeColor', 'black')
    scatter(x(1,:), x(2,:), 10, y, 'filled')
end
