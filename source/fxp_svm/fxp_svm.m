classdef fxp_svm
    properties
        C;
        W;
        lr;
        precision;
        word_length;
        frac_length;
    end
    
    methods
        function obj = fxp_svm(CParam, LearningRate, WordLength, FracLength)
           obj.word_length = WordLength;
           obj.frac_length = FracLength;
           obj.precision = fimath('SumMode', 'SpecifyPrecision', 'SumWordLength', WordLength, 'SumFractionLength', FracLength, 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', WordLength, 'ProductFractionLength', FracLength);
           obj.lr = fi(LearningRate, 1, obj.word_length, obj.frac_length, obj.precision);
           obj.C = fi(CParam, 1, obj.word_length, obj.frac_length, obj.precision);
        end
        
        function obj = fit(obj, X, y, epochs)
            % Number of features
            m = size(X, 2);
            
            % Append column of ones at the end of X
            ones_vec = ones(size(X, 1), 1);
            data = fi([X, ones_vec], 1, obj.word_length, obj.frac_length, obj.precision);
            y_data = fi(y(:, :), 1, obj.word_length, obj.frac_length, obj.precision);
            obj.W = fi(rand(m + 1, 1), 1, obj.word_length, obj.frac_length, obj.precision);
            
            for epoch = 1:epochs
                ix_shuffle = randperm(size(X, 1));
                data = data(ix_shuffle, :);
                y_data = y_data(ix_shuffle, :);
                
                grad = obj.compute_grads(data, y_data);
                obj.W = obj.W - obj.lr .* grad;
                disp("Cost");
                disp(obj.compute_loss(data, y_data));
            end                     
        end
        
        function prediction = predict(obj, X)
           ones_vec = fi(ones(size(X, 1), 1), 1, obj.word_length, obj.frac_length, obj.precision);
           data = fi([X, ones_vec], 1, obj.word_length, obj.frac_length, obj.precision);
           prediction = sign(data * obj.W);
        end
        
        function loss = compute_loss(obj, X, y)
            N = size(X, 1);
            distances = fi(ones(length(y), 1), 1, obj.word_length, obj.frac_length, obj.precision) - y .* (X * obj.W);
            distances(distances < 0) = 0;
            
            % Compute Hinge loss
            hinge_loss = obj.C .* fi(sum(distances), 1, obj.word_length, obj.frac_length, obj.precision) ./ fi(N, 1, obj.word_length, obj.frac_length, obj.precision);
            
            % Calculate cost
            loss = fi((1 / 2), 1, obj.word_length, obj.frac_length, obj.precision) .* (obj.W' * obj.W) + hinge_loss;
        end
        
        function grads = compute_grads(obj, X, y)
           distance = fi(ones(length(y), 1), 1, obj.word_length, obj.frac_length, obj.precision) - y .* (X * obj.W);
           grads = fi(zeros(size(obj.W)), 1, obj.word_length, obj.frac_length, obj.precision);
           
           for index = 1:length(distance)
               if max(0, distance(index)) == 0
                   dist_i = obj.W;
               else
                   dist_i = obj.W - obj.C .* y(index) .* X(index,:)';
               end
               grads = grads + dist_i;
           end
           
           grads = grads ./ fi(size(X, 1), 1, obj.word_length, obj.frac_length, obj.precision);
        end
        
        function sc = score(obj, X, y_true)
           prediction = obj.predict(X);
           sc = sum(y_true == prediction) / size(X, 1);
        end
    end
end
