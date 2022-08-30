svm = fxp_svm(100, 0.01, 32, 15);

data = csvread("../Datasets/toy_dataset.csv", 1, 0);
X = data(:, [1, 2]);
y = data(:, 3);

disp(size(X))
disp(size(y))

svm = svm.fit(X, y, 10);
svm.score(X, y)