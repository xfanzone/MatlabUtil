function selected_idx = adaselect(features, labels, num_select)
% features: m by n features matrix. each row is an instance
% labels: m by 1 label vector, each row is an instance
% num_select: scalar, number of features to selected_idx

assert(size(features, 2) >= num_select); % select a subset of features, for sure
assert(size(features, 1) == size(labels, 1)); % labels and features should match
assert(any(labels == 0) & any(labels == 1)); % there must be both positive samples and negative samples

num_instance = size(features, 1); % number of instance 
num_feature = size(features, 2); % number of feature dimension
num_pos = sum(labels == 1); %number of positive samples
num_neg = sum(labels == 0); %number of negative samples

T = num_select; % T: number of rounds 
w = zeros(T + 1, num_instance); % actually only T rows are needed. make it T + 1 for convenient;
w(1, find(labels == 1)) = 1 / (2 * num_pos); %initial weights for positive samples
w(1, find(labels == 0)) = 1 / (2 * num_neg); %initial weights for negative samples

candidate_idx = 1:num_feature; %candidate index. making an entry zero means it has been moved to selected idx 
selected_idx = zeros(1, num_select); % selected index

for t = 1:T %T rounds for T selected features index
    t
    tic
	w(t, :) = w(t, :)  / sum(w(t,:)); % normalization

	best_err = Inf; 
	best_predicted = 0; %just declare 
	best_idx = 0; %just declare

	for j = 1:num_feature %iterate all features
		if candidate_idx(j) == -1 
			continue; %ignore features that have already been selected
		end
		predicted = weak_classifier(features(:, j), labels); %get weak classifier trained on the j-th feature dimension
		err = w(t, :)*abs(predicted - labels); % calculate the weighted error
		if err < best_err %update best error rate
			best_err = err; %lowest error
			best_predicted = predicted; % best predicted output label
			best_idx = j; % best predicted features dimension index
		end
	end
	w(t + 1, :) = w(t, :) * bsxfun(@power, best_err / (1 - best_err), (best_predicted == labels)); % update the weights. beta = best_err / ( 1 - best_err)
	selected_idx(t) = best_idx;
	candidate_idx(t) = -1;
    toc;
end %end of for
end %function end

function out_label = weak_classifier(feature, labels)
	assert(all(size(feature) == size(labels)));
	assert(size(feature, 2) == 1);
	pos_mean = mean(feature(labels == 1));
	neg_mean = mean(feature(labels == 0));
	th = (pos_mean + neg_mean) / 2;
	out_label = zeros(size(labels));
	if pos_mean >= neg_mean
		out_label(feature > th) = 1;
	else
		out_label(feature < th) = 1;
    end
end