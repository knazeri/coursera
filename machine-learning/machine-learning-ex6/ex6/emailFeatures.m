function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;

vocab_indices = (1:n)';

% ismember: returns a vector containing 1 (true) 
% where the data in vocab_indices is found in word_indices
x = ismember(vocab_indices, word_indices);

end