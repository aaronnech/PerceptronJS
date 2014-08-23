/**
 * A MultiPerceptron is a single artificial neuron-like structure that performs
 * multiple classification for linearly seperable data in N dimensions.
 * @param {int} inputDimension The space of input that can be fed.
 * @param {int} classCount The number of classifications this perceptron can make
 * @param {int} learningRate The rate at which this perceptron learns
 */
function MultiPerceptron(inputDimension, classCount, learningRate) {
	this.inputs_ = [];
	this.inputWeights_ = [];
	this.learningRate_ = learningRate;
	for (var classification = 0; classification < classCount; classification++) {
		this.inputWeights_.push([]);
		for (var i = 0; i < inputDimension + 1; i++) {
			this.inputWeights_[classification].push(0.0);
		}
	}
	for (var i = 0; i < inputDimension + 1; i++) {
		this.inputs_.push(0.0);
	}
	// Set the bias
	this.inputs_[0] = 1.0;
}


/**
 * Gets the score of a particular class index in the current
 * input state.
 * @param {int} classification The classification index to score.
 * @return {int} The score for the current input
 * @private
 */
MultiPerceptron.prototype.getScore_ = function(classification) {
	var score = 0;
	for (var i = 0; i < this.inputWeights_[classification].length; i++) {
		score += this.inputWeights_[classification][i] * this.inputs_[i];
	}
	return score;
};


/**
 * Sets the input for this perceptron
 * @param {Array.<float>} input The input to set
 * @private
 */
MultiPerceptron.prototype.setInput_ = function(input) {
	// Offset by one to keep bias constant.
	for (var i = 1; i < this.inputs_.length; i++) {
		this.inputs_[i] = input[i - 1];
	}
};


/**
 * Gets the current weights of the perceptron.
 */
MultiPerceptron.prototype.getWeights = function() {
	return this.inputWeights_;
};


/**
 * Sets the current weights of the perceptron.
 * @return {Array.<Array.<float>>} The perceptron weights
 */
MultiPerceptron.prototype.setWeights = function(weights) {
	if (weights.length != this.inputWeights_.length ||
		weights[0].length != this.inputWeights_[0].length) {
		throw 'Cannot change weight dimension';
	}
	return this.inputWeights_ = weights;
};


/**
 * Punishes the given classifcation by the input
 * @param {int} classification The classification index to score.
 * @private
 */
MultiPerceptron.prototype.punishWeights_ = function(classification) {
	for (var i = 0; i < this.inputWeights_[classification].length; i++) {
		var delta = this.learningRate_ * this.inputs_[i];
		this.inputWeights_[classification][i] -= delta;
	}
};


/**
 * Encourages the given classification by the input
 * @param {int} classification The classification index to score.
 * @private
 */
MultiPerceptron.prototype.encourageWeights_ = function(classification) {
	for (var i = 0; i < this.inputWeights_[classification].length; i++) {
		var delta = this.learningRate_ * this.inputs_[i];
		this.inputWeights_[classification][i] += delta;
	}
};


/**
 * Trains the perceptron on one example. (Executes one
 * training iteration).
 * @param {Array.<float>} input The input to train on
 * @param {int} expected The classification expected.
 * @return {boolean} True if this training example passed, false otherwise
 */
MultiPerceptron.prototype.train = function(input, expected) {
	if (input.length != this.inputs_.length - 1) {
		throw 'Illegal input dimension';
	}

	this.setInput_(input);
	var maxScore = this.getScore_(0);
	var maxScoreClassification = 0;
	for (var i = 1; i < this.inputWeights_.length; i++) {
		var score = this.getScore_(i);
		if (score > maxScore) {
			maxScore = score;
			maxScoreClassification = i;
		}
	}

	if (expected == maxScoreClassification) {
		return true;
	} else {
		this.encourageWeights_(expected);
		this.punishWeights_(maxScoreClassification);
		return false;
	}
};


/**
 * Predicts a classification for the given input
 * @param  {Array.<float>} input The input
 * @return {int}       The classification
 */
MultiPerceptron.prototype.classify = function(input) {
	if (input.length != this.inputs_.length - 1) {
		throw 'Illegal input dimension';
	}

	this.setInput_(input);
	var maxScore = this.getScore_(0);
	var maxScoreClassification = 0;
	for (var i = 1; i < this.inputWeights_.length; i++) {
		var score = this.getScore_(i);
		if (score > maxScore) {
			maxScore = score;
			maxScoreClassification = i;
		}
	}
	return maxScoreClassification;
};


/**
 * Trains the perceptron on an input set until either
 * the given maximum iterators have been met, or the perceptron
 * correctly classifies all inputs.
 * @param  {Array.<Array.<float>>} inputSet The input
 * @param  {Array.<int>} expectedSet The expected classifications
 * @param  {int} maxIterations The maximum iterations allowed for this session.
 * @return {boolean} False if max iterations was reached, true otherwise.
 */
MultiPerceptron.prototype.trainSet = function(inputSet, expectedSet, maxIterations, callback) {
	if (inputSet.length != expectedSet.length) {
		throw 'Input and expectation mismatch';
	}
	var numberFailed = -1;
	var iteration = 0;
	while (numberFailed != 0 && iteration < maxIterations) {
		numberFailed = 0;
		for (var i = 0; i < inputSet.length; i++) {
			if(!this.train(inputSet[i], expectedSet[i])) {
				numberFailed++;
			}
		}
		callback(numberFailed, inputSet.length, iteration);
		iteration++;
	}
	return iteration != maxIterations;
};