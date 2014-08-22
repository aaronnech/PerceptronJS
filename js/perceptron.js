/**
 * A Perceptron is a single artificial neuron-like structure that performs
 * classification for linearly seperable data in N dimensions.
 * @param {int} inputDimension The space of input that can be fed.
 * @param {int} learningRate The rate at which this perceptron learns
 * @param {float} threshold A float between 0 and 1 that when the output of
 *                          the perceptron is greater than, it will classify
 *                          as true.
 */
function Perceptron(inputDimension, learningRate, threshold) {
	this.inputs_ = [];
	this.inputWeights_ = [];
	this.learningRate_ = learningRate;
	this.threshold_ = threshold;

	for (var i = 0; i < inputDimension + 1; i++) {
		this.inputWeights_.push(0.0);
		this.inputs_.push(0.0);
	}
	// Set the bias
	this.inputs_[0] = 1.0;
}


/**
 * Gets the score of a particular class index in the current
 * input state.
 * @return {int} The score for the current input
 * @private
 */
Perceptron.prototype.getScore_ = function() {
	var score = 0;
	for (var i = 0; i < this.inputWeights_.length; i++) {
		score += this.inputWeights_[i] * this.inputs_[i];
	}
	return score;
};


/**
 * Sets the input for this perceptron
 * @param {Array.<float>} input The input to set
 * @private
 */
Perceptron.prototype.setInput_ = function(input) {
	// Offset by one to keep bias constant.
	for (var i = 1; i < this.inputs_.length; i++) {
		this.inputs_[i] = input[i - 1];
	}
};


/**
 * Updates the weights of the perceptron (called after failed training)
 * @param {float} error The training error to correct
 * @private
 */
Perceptron.prototype.updateWeights_ = function(error) {
	for (var i = 0; i < this.inputWeights_.length; i++) {
		var delta = this.learningRate_ * error * this.inputs_[i];
		this.inputWeights_[i] += delta;
	}
};


/**
 * Trains the perceptron on one example. (Executes one
 * training iteration).
 * @param {Array.<float>} input The input to train on
 * @param {boolean} expected The classification expected.
 * @return {boolean} True if this training example passed, false otherwise
 */
Perceptron.prototype.train = function(input, expected) {
	if (input.length != this.inputs_.length - 1) {
		throw 'Illegal input dimension';
	}

	this.setInput_(input);
	var score = this.getScore_();

	var expectedValue = 0;
	if (expected) {
		expectedValue = 1;
	}

	var result = 0;
	if (score > this.threshold_) {
		result = 1;
	}

	if (expectedValue == result) {
		return true;
	} else {
		this.updateWeights_(expectedValue - result);
		return false;
	}
};


/**
 * Predicts a classification for the given input
 * @param  {Array.<float>} input The input
 * @return {boolean}       The classification
 */
Perceptron.prototype.classify = function(input) {
	if (input.length != this.inputs_.length - 1) {
		throw 'Illegal input dimension';
	}

	this.setInput_(input);
	var score = this.getScore_();
	return score > this.threshold_;
};


/**
 * Trains the perceptron on an input set until either
 * the given maximum iterators have been met, or the perceptron
 * correctly classifies all inputs.
 * @param  {Array.<Array.<float>>} inputSet The input
 * @param  {Array.<boolean>} expectedSet The expected classifications
 * @param  {int} maxIterations The maximum iterations allowed for this session.
 * @return {boolean} False if max iterations was reached, true otherwise.
 */
Perceptron.prototype.trainSet = function(inputSet, expectedSet, maxIterations) {
	if (inputSet.length != expectedSet.length) {
		throw 'Input and expectation mismatch';
	}
	var allPassed = false;
	var iteration = 0;
	while (!allPassed && iteration < maxIterations) {
		allPassed = true;
		for (var i = 0; i < inputSet.length; i++) {
			allPassed = allPassed && this.train(inputSet[i], expectedSet[i]);
		}
		iteration++;
	}
	return iteration != maxIterations;
};