import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        weights = self.get_weights()
        return nn.DotProduct(weights, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(nn.DotProduct(self.get_weights(), x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        ans = 0
        while ans < 1:
            i = 0
            right = 0
            for x, y in dataset.iterate_once(1):
                multiplier = nn.as_scalar(y)

                if multiplier != self.get_prediction(x):
                    nn.Parameter.update(self.get_weights(), x, multiplier)
                else:
                    right += 1
                
                i += 1
            ans = right/ i
        return ans

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.x1 = nn.Parameter(1, 200)
        self.y1 = nn.Parameter(200, 1)
        self.x2 = nn.Parameter(1, 200)
        self.y2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        linear = nn.Linear(x, self.x1)
        bias = nn.AddBias(linear, self.x2)
        relu = nn.ReLU(bias)
        relu_lin = nn.Linear(relu, self.y1)
        predicted_y = nn.AddBias(relu_lin, self.y2)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        ran = self.run(x)
        return nn.SquareLoss(ran, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.075
        loss = 10000000
        batch_size = 50

        while loss > 0.02:
            for x, y in dataset.iterate_once(batch_size):
                current_loss = self.get_loss(x, y)
                loss = nn.as_scalar(current_loss)
                grad_x1, grad_y1, grad_x2, grad_y2 = nn.gradients(current_loss, [self.x1, self.y1, self.x2, self.y2])
                self.x1.update(grad_x1, -1 * learning_rate)
                self.y1.update(grad_y1, -1 * learning_rate)
                self.x2.update(grad_x2, -1 * learning_rate)
                self.y2.update(grad_y2, -1 * learning_rate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.x1 = nn.Parameter(784, 250)
        self.x2 = nn.Parameter(250, 10)
        self.y1 = nn.Parameter(1, 250)
        self.y2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        linear1 = nn.Linear(x, self.x1)
        bias1 = nn.AddBias(linear1, self.y1)
        relu1 = nn.ReLU(bias1)

        linear3 = nn.Linear(relu1, self.x2)
        predicted = nn.AddBias(linear3, self.y2)
        return predicted


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        softy = nn.SoftmaxLoss(self.run(x), y)
        return softy

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.5
        accuracy = 0
        batch_size = 100

        while accuracy < 0.975:
            for x, y in dataset.iterate_once(batch_size):
                current_loss = self.get_loss(x, y)

                grad_x1, grad_y1, grad_x2, grad_y2 = nn.gradients(current_loss, [self.x1, self.y1, self.x2, self.y2])
                self.x1.update(grad_x1, -1 * learning_rate)
                self.y1.update(grad_y1, -1 * learning_rate)
                self.x2.update(grad_x2, -1 * learning_rate)
                self.y2.update(grad_y2, -1 * learning_rate)

                accuracy = dataset.get_validation_accuracy()
                print(accuracy)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.x1 = nn.Parameter(47, 250)
        self.x1_hid = nn.Parameter(47, 250)
        self.x2 = nn.Parameter(250, 250)
        self.x_res = nn.Parameter(250, 5)
        
        self.y1 = nn.Parameter(1, 250)
        self.y2 = nn.Parameter(1, 250)
        self.y_res = nn.Parameter(1, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        lin = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.x1), self.y1))
        for x in xs[1:]:
            
            linear = nn.Linear(x, self.x1_hid)
            lin_after = nn.Linear(lin, self.x2)
            add = nn.Add(linear, lin_after)
            bias = nn.AddBias(add, self.y2)
            relu = nn.ReLU(bias)
            lin = relu

        predicted = nn.AddBias(nn.Linear(lin, self.x_res), self.y_res)
        return predicted



    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        softy = nn.SoftmaxLoss(self.run(xs), y)
        return softy

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.5
        accuracy = 0
        batch_size = 100

        while accuracy < 0.83:
            for x, y in dataset.iterate_once(batch_size):
                current_loss = self.get_loss(x, y)

                grad_x1, grad_x1_hid, grad_x2, grad_x_res, grad_y1, grad_y2, grad_y_res = nn.gradients(current_loss, [self.x1, self.x1_hid, self.x2, self.x_res, self.y1, self.y2, self.y_res])
                self.x1.update(grad_x1, -1 * learning_rate)
                self.x1_hid.update(grad_x1_hid, -1 * learning_rate)
                self.x2.update(grad_x2, -1 * learning_rate)
                self.x_res.update(grad_x_res, -1 * learning_rate)
                self.y1.update(grad_y1, -1 * learning_rate)
                self.y2.update(grad_y2, -1 * learning_rate)
                self.y_res.update(grad_y_res, -1 * learning_rate)

                accuracy = dataset.get_validation_accuracy()
                # print(accuracy)

