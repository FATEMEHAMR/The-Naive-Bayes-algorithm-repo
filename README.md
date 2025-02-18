# The-Naive-Bayes-algorithm


In this project, the goal is to implement a simple learning process using the Naïve Bayes algorithm. As we will see, this algorithm uses the principles of conditional probabilities and Bayes' theorem to classify data.

The learning process mentioned is implemented in two stages. In the first stage, you receive a dataset named X, which is a collection of samples. Each sample is represented by an n-dimensional vector \( \mathbf{x} = (x_1, \ldots, x_n) \in X \), and each component of the vector is called a feature. Therefore, each vector includes n features. Additionally, each \( \mathbf{x} \) belongs to one of m categories \( c_1, c_2, \ldots, c_m \). For example, each member of the dataset dogs.csv in this project represents the features of a dog (e.g., height and width), and the number of categories for dogs is 3. Based on the information available in this stage, i.e., the set of samples and the class of each sample, you can obtain the distribution of features in each category. Then, in the next stage, using these distributions and the Naïve Bayes algorithm, you can classify new data that was not previously in your dataset. Below, we first explain the learning part and then the Naïve Bayes algorithm in detail.

#### Stage One: Training

In this stage, you receive the dataset X, which includes samples \( \mathbf{x} \) and their classifications. Then, you obtain the feature distribution in each category. More precisely, the goal is to obtain the following probability distribution for each feature \( 1 \leq i \leq n \) and each category \( 1 \leq k \leq m \):

\[ P(x_i | c_k) \]

If \( x_i \) is a discrete random variable, you can find the number of samples in category \( c_k \) that have the feature \( x_i = \alpha \), divide it by the total number of samples in that category, and obtain \( P(x_i = \alpha | c_k) \). In this project, the type of distribution is given to you, and you just need to find the distribution parameters. The following distributions are used in this project:
1. Uniform PDF with parameters a and b:
\( f_{X}(x)=\left\{\begin{array}{l}\frac{1}{b-a}, \quad \text { if } x \in[a, b] . \\ 0, \quad \text { otherwise. }\end{array}\right. \)
2. Gaussian PDF with parameters \( \mu \) and \( \sigma \):
\( f(x)=\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2}} \)
3. PMF for a binomial random variable with parameters n and p:
\( p_{X}(x)=\binom{n}{x} p^{x}(1-p)^{n-x}, \quad x=1,2, \ldots, n \)

#### Stage Two: Classification and Naïve Bayes Algorithm

In this section, the Naïve Bayes algorithm is explained to classify new data. Suppose \( \mathbf{x} \) is a vector \( \mathbf{x} = (x_1, x_2, \ldots, x_n) \) where n is the number of features of each sample. Also, as mentioned earlier, assume each \( \mathbf{x} \) belongs to one of the m categories \( c_1, c_2, \ldots, c_m \).

We want to determine to which category a given \( \mathbf{x} \) that is not in our dataset belongs. To do this, we choose the category that maximizes the probability \( P(c_k | \mathbf{x}) \). In fact, we select the category for which the probability of belonging is maximized. Therefore, we solve the following optimization:

\[ \hat{k} = \arg\max_k P(c_k | \mathbf{x}) \]

According to Bayes' theorem, the probability \( P(c_k | \mathbf{x}) \) is calculated as follows:

\[ P(c_k | \mathbf{x}) = \frac{P(\mathbf{x} | c_k) P(c_k)}{P(\mathbf{x})} \]

We know that the denominator is the same for all categories. So, to determine the desired category, we just need to maximize the numerator. It is common to assume that each class has an equal probability \( P(c_1) = P(c_2) = \ldots = P(c_m) \). Hence, we need to maximize \( P(\mathbf{x} | c_k) \). It is also assumed that, given the category, the features are independent (hence the algorithm is called Naïve or simple):

\[ P(\mathbf{x} | c_k) = P(x_1 | c_k) P(x_2 | c_k) \ldots P(x_n | c_k) \]

If \( x_i \) is discrete, to obtain \( P(x_i = \nu | c_k) \), it is sufficient to divide the number of samples in \( c_k \) that have the feature \( x_i \) equal to \( \nu \) (denoted by \( n_{i,k} \)) by the total number of samples in category \( c_k \) (denoted by \( n_k \)):

\[ P(x_i = \nu | c_k) = \frac{n_{i,k}}{n_k} \]

If \( x_i \) is continuous, we assume that \( P(x_i | c_k) \) follows the distribution given for that feature. For example, if the feature \( x_i \) in category \( c_k \) follows a Gaussian distribution with parameters \( \mu_{i,k} \) and \( \sigma_{i,k}^2 \), \( P(x_i | c_k) \) is defined as:

\[ P(x_i | c_k) = \frac{1}{\sqrt{2 \pi \sigma_{i,k}^2}} e^{-\frac{(x_i - \mu_{i,k})^2}{2 \sigma_{i,k}^2}} \]

In this project, the distribution of features is given to you, and you just need to find the distribution parameters. Next, you will implement the introduced learning algorithm for both continuous and discrete features.

### Continuous Section

For this section, use the file dogs.csv. This file is actually the dataset X, which includes the sample vectors \( \mathbf{x} \) and their classes. Some features are continuous and one is discrete. Specifically, height and weight follow a Gaussian distribution. Bark_days follow a binomial distribution, and ear_head_ratio follows a uniform distribution.

#### Part One: Learning
In this part, based on the given distribution of each feature, obtain the parameters of each distribution. For this purpose, according to the classification of each feature, find the specific parameter. To simplify the work, you can use the numpy library in Python.

#### Part Two: Classification
Now, write a function that takes the desired features (which can be in the form of an array or list) and, based on the distribution functions obtained in part one and using the Naïve Bayes algorithm, provides the category of the given sample as output.

### Discrete Section

For this section, use the file emails.csv as the dataset. Here, each email text is a sample where each word is a feature. Thus, the number of features in each sample can vary. Each sample can be spam or not.

#### Part One: Learning
In this part, write a function to find a list of all words and then determine the probability of each word being used in spam and non-spam emails. In fact, you have two categories \( c_0 = \text{ham} \) and \( c_1 = \text{spam} \) in this problem, and you need to find the probabilities \( P(x_i | c_0) \) and \( P(x_i | c_1) \) where \( x_i \) is a word. To do this, count the occurrences of word \( x_i \) in spam and non-spam emails and divide by the total number of words in each group.

#### Part Two: Classification
In this part, you are given a string as a sentence, and based on the words used in this sentence, determine whether it is spam or not. (To do this, you need to find the probability that the sentence (or email) is spam.)

