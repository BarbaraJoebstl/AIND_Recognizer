import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            print("failure on {} with {} states".format(self.this_word, num_states))
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_states = self.n_constant
        return self.base_model(best_num_states)


class SelectorBIC(ModelSelector):
    """ select the model with the ***lowest*** Bayesian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    
    BIC = -2 * logL + p * logN

    In the BIC equation, a penalty term penalizes complexity to avoid overfitting, so that it is not necessary to also use cross-validation in the selection process. 

    L: likelihood of the fitted model
    p: number of parameters (initial state prob, transition prob and emission prob or complexity)
    p * log N: creates penalty for bigger models to avoid overfitting
    N: number of data points
    """
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        # compute BIC score for every model and keep the ***lowest*** BIC.
        # If two BIC s are equal take the one, with the lowest complexitiy

        # "free parameters" are parameters learned by the model.
        # m = num_components
        # f = num_features
        # The free transition probability parameters, is the size of the transmat matrix less one row,
        # because they add up to 1 and therefore the final row is deterministic: m*(m-1)
        # The free starting probabilities, which is the size of startprob minus 1 because it adds to 1.0 and last one can be calculated so m-1
        # Number of means, which is m*f
        # Number of covariances which is the size of the covars matrix, which for “diag” is m*f

        # free parameters = m^2 + 2*m*f - 1

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        lowest_bic = float('inf')
        current_bic = float('inf')
        best_model = None
        
        # go through each model and calc
        for num_comps in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(num_comps)
                log_lHood = hmm_model.score(self.X, self.lengths)

                free_params = (num_comps ** 2) + (2 * num_comps * hmm_model.n_features) - 1
                current_bic = (-2 * log_lHood) + (free_params * np.log(hmm_model.n_features))

            except Exception as e:
                #print (e)
                pass 

        if lowest_bic > current_bic:
            lowest_bic = current_bic
            best_model = hmm_model

        return best_model


class SelectorDIC(ModelSelector):
    ''' select ***best*** model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    
    DIC scores the discriminant ability of a training set for one word against competing words. Instead of a penalty term for complexity, it provides a penalty if model liklihoods for non-matching words are too similar to model likelihoods for the correct word in the word set.

    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    or: 
    DIC = log(P(word)) - average(log(P(other words)))

    M: Model
    X(i): currentWord
    X: words (our training data)
    log(P(X(i))): log likelihood for the fitted model of the current word

    The log likelihood for any individual sample or group of samples can also be calculated with the score method.    
    '''

    def calc_log_likelihood_other_words(self, model, other_words):
        return [model[1].score(word[0], word[1]) for word in other_words]


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        other_words = []
        all_models = []
        all_dics = []
        highest_dic = None

        for w in self.words:
            if w is not self.this_word:
                other_words.append(self.hwords[w])
        try:
            for num_states in range(self.min_n_components, self.max_n_components +1):
                hmm_model = self.base_model(num_states)
                log_lHood_word = hmm_model.score(self.X, self.lengths)
                all_models.append((log_lHood_word, hmm_model))
        except Exception as e:
            #print (e)
            pass         

        for i, m in enumerate(all_models):
            log_lHood_word, hmm_model = m
            current_dic = log_lHood_word - np.mean(self.calc_log_likelihood_other_words(m, other_words))
            all_dics.append((current_dic, m[1]))

        if all_dics:
            # find the best dic and return the related model
            return max(all_dics, key = lambda x: x[0])[1]
        else:
            return None 
        

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    return the ***max*** value of the average log Likelihood

    In order to run hmmlearn training using the X,lengths tuples on the new folds, subsets must be combined based on the indices given for the folds. A helper utility has been provided in the asl_utils module named combine_sequences for this purpose.
    '''
           
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        log_LHoods = []
        best_score_cv = float('-inf')
        score_cv_avg = float('-inf')
        best_model = None

        #go through each model and calc
        for num_states in range(self.min_n_components, self.max_n_components +1):
            try:    
                if len(self.sequences) > 2:
                    split_method = KFold();
                    for train_idx, test_idx in split_method.split(self.sequences):
                        #Train
                        self.X, self.lengths = combine_sequences(train_idx, self.sequences) 
                        #Test
                        X_test, length_test = combine_sequences(test_idx, self.sequences) 
                        
                        hmm_model = self.base_model(num_states)
                        log_current_LHood = hmm_model.score(X_test, length_test)
                else:
                    hmm_model = self.base_model(num_states)
                    log_current_LHood = hmm_model.score(self.X, self.lengths)
                    
                log_LHoods.append(log_current_LHood)
                score_cv_avg = np.mean(log_LHoods)

                if score_cv_avg > best_score_cv:
                    best_score_cv = score_cv_avg
                    best_model = hmm_model
               
            except Exception as e:
                #print (e)
                pass
        return best_model

            

