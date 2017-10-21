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
        # compute BIC score for every model and keep the lowest BIC. If tow BIC s are equal take the one, with the lowest complexitiy

        # get the numer of parameters:
        # Transition Matrix (rows sum up to one):
        # params_tm = State * (State -1)

        # Emission Matrix:
        # params_em = 2 * number_states * number_features

        # parameters = params_tm + params_em

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_num_states, lowest_bic = None, None
        
        # go through each model and calc
        for num_states in range(self.min_n_components, self.max_n_components + 1):
                log_L = self.base_model(num_states, self.max_n_components + 1)
                log_N = np.log(len(self.X))
                p = num_states * (num_states -1) + 2 * len(self.X[0] *num_states)    
                bic_current = -2 * log_L + p * log_N   
                if lowest_bic > bic_current:
                    lowest_bic, best_num_states = bic_current, num_states
        
        if best_num_states is None:
            return self.n_constant
        else:
            return best_num_states
    
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

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_num_states, best_dic = None, None 
        #go through each model and calc
        for num_states in range(self.min_n_components, self.max_n_components +1):
            log_likelihood = self.base_model(num_states).score(self.X, self.lengths)
            log_other = 0
            words = list(self.wordk.keys())
            words.remove(self.this_word)

            for w in words:
                selector_other = ModelSelector(self.words, self.hwords, self.n_constant, self.min_n_components, self.max_n_components, self.random_state, self.verbose)

                log_other += selector_other.base_model(num_states).score(selector_other, len(selector_other))

            current_dic = log_likelihood - log_other / (len(words) - 1)

            if best_dic < current_dic:
                best_dic, best_num_states = current_dic, num_states

        if best_num_states is None:
            return self.n_constant
        else:
            return best_num_states
        


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    return the ***max*** value of the average log Likelihood

    In order to run hmmlearn training using the X,lengths tuples on the new folds, subsets must be combined based on the indices given for the folds. A helper utility has been provided in the asl_utils module named combine_sequences for this purpose.
    '''
           
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        log_LHoods = []
        current_score_cv = None
        best_score_cv = None
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

                if best_score_cv is None:
                    return hmm_model
                elif score_cv_avg is None:
                    return hmm_model
                elif score_cv_avg > best_score_cv:
                    best_score_cv = current_score_cv
                    best_model = hmm_model
               
            except Exception as e:
                print (e)
                pass
        # return max of list of lists comparing each item by value at index 0
        return best_model

            

