# markov_chains.py
"""Volume 2: Markov Chains.
<Name> Sophia Rawlings
<Class> Math 345 Section 3
<Date> November 7th 2019
"""

import numpy as np
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        (fill this out)
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        summ = np.sum(A, axis = 0)
        a = np.allclose(summ, np.ones(A.shape[1]))
        if a == False:  #checks column stochastic by checking the allclose result
                raise ValueError("Invalid matrix")
        if states == None:
            states = []
            for i in range(len(A)):
                states.append(i)
            self.labels = states
        else:
            self.labels = states
        self.transmatrix = A
        self.dict = {}
        i = 0
        for x in self.labels:   #creates the dictionary for the label and col
            self.dict[x] = i
            i += 1


    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        col = self.dict[state]
        stateprob = np.random.multinomial(1, self.transmatrix[:,col])
        outcome = np.argmax(stateprob)  #finds the col to know which label to return
        return self.labels[outcome] 

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        all_the_states = []
        state = start
        all_the_states.append(state)
        for x in range(0,N-1):  #uses transition to go between states and tracks it
            state = self.transition(state)
            all_the_states.append(state)
        return all_the_states

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        state = start
        more_states = []
        more_states.append(state)
        while (state != stop):  #goes till it gets to end state
            state = self.transition(state)
            more_states.append(state)
        return more_states   

    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        x = np.random.random(len(self.transmatrix))
        x = x/(np.sum(x))
        xk = np.matmul(self.transmatrix, x)
        k = 0
        while (la.norm(x - xk)) > tol:  #finds the steady state of the matrix
            x = xk
            xk = np.matmul(self.transmatrix, x)
            k += 1
            if k > maxiter:
                raise ValueError("A^k doesn't converge")
        return xk
            
class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        self.filename = filename
        self.labels = []
        wordset = set()
        lines = []
        with open(filename, 'r') as fp:
            line = fp.readline()
            while line:
                lines.append(line.rstrip("\r\n"))
                wordset.update(line.split())
                line = fp.readline()
        self.labels.extend(wordset)
        self.labels.append("$tart")
        self.labels.append("$top")
        self.dict = {}
        k = 0
        for i in self.labels:
            self.dict[i] = k
            k += 1
        self.transmatrix = np.zeros((len(self.labels),len(self.labels)))
        for line in lines:  #creates the transition matrix
            words = line.split()
            words.insert(0, "$tart")
            words.append("$top")
            for i in range(0,(len(words)-1)):   #connects nodes
                word1 = words[i]
                word2 = words[i+1]
                l = self.dict[word1]
                j = self.dict[word2]
                self.transmatrix[j][l] += 1
        self.transmatrix[self.dict["$top"]][self.dict["$top"]] = 1
        self.transmatrix = self.transmatrix/self.transmatrix.sum(axis = 0)       
        

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        sentence = self.path("$tart","$top")
        result = ""
        for i in range(1,len(sentence)-1):  #makes random sentences
            result += sentence[i]
            if i != len(sentence)-2:
                result += " "
        return result
