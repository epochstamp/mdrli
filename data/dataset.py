import sys
import numpy as np
import tree

class DataSet(object):
    """A replay memory consisting of circular buffers for observations, actions, rewards and terminals."""

    def __init__(self, env, random_state=None, max_size=1000000, use_priority=False, only_full_history=True):
        """Initializer.
        Parameters
        -----------
        inputDims : list of tuples
            For each subject i, inputDims[i] is a tuple where the first value is the memory size for this
            subject and the rest describes the shape of each single observation on this subject (number, vector or
            matrix). See base_classes.Environment.inputDimensions() documentation for more info about this format.
        random_state : Numpy random number generator
            If None, a new one is created with default numpy seed.
        max_size : The replay memory maximum size.
        """

        self._batch_dimensions = env.inputDimensions()
        self._max_history_size = np.max([self._batch_dimensions[i][0] for i in range (len(self._batch_dimensions))])
        self._size = max_size
        self._use_priority = use_priority
        self._only_full_history = only_full_history
        if ( isinstance(env.nActions(),int) ):
            self._actions      = CircularBuffer(max_size, dtype="int8")
        else:
            self._actions      = CircularBuffer(max_size, dtype='object')
        self._rewards      = CircularBuffer(max_size)
        self._terminals    = CircularBuffer(max_size, dtype="bool")
        if (self._use_priority):
            self._prioritiy_tree = tree.SumTree(max_size) 
            self._translation_array = np.zeros(max_size)

        self._observations = np.zeros(len(self._batch_dimensions), dtype='object')
        # Initialize the observations container if necessary
        for i in range(len(self._batch_dimensions)):
            self._observations[i] = CircularBuffer(max_size, elemShape=self._batch_dimensions[i][1:], dtype=env.observationType(i))

        if (random_state == None):
            self._random_state = np.random.RandomState()
        else:
            self._random_state = random_state

        self.n_elems  = 0

    def actions(self):
        """Get all actions currently in the replay memory, ordered by time where they were taken."""

        return self._actions.getSlice(0)

    def rewards(self):
        """Get all rewards currently in the replay memory, ordered by time where they were received."""

        return self._rewards.getSlice(0)

    def terminals(self):
        """Get all terminals currently in the replay memory, ordered by time where they were observed.
        
        terminals[i] is True if actions()[i] lead to a terminal state (i.e. corresponded to a terminal 
        transition), and False otherwise.
        """

        return self._terminals.getSlice(0)

    def observations(self):
        """Get all observations currently in the replay memory, ordered by time where they were observed.
        
        observations[s][i] corresponds to the observation made on subject s before the agent took actions()[i].
        """

        ret = np.zeros_like(self._observations)
        for input in range(len(self._observations)):
            ret[input] = self._observations[input].getSlice(0)

        return ret

    def updatePriorities(self, priorities, rndValidIndices):
        """
        """
        for i in range( len(rndValidIndices) ):
            self._prioritiy_tree.update(rndValidIndices[i], priorities[i])

    def randomBatch(self, size, use_priority):
        """Return corresponding states, actions, rewards, terminal status, and next_states for size randomly 
        chosen transitions. Note that if terminal[i] == True, then next_states[s][i] == np.zeros_like(states[s][i]) for 
        each subject s.
        
        Parameters
        -----------
        size : int
            Number of transitions to return.
        Returns
        -------
        states : ndarray
            An ndarray(size=number_of_subjects, dtype='object), where states[s] is a 2+D matrix of dimensions
            size x s.memorySize x "shape of a given observation for this subject". States were taken randomly in
            the data with the only constraint that they are complete regarding the histories for each observed
            subject.
        actions : ndarray
            An ndarray(size=number_of_subjects, dtype='int32') where actions[i] is the action taken after
            having observed states[:][i].
        rewards : ndarray
            An ndarray(size=number_of_subjects, dtype='float32') where rewards[i] is the reward obtained for
            taking actions[i-1].
        next_states : ndarray
            Same structure than states, but next_states[s][i] is guaranteed to be the information
            concerning the state following the one described by states[s][i] for each subject s.
        terminals : ndarray
            An ndarray(size=number_of_subjects, dtype='bool') where terminals[i] is True if actions[i] lead
            to terminal states and False otherwise
        Throws
        -------
            SliceError
                If a batch of this size could not be built based on current data set (not enough data or all
                trajectories are too short).
        """

        if (self._max_history_size - 1 >= self.n_elems):
            raise SliceError(
                "Not enough elements in the dataset to create a "
                "complete state. {} elements in dataset; requires {}"
                .format(self.n_elems, self._max_history_size))

        if (self._use_priority):
            #FIXME : take into account the case where self._only_full_history is false
            rndValidIndices, rndValidIndices_tree = self._randomPrioritizedBatch(size)
            if (rndValidIndices.size == 0):
                raise SliceError("Could not find a state with full histories")
        else:
            rndValidIndices = np.zeros(size, dtype='int32')
            if (self._only_full_history):
                for i in range(size): # TODO: multithread this loop?
                    rndValidIndices[i] = self._randomValidStateIndex(self._max_history_size)
            else:
                for i in range(size): # TODO: multithread this loop?
                    rndValidIndices[i] = self._randomValidStateIndex(minimum_without_terminal=1)
                

        actions   = np.vstack( self._actions.getSliceBySeq(rndValidIndices) )
        rewards   = self._rewards.getSliceBySeq(rndValidIndices)
        terminals = self._terminals.getSliceBySeq(rndValidIndices)
    
        states = np.zeros(len(self._batch_dimensions), dtype='object')
        next_states = np.zeros_like(states)
        # We calculate the first terminal index backward in time and set it 
        # at maximum to the value self._max_history_size
        first_terminals=[]
        for rndValidIndex in rndValidIndices:
            first_terminal=1
            while first_terminal<self._max_history_size:
                if (self._terminals[rndValidIndex-first_terminal]==True or first_terminal>rndValidIndex):
                    break 
                first_terminal+=1
            first_terminals.append(first_terminal)
            
        for input in range(len(self._batch_dimensions)):
            states[input] = np.zeros((size,) + self._batch_dimensions[input], dtype=self._observations[input].dtype)
            next_states[input] = np.zeros_like(states[input])
            for i in range(size):
                slice=self._observations[input].getSlice(rndValidIndices[i]+1-min(self._batch_dimensions[input][0],first_terminals[i]), rndValidIndices[i]+1)
                if (len(slice)==len(states[input][i])):
                    states[input][i] = slice
                else:
                    for j in range(len(slice)):
                        states[input][i][-j-1]=slice[-j-1]
                 # If transition leads to terminal, we don't care about next state
                if rndValidIndices[i] >= self.n_elems - 1 or terminals[i]:
                    next_states[input][i] = np.zeros_like(states[input][i])
                else:
                    slice=self._observations[input].getSlice(rndValidIndices[i]+2-min(self._batch_dimensions[input][0],first_terminals[i]+1), rndValidIndices[i]+2)
                    if (len(slice)==len(states[input][i])):
                        next_states[input][i] = slice
                    else:
                        for j in range(len(slice)):
                            next_states[input][i][-j-1]=slice[-j-1]
                    #next_states[input][i] = self._observations[input].getSlice(rndValidIndices[i]+2-min(self._batch_dimensions[input][0],first_terminal), rndValidIndices[i]+2)
        
        if (self._use_priority):
            return states, actions, rewards, next_states, terminals, [rndValidIndices, rndValidIndices_tree]
        else:
            return states, actions, rewards, next_states, terminals, rndValidIndices

    def _randomValidStateIndex(self, minimum_without_terminal):
        """ Returns the index corresponding to a timestep that is valid
        """
        index_lowerBound = minimum_without_terminal - 1
        # We try out an index in the acceptable range of the replay memory
        index = self._random_state.randint(index_lowerBound, self.n_elems-1) 

        # Check if slice is valid wrt terminals
        # The selected index may correspond to a terminal transition but not 
        # the previous minimum_without_terminal-1 transition
        firstTry = index
        startWrapped = False
        while True:
            i = index-1
            processed = 0
            for _ in range(minimum_without_terminal-1):
                if (i < 0 or self._terminals[i]):
                    break;

                i -= 1
                processed += 1
            if (processed < minimum_without_terminal - 1):
                # if we stopped prematurely, shift slice to the left and try again
                index = i
                if (index < index_lowerBound):
                    startWrapped = True
                    index = self.n_elems - 1
                if (startWrapped and index <= firstTry):
                    raise SliceError("Could not find a state with full histories")
            else:
                # else index was ok according to terminals
                return index
    
    def _randomPrioritizedBatch(self, size):
        indices_tree = self._prioritiy_tree.getBatch(
            size, self._random_state, self)
        indices_replay_mem=np.zeros(indices_tree.size,dtype='int32')
        for i in range(len(indices_tree)):
            indices_replay_mem[i]= int(self._translation_array[indices_tree[i]] \
                         - self._actions.getLowerBound())
        
        return indices_replay_mem, indices_tree

    def addSample(self, obs, action, reward, is_terminal, priority):
        """Store a (observation[for all subjects], action, reward, is_terminal) in the dataset. 
        Parameters
        -----------
        obs : ndarray
            An ndarray(dtype='object') where obs[s] corresponds to the observation made on subject s before the
            agent took action [action].
        action :  int
            The action taken after having observed [obs].
        reward : float
            The reward associated to taking this [action].
        is_terminal : bool
            Tells whether [action] lead to a terminal state (i.e. corresponded to a terminal transition).
        priority : float
            The priority to be associated with the sample
        """        
        # Store observations
        for i in range(len(self._batch_dimensions)):
            self._observations[i].append(obs[i])

        # Update tree and translation table
        if (self._use_priority):
            index = self._actions.getIndex()
            if (index >= self._size):
                ub = self._actions.getUpperBound()
                true_size = self._actions.getTrueSize()
                tree_ind = index%self._size
                if (ub == true_size):
                    size_extension = true_size - self._size
                    # New index
                    index = self._size - 1
                    tree_ind = -1
                    # Shift translation array
                    self._translation_array -= size_extension + 1
                tree_ind = np.where(self._translation_array==tree_ind)[0][0]
            else:
                tree_ind = index

            self._prioritiy_tree.update(tree_ind)
            self._translation_array[tree_ind] = index

        # Store rest of sample
        self._actions.append(action)
        self._rewards.append(reward)
        self._terminals.append(is_terminal)

        if (self.n_elems < self._size):
            self.n_elems += 1

        
class CircularBuffer(object):
    def __init__(self, size, elemShape=(), extension=0.1, dtype="float32"):
        self._size = size
        self._data = np.zeros((int(size+extension*size),) + elemShape, dtype=dtype)
        self._trueSize = self._data.shape[0]
        self._lb   = 0
        self._ub   = size
        self._cur  = 0
        self.dtype = dtype
    
    def append(self, obj):
        if self._cur >= self._size:
            self._lb += 1
            self._ub += 1

        if self._ub > self._trueSize:
            self._data[0:self._size-1] = self._data[self._lb:]
            self._lb  = 0
            self._ub  = self._size
            self._cur = self._size - 1
            
        self._data[self._cur] = obj

        self._cur += 1

    def __getitem__(self, i):
        return self._data[self._lb + i]

    def getSliceBySeq(self, seq):
        return self._data[seq + self._lb]

    def getSlice(self, start, end=sys.maxsize):
        if end == sys.maxsize:
            return self._data[self._lb+start:self._cur]
        else:
            return self._data[self._lb+start:self._lb+end]

    def getLowerBound(self):
        return self._lb

    def getUpperBound(self):
        return self._ub

    def getIndex(self):
        return self._cur

    def getTrueSize(self):
        return self._trueSize


class SliceError(LookupError):
    """Exception raised for errors when getting slices from CircularBuffers.
    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

if __name__ == "__main__":
    pass
