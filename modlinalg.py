import itertools
import operator

### Helper Functions ###

def clone_matrix(matrix, new_modulus):
    """Return a deep copy of matrix, after applying new_modulus on any entry."""
    return [[value % new_modulus for value in row] for row in matrix]

def identity_matrix(size):
    """Return the identity matrix of the given size."""
    result = [[0] * size for i in xrange(size)]
    for i in xrange(size):
        result[i][i] = 1
    return result

def egcd(a, b):
    """Return (s,t,d), such that d = gcd(a,b), and s*a + t*b = d."""
    if b == 0:
        return (1, 0, a)
    else:
        (s, t, d) = egcd(b, a % b)
        return (t, s - t * (a / b), d)

def crt_coefs(mods):
    """Return Chinese-Remainder-Theorem coefficents of the gives moduli.

    Given that m_1, ..., m_k are pairwise relatively prime numbers, the function
    return number c_1, ..., c_k, such that the sum a_1 * c_1 + ... + a_k * c_k
    is equivalent to a_1 modulo m_1, to a_2 modulo m_2, etc.
    """
    result = [None] * len(mods)
    prod = reduce(operator.mul, mods)
    for i in xrange(len(mods)):
        (s, t, d) = egcd(prod / mods[i], mods[i])
        result[i] = s * prod / mods[i]
    return result

def k_factorization(n, k):
    """Given that k divides n, return a factorization of n to powers of numbers,
    which are pairwise relatively prime, and each one either divides k, or is
    relatively prime to k.

    The return value is a list of (factor, exponent) tuples.
    """
    factors = [n / k, k]
    done = False
    while not done:
        done = True
        for i in xrange(len(factors)):
            for j in xrange(i):
                (s, t, d) = egcd(factors[i], factors[j])
                if factors[i] != factors[j] and d != 1:
                    if factors[i] == d:
                        factors[j] /= d
                    else:
                        factors[i] /= d
                    factors.append(d)
                    done = False
    factors.sort()
    result = []
    item = factors[0]
    count = 0
    for factor in factors:
        if factor == item:
            count += 1
        else:
            result.append((item, count))
            item = factor
            count = 1
    result.append((item, count))
    return result

class ModularLinearTransformation(object):
    """A modular linear equation system solver.

    Usage Example:
    
    # Define a linear transformation (modulo 6)
    m = MLT(6, [[0,1,3,1], [1,0,4,2], [2,3,3,5]])

    # Apply the transformation to a vector
    m([1,4,3,3])

    # Iterate over the solutions for m([x,y,z,w]) = [4,0,5]
    for solution in m.solutions([4,0,5]):
        print solution
    """
    def __init__(self, modulus, matrix):
        # The modulus is actually kept as base and exponent, since we have
        # special methods for solving equations modulo n**k.
        # While the user is never asked to supply a factorization, such a
        # a factorization may be found and used during future forking.
        # See self._preprocess() for more details.
        self._modulus_base = modulus
        self._modulus_exponent = 1

        self._n_rows = len(matrix)
        self._n_cols = len(matrix[0])
        self._matrix = matrix
        
        # This is a canonical representation of the matrix, created by
        # Gauss elimination in self._preprocess().
        # Note that it's modulo self._modulus_base.
        self._canonical = clone_matrix(matrix, self._modulus_base)

        # This matrix represents the operations commited on the matrice's
        # rows during Gauss elimination. These operations will be applied
        # to input vectors when solving equation systems in the future.
        self._operations = identity_matrix(self._n_rows)

        # These two variables keep the state of the Gauss elimination, so
        # if we fork, our children can inherit those from us, and continue
        # from the point where we stopped.
        self._current_row = 0
        self._current_col = 0

        # _pivot_table keeps a boolean for each column in the elimination:
        # is it a pivot column or not.
        # _index_table keeps an index for each column:
        # for pivot columns, it's the index of the corresponding row;
        # for other columns, it's the index of the corresponding freedom
        # degree.
        self._pivot_table = [None] * self._n_cols
        self._index_table = [None] * self._n_cols

        # This is calculated during elimination.
        self._dim_kernel = 0

        # This flag is set once self._preprocess() finishes.
        # When it's set another flag is defined: self._is_canonical.
        # See self._preprocess() for an explaination about canonical vs.
        # forked transformations.
        self._is_preprocessed = False

    def __call__(self, vector):
        """Apply the linear transformation on a vector of numbers."""
        mod = self._modulus_base ** self._modulus_exponent
        result = [0] * self._n_rows
        for i in xrange(self._n_rows):
            for (a, b) in zip(self._matrix[i], vector):
                result[i] = (result[i] + a * b) % mod
        return result

    def solutions(self, free_vector):
        """Iterate over the solutions of the equation self(x) = free_vector."""
        if not self._is_preprocessed:
            self._preprocess()
        if self._is_canonical:
            for solution in self._canonical_solutions(free_vector):
                yield solution
        else:
            for solution in self._forked_solutions(free_vector):
                yield solution

    def _preprocess(self):
        """Prepare the object, so future calls to self.solutions() will be handled faster."""

        # This function tries to do Gauss Elimination.
        #
        # If at some point it has to invert a number which is a nonzero invertible,
        # it uses this number to factorize the modulus to powers of pairwise relatively
        # prime numbers. Then it solves recursively the system for these moduli, so it
        # can iterate over the solutions for each and use the CRT to combine them.
        # This operation is called "a fork".
        #
        # If the function succeeds to complete the elimination, the system is said to be
        # canonical.
        #
        # Note that the elimination is done modulo self._modulus_base, even though the
        # object solves the system modulo self._modulus_base ** self._modulus_exponent.
        #
        # That's because, given a blackbox that solves an equation systems of the form
        # Ax = b (mod n) (solve for x for a specific linear transformation A, and any b)
        # it's easy to solve Ax = b (mod n**k).
        #
        # The implementation of this reduction can be found in self._canonical_solutions().
        
        m = self._canonical
        i = self._current_row
        j = self._current_col
        n_rows = self._n_rows
        n_cols = self._n_cols
        
        while not self._is_preprocessed:
            for k in xrange(i, n_rows):
                if m[k][j] % self._modulus_base != 0:
                    break
            if m[k][j] % self._modulus_base == 0:
                self._pivot_table[j] = False
                self._index_table[j] = self._dim_kernel
                self._dim_kernel += 1
                j += 1
            else:
                self._row_swap(i, k)
                (a, b, d) = egcd(self._modulus_base, m[i][j])
                if d != 1:
                    mods = k_factorization(self._modulus_base, d)
                    self._children = [self._fork(base, exponent)
                                      for (base, exponent) in mods]
                    self._is_preprocessed = True
                    self._is_canonical = False
                else:
                    self._multiply_row(i, b)
                    for k in xrange(n_rows):
                        if k != i:
                            self._add_to_row(k, i, self._modulus_base - m[k][j])
                    self._pivot_table[j] = True
                    self._index_table[j] = i
                    i += 1
                    j += 1
            if j == n_cols:
                self._is_preprocessed = True
                self._is_canonical = True
                self._current_row = i

    def _base_solutions(self, free_vector):
        """Iterate over the solutions of the equation self(x) = free_vector
        taken modulo self._modulo_base.

        Assumes the MLT is canonical (i.e. it has been preprocessed, and did
        not fork during preprocessing).
        """
        m = self._canonical
        ops = MLT(self._modulus_base, self._operations)
        v = ops(free_vector)
        for i in xrange(self._current_row, self._n_rows):
            if v[i] != 0 and m[i] == [0] * self._n_cols:
                return
        its = [range(self._modulus_base) for i in xrange(self._dim_kernel)]
        for coefs in itertools.product(*its):
            b = v[:]
            for i in xrange(self._n_rows):
                for j in xrange(self._n_cols):
                    if not self._pivot_table[j]:
                        b[i] -= coefs[self._index_table[j]] * m[i][j]
                        b[i] %= self._modulus_base
            solution = [None] * self._n_cols
            for i in xrange(self._n_cols):
                if self._pivot_table[i]:
                    solution[i] = b[self._index_table[i]]
                else:
                    solution[i] = coefs[self._index_table[i]]
            yield solution

    def _canonical_solutions(self, free_vector, modulus_exponent = None):
        """Iterate over the solutions of the equation self(x) = free_vector.

        Assumes the MLT is canonical (i.e. it has been preprocessed, and did
        not fork during preprocessing).
        """

        # This function mostly reduces the problem of finding solutions
        # modulo (self._modulus_base ** self._modulus_exponent), to
        # finding solutions modulo self._modulus_base.
        #
        # Finding solutions modulo self._modulus_base is done by
        # self._base_solutions()

        if modulus_exponent is None:
            modulus_exponent = self._modulus_exponent
        if modulus_exponent == 1:
            for solution in self._base_solutions(free_vector):
                yield solution
        else:
            for base_solution in self._base_solutions(free_vector):
                partial_result = self(base_solution)
                new_free_vector = [None] * self._n_rows
                for i in xrange(self._n_rows):
                    new_free_vector[i] = (free_vector[i] - partial_result[i])
                    new_free_vector[i] /= self._modulus_base
                for upper_solution in self._canonical_solutions(new_free_vector, modulus_exponent - 1):
                    solution = base_solution[:]
                    for j in xrange(self._n_cols):
                        solution[j] += upper_solution[j] * self._modulus_base
                    yield solution
                    
    def _forked_solutions(self, free_vector):
        """Iterate over the solutions of the equation self(x) = free_vector.

        Assumes the MLT is forked (i.e. it has been preprocessed, and forked
        during preprocessing).
        """
        mods = [t._modulus_base ** t._modulus_exponent for t in self._children]
        coefs = crt_coefs(mods)
        its = [t.solutions(free_vector) for t in self._children]
        for comb in itertools.product(*its):
            solution = [0] * self._n_rows
            for i in xrange(self._n_rows):
                for (a, b) in zip(comb, coefs):
                    solution[i] += a[i] * b
                solution[i] %= self._modulus_base ** self._modulus_exponent
            yield solution

    def _row_swap(self, i, j):
        """Swap rows i and j in both self._canonical and self._operations."""
        self._canonical[i], self._canonical[j] = \
                            self._canonical[j], self._canonical[i]
        self._operations[i], self._operations[j] = \
                             self._operations[j], self._operations[i]

    def _multiply_row(self, i, multiplier):
        """Multiply row i by multiplier, both in self._canonical and self._operations."""
        for j in xrange(self._n_cols):
            self._canonical[i][j] *= multiplier
            self._canonical[i][j] %= self._modulus_base
        for j in xrange(self._n_rows):
            self._operations[i][j] *= multiplier
            self._operations[i][j] %= self._modulus_base

    def _add_to_row(self, i, j, multiplier):
        """Add row j, multiplied by multiplier, to row i in both self._canonical
        and self._operations.
        """
        for k in xrange(self._n_cols):
            self._canonical[i][k] += self._canonical[j][k] * multiplier
            self._canonical[i][k] %= self._modulus_base
        for k in xrange(self._n_rows):
            self._operations[i][k] += self._operations[j][k] * multiplier
            self._operations[i][k] %= self._modulus_base

    def _fork(self, base, exponent):
        """Return a clone of the object with a smaller modulus.

        Note that exponent is automatically multiplied by self._modulus_exponent.
        That means that if you have a transformation with modulus_base = 4 and
        modulus_exponent = 3, and you want to fork it, using the factorization
        4 = 2 * 2, you should call self._fork(2, 2). This will create an object
        with modulus_base = 2 and modulus_exponent = 6.
        """

        # For explaination about the different fields, see self.__init__().
        
        exponent *= self._modulus_exponent
        new_matrix = clone_matrix(self._matrix, base ** exponent)
        child = ModularLinearTransformation(base, new_matrix)
        child._modulus_exponent = exponent
        child._canonical = clone_matrix(self._canonical, base)
        child._operations = clone_matrix(self._operations, base ** exponent)
        child._current_row = self._current_row
        child._current_col = self._current_col
        child._pivot_table = self._pivot_table[:]
        child._index_table = self._index_table[:]
        child._dim_kernel = self._dim_kernel
        child._is_preprocessed = self._is_preprocessed
        if child._is_preprocessed:
            child._is_canonical = self._is_canonical
        return child

MLT = ModularLinearTransformation

__all__ = ["MLT", "ModularLinearTransformation"]
