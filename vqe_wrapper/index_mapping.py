def total_parity(n):
    """
    Checks the binary parity of the value `n`. 01001 is even parity, 011001 is odd parity.
    
    :param n: any positive integer 
    :returns: 1 for odd parity, 0 for even parity.
    """
    if n == 0:
        return 0
    N = int(np.ceil(np.log2(n)))
    if 2 ** N == n:
        N += 1
    set_bits = 0
    for i in range(N):
        if n & 2**i:
            set_bits += 1
    
    return set_bits % 2

def parity_representation(n, N):
    """
    Converts from the Fock representation to the parity representation.
    
    :param n: any positive integer
    :param N: number of bits/qubits used in representing the integer `n`
    :returns: the integer representing the parity mapped value
    """
    if n == 0:
        return 0
    mask = 2 ** (N) - 1
    parity_value = 0
    for i in range(N):
        parity_value = parity_value << 1
        if total_parity(n & mask):
            parity_value += 1
        #print(f'{n} = {n:b}: {mask:04b} {n & mask:04b} {parity_value:04b}')
        mask = (mask - 1) >> 1
        
    return parity_value

def fock_representation(n, N):
    """
    Converts from the parity representation to the Fock representation.
    
    :param n: any positive integer
    :param N: number of bits/qubits used in representing the integer `n`
    :returns: the integer representing the Fock mapped value
    """
    mask = 2 ** (N) - 1
    fock_rep = n ^ (n << 1)
    #print(f'{n} = {n:b}: {fock_rep:04b} {mask:04b}')
    return fock_rep & mask


def z2_reduction(n, N):
    """
    Performs so-called Z2-symmetry reduction on a parity representation.
    
    :param n: any positive integer
    :param N: number of bits/qubits used in representing the integer `n`
    :returns: the integer representing the Z2 reduced parity value
    """
    lower_mask = 2 ** (N//2 - 1) - 1
    upper_mask = lower_mask << N//2
    
    z2_reduced = (n & lower_mask) + ((n & upper_mask) >> 1)
    
    #print(f'{n} = {n:0{N}b} : {z2_reduced:0{N-2}b} {lower_mask:0{N}b} {upper_mask:0{N}b}')
    return z2_reduced


def z2_expansion(n, N, n_a, n_b):
    """
    Performs so-called Z2-symmetry expansion on a parity representation.
    
    :param n: any positive integer
    :param N: number of bits/qubits used in representing the integer `n`
    :param n_a: the number of alpha electrons
    :param n_b: the number of beta spin electrons
    :returns: the integer representing the parity value after expansion
    """
    lower_mask = 2 ** (N//2) - 1
    upper_mask = lower_mask << N//2
    
    z2_expanded = (n & lower_mask) + (((n_a) % 2) << N//2) + ((n & upper_mask) << 1) + (((n_a + n_b) % 2) << (N + 1))
    
    #print(f'{n} = {n:0{N}b} : {z2_expanded:0{N+2}b} {lower_mask:0{N+2}b} {upper_mask:0{N+2}b}')
    return z2_expanded
 
# For Fock representations
def num_alpha(n, N):
    """
    Counts the number of alpha electrons in a Fock representation. Assumes that the orbitals are sorted by spin first.
    
    :param n: any positive integer
    :param N: number of bits/qubits used in representing the integer `n`
    :returns: an integer giving the number of alpha electrons
    """
    lower_mask = (2 ** (N//2) - 1)
    masked = (n & lower_mask)
    counter = 0
    indexer = 1
    for i in range(N//2):
        counter += bool(masked & indexer)
        indexer = indexer << 1
        
    return counter
    
# For Fock representations
def num_beta(n, N):
    """
    Counts the number of beta electrons in a Fock representation. Assumes that the orbitals are sorted by spin first.
    
    :param n: any positive integer
    :param N: number of bits/qubits used in representing the integer `n`
    :returns: an integer giving the number of alpha electrons
    """
    upper_mask = (2 ** (N//2) - 1) << N//2
    masked = (n & upper_mask)
    counter = 0
    indexer = 2 ** (N//2)
    for i in range(N//2):
        counter += bool(masked & indexer) # a bool is automatically cast to 1 or 0
        indexer = indexer << 1
        
    return counter


def z2_parity_to_fock(n, N, n_a, n_b):
    """
    Performs so-called Z2-symmetry expansion on a Z2 reduced parity representation and returns a Fock representation.
    
    :param n: any positive integer
    :param N: number of bits/qubits used in representing the integer `n`
    :param n_a: the number of alpha electrons
    :param n_b: the number of beta spin electrons
    :returns: the integer representing the Fock value after expansion and mapping
    """
    parity_rep = z2_expansion(n, N, n_a, n_b)
    fock_rep = fock_representation(parity_rep, N+2)
    return fock_rep