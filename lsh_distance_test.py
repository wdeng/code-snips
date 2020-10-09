import numpy
import math

# LSH signature generation using random projection
def get_signature(user_vector, rand_proj): 
    res = 0
    for p in (rand_proj):
        res = res << 1
        val = numpy.dot(p, user_vector)
        if val >= 0:
            res |= 1
    return res

# get number of '1's in binary
# running time: O(# of '1's)
def nnz(num):
    if num == 0:
        return 0
    res = 1
    num = num & (num-1)
    while num:
        res += 1
        num = num & (num-1)
    return res     

# angular similarity using definitions
# http://en.wikipedia.org/wiki/Cosine_similarity
def angular_similarity(a,b):
    dot_prod = numpy.dot(a,b)
    sum_a = sum(a**2) **.5
    sum_b = sum(b**2) **.5
    cosine = dot_prod/sum_a/sum_b # cosine similarity
    theta = math.acos(cosine)
    return 1.0-(theta/math.pi)

if __name__ == '__main__':
    dim = 200 # number of dimensions per data
    d = 2**10 # number of bits per signature
    
    nruns = 24 # repeat times
    
    avg = 0
    for run in xrange(nruns):
        user1 = numpy.random.randn(dim)
        user2 = numpy.random.randn(dim)
        randv = numpy.random.randn(d, dim)    
        r1 = get_signature(user1, randv)
        r2 = get_signature(user2, randv)
        xor = r1^r2
        true_sim, hash_sim = (angular_similarity(user1, user2), (d-nnz(xor))/float(d))
        diff = abs(hash_sim-true_sim)/true_sim
        avg += diff
        print 'true %.4f, hash %.4f, diff %.4f' % (true_sim, hash_sim, diff) 
    print 'avg diff' , avg / nruns

"""running result:
true 0.5010, hash 0.5098, diff 0.0176
true 0.4936, hash 0.4814, diff 0.0247
true 0.4963, hash 0.4844, diff 0.0240
true 0.5140, hash 0.4883, diff 0.0500
true 0.5266, hash 0.5127, diff 0.0265
true 0.5041, hash 0.5176, diff 0.0268
true 0.5024, hash 0.5107, diff 0.0167
true 0.5265, hash 0.5049, diff 0.0411
true 0.4939, hash 0.4922, diff 0.0035
true 0.4983, hash 0.5107, diff 0.0249
true 0.4838, hash 0.4912, diff 0.0154
true 0.4515, hash 0.4463, diff 0.0117
true 0.4996, hash 0.5176, diff 0.0360
true 0.4942, hash 0.5264, diff 0.0651
true 0.4854, hash 0.5000, diff 0.0302
true 0.4590, hash 0.4609, diff 0.0042
true 0.4742, hash 0.5068, diff 0.0688
true 0.5515, hash 0.5449, diff 0.0120
true 0.4940, hash 0.4873, diff 0.0135
true 0.4924, hash 0.5000, diff 0.0154
true 0.4510, hash 0.4355, diff 0.0342
true 0.4556, hash 0.4492, diff 0.0141
true 0.4789, hash 0.4795, diff 0.0012
true 0.4831, hash 0.4629, diff 0.0419
avg diff 0.0257997833009"""