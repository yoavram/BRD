import sys
import numpy as np

ϵ = 1e-10
br_potential = np.array([1, 0, 1])

def best_response_male(profile, t):
    T = profile.shape[1]
    profile = np.concatenate((profile, profile[:, :2]), axis=1)
    M = profile[0, :]
    F = profile[1, :]
    if t == 0: 
        t = T
    ratios = (M[t-1:t+2] + br_potential) / (ϵ + F[t-1:t+2])
    ratios[1] += ϵ
    return ratios.argmin() - 1


def test_best_response_male():
    profile = np.array([[0, 10], [2, 10]])
    # print(profile)
    assert best_response_male(profile, 0) == 0
    assert best_response_male(profile, 1) == -1


def best_response_female(profile, t):
    T = profile.shape[1]
    profile = np.concatenate((profile, profile[:, :2]), axis=1)
    M = profile[0, :]
    F = profile[1, :]
    if t == 0: 
        t = T
    ratios = (M[t-1:t+2]) / (ϵ + F[t-1:t+2] + br_potential)
    ratios[1] += ϵ
    return ratios.argmax() - 1


def test_best_response_female():
    profile = np.array([[3, 10], [4, 11]])
    # print(profile)
    assert best_response_female(profile, 0) != 0
    assert best_response_female(profile, 1) == 0


def random_deviator(profile):
    N, T = profile.shape
    while True:
        sex, t = np.random.randint(0, N), np.random.randint(0, T) 
        if profile[sex, t] > 0: return sex, t
    # TODO more efficient with np.nonzero


def is_nash(profile):
    N, T = profile.shape
    for t in range(T):
        if profile[0, t] > 0 and best_response_male(profile, t): return False
        if profile[1, t] > 0 and best_response_female(profile, t): return False
    return True


def best_response(init_profile, deviator_func, PRINT=False):
    profile = init_profile.copy()
    N, T = profile.shape
    stationary_count = 0
    while stationary_count < 2*T or not is_nash(profile):        
        sex, t = deviator_func(profile)
        best_response_step = (best_response_male, best_response_female)[sex]
        move = best_response_step(profile, t)
        if move != 0: 
            if PRINT:
                print(profile)
                print()
            profile[sex, t] -= 1
            profile[sex, (t + move) % T] += 1
            stationary_count = 0
        else:
            stationary_count += 1
#             if stationary_count > T: print(stationary_count)
    return profile


def test_best_response():
    # this is not a good test, sometimes it gets to:
    # [[8 4 8 0], [8 4 8 0]]
    profile = np.array([[20, 0, 0, 0], [0, 1, 19, 0]])
    profile = best_response(profile, random_deviator)
    assert (profile[:, 1] == 0).all(), profile
    assert (profile[:, 3] == 0).all(), profile
    assert is_nash(profile)


if __name__ == '__main__':
    # testing
    test_best_response_male()
    test_best_response_female()
    # test_best_response()

    # running
    profile = sys.argv[1].split(';')
    profile = [row.split(',') for row in profile]
    profile = [[int(x) for x in row] for row in profile]
    profile = np.array(profile)
    print("Initial:")
    print(profile)
    profile = best_response(profile, random_deviator)
    print("BRD:")
    print(profile)
    print("Nash?", is_nash(profile))
    