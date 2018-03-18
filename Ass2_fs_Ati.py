import numpy as np

def feature_ss(A, y, gamma):
    fe=np.dot(A.T,A)
    tr=np.dot(A.T,y)
    sd = np.dot(y.T, y)
    entry = np.zeros(fe.shape[0])
    signs = np.zeros(fe.shape[0], dtype=np.int8)
    active_set = set()
    l = np.inf
    ch = 0
    grad = - 2 * tr
    max_grad_zero = np.argmax(np.abs(grad))
    while l > gamma or np.allclose(ch, 0)==False:
        if np.allclose(ch, 0):
            el = np.argmax(np.abs(grad) * (signs == 0))
            if grad[el] > gamma:
                signs[el] = -1.
                entry[el] = 0.
                active_set.add(el)
            elif grad[el] < -gamma:
                signs[el] = 1.
                entry[el] = 0.
                active_set.add(el)
            if len(active_set) == 0:
                break
        indices = np.array(sorted(active_set))
        fen = fe[np.ix_(indices, indices)]
        trn = tr[indices]
        signs_n = signs[indices]
        rhs = trn - gamma * signs_n / 2
        new_entry = np.linalg.solve(np.atleast_2d(fen), rhs)
        new_signs = np.sign(new_entry)
        old_entr = entry[indices]
        sign_flips = np.where(abs(new_signs - signs_n) > 1)[0]
        if len(sign_flips) > 0:
            l_n = np.inf
            cu_n = new_entry
            l_n = (sd + (np.dot(new_entry,np.dot(fen, new_entry))- 2 *np.dot(new_entry, trn)) + gamma * abs(new_entry).sum())
            for j in sign_flips:
                a = new_entry[j]
                b = old_entr[j]
                prop = b / (b - a)
                curr = old_entr - prop * (old_entr - new_entry)
                cost = sd + (np.dot(curr, np.dot(fen, curr))
                              - 2 * np.dot(curr, trn)
                              + gamma * abs(curr).sum())
                if cost < l_n:
                    l_n = cost
                    best_prop = prop
                    cu_n = curr
        else:
            cu_n = new_entry;
        entry[indices] = cu_n
        zeros = indices[np.abs(entry[indices]) < 1e-15]
        entry[zeros] = 0.
        signs[indices] = np.int8(np.sign(entry[indices]))
        active_set.difference_update(zeros)
        grad = - 2 * tr + 2 * np.dot(fe, entry)
        l = np.max(abs(grad[signs == 0]))
        ch = np.max(abs(grad[signs != 0] + gamma * signs[signs != 0]))
    return entry

##################################
#Test case
#A=np.random.random((200,1000));
#y=np.random.random((200));
#print feature_ss(A,y,0.1);
