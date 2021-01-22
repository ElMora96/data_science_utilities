import math
import numpy as np

from numpy.random import randint
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft


def zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd
    return np.nan_to_num(res)


def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def _ncc_c(x, y):
    #Normalized Cross-Correlation
    den = np.array(norm(x) * norm(y))
    den[den == 0] = np.Inf

    x_len = len(x)
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    return np.real(cc) / den


def _ncc_c_2dim(x, y):
    """
    Variant of NCCc that operates with 2 dimensional X arrays and 1 dimensional
    y vector
    Returns a 2 dimensional array of normalized fourier transforms
    """
    den = np.array(norm(x, axis=1) * norm(y))
    den[den == 0] = np.Inf
    x_len = x.shape[-1]
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[:,-(x_len-1):], cc[:,:x_len]), axis=1)
    return np.real(cc) / den[:, np.newaxis]


def _ncc_c_3dim(x, y):
    """
    Variant of NCCc that operates with 2 dimensional X arrays and 2 dimensional
    y vector
    Returns a 3 dimensional array of normalized fourier transforms
    """
    den = norm(x, axis=1)[:, None] * norm(y, axis=1)
    den[den == 0] = np.Inf
    x_len = x.shape[-1]
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size))[:, None])
    cc = np.concatenate((cc[:,:,-(x_len-1):], cc[:,:,:x_len]), axis=2)
    return np.real(cc) / den.T[:, :, None]


def _sbd(x, y):
    #Shape-Based Distance
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return dist, yshift


def _extract_shape(idx, x, j, cur_center):

    _a = []
    for i in range(len(idx)):
        if idx[i] == j:
            if cur_center.sum() == 0:
                opt_x = x[i]
            else:
                _, opt_x = _sbd(cur_center, x[i])
            _a.append(opt_x)
    a = np.array(_a)

    if len(a) == 0:
        return np.zeros((1, x.shape[1]))
    columns = a.shape[1]
    y = zscore(a, axis=1, ddof=1)
    s = np.dot(y.transpose(), y)

    p = np.empty((columns, columns))
    p.fill(1.0/columns)
    p = np.eye(columns) - p

    m = np.dot(np.dot(p, s), p)
    _, vec = eigh(m)
    centroid = vec[:, -1]
    finddistance1 = math.sqrt(((a[0] - centroid) ** 2).sum())
    finddistance2 = math.sqrt(((a[0] + centroid) ** 2).sum())

    if finddistance1 >= finddistance2:
        centroid *= -1

    return zscore(centroid, ddof=1)


def _kshape(x, k):
	
    m = x.shape[0]
    idx = randint(0, k, size=m)
    centroids = np.zeros((k, x.shape[1]))
    distances = np.empty((m, k))

    for _ in range(100):
        old_idx = idx
        for j in range(k):
            centroids[j] = _extract_shape(idx, x, j, centroids[j])

        distances = (1 - _ncc_c_3dim(x, centroids).max(axis=2)).T

        idx = distances.argmin(1)
        if np.array_equal(old_idx, idx):
            break

    return idx, centroids


def kshape(x, k):
    idx, centroids = _kshape(np.array(x), k)
    clusters = []
    for i, centroid in enumerate(centroids):
        series = []
        for j, val in enumerate(idx):
            if i == val:
                series.append(j)
        clusters.append((centroid, series))
    return clusters


def sbd(x, y):
    #Return only distances - rounded
    return np.round(_sbd(x,y)[0], 4)