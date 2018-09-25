"""
Adapted from Evan Archer
https://github.com/earcher/vilds/blob/master/code/lib/blk_tridiag_chol_tools.py
"""

import tensorflow as tf
import numpy as np


def blk_tridiag_chol(D, B):
    """
    Compute the Cholesky decomposition of a symmetric, positive definite
    block-tridiagonal matrix.

    Pseudo-code at https://software.intel.com/en-us/node/531896

    Args:
        D (T x n x n tensor): each D[i, :, :] is the ith block diagonal matrix
        B (T-1 x n x n tensor): each B[i, :, :] is the ith (lower) 1st block
            off-diagonal matrix

    Returns:
        chol_blocks (list of tensors)
            chol_blocks[0] (T x n x n tensor): block diagonal elements of
                Cholesky decomposition
            chol_blocks[1] (T-1 x n x n tensor): (lower) 1st block off-diagonal
                elements of Cholesky decomposition

    """

    # Code for computing the Cholesky decomposition of a symmetric block
    # tridiagonal matrix
    def compute_chol(outputs, inputs):

        [Li, Ci] = outputs
        [Di, Bi] = inputs

        # compute the off-diagonal block of the triangular factor
        Ci = tf.matmul(Bi, tf.matrix_inverse(Li), transpose_b=True)
        # update the diagonal block with the newly computed off-diagonal block
        Dii = Di - tf.matmul(Ci, Ci, transpose_b=True)
        # perform Cholesky factorization of a diagonal block
        Li = tf.cholesky(Dii)
        return [Li, Ci]

    # perform Cholesky factorization of the first block
    L1 = tf.cholesky(D[0])
    # initializer for scan function (not used on first iteration)
    C1 = tf.zeros_like(B[0])

    # this scan returns the diagonal and off-diagonal blocks of the Cholesky
    # decomposition
    chol_blocks = tf.scan(
        fn=compute_chol, elems=[D[1:], B], initializer=[L1, C1])

    # add Cholesky factorization of first block to diagonal entries
    chol_blocks[0] = tf.concat(
        [tf.expand_dims(L1, axis=0), chol_blocks[0]],
        axis=0)

    return chol_blocks


def blk_chol_inv(D, B, b, lower=True, transpose=False):
    """
    Solve the equation Cx = b for x, where C is assumed to be a
    block-bidiagonal matrix (where only the first (lower or upper) off-diagonal
    block is nonzero.

    Pseudo-code at https://software.intel.com/en-us/node/531897

    Args:
        D (T x n x n tensor): each D[i, :, :] is the ith block diagonal matrix
        B (T-1 x n x n tensor): each B[i,:,:] is the ith (upper or lower) 1st
            block off-diagonal matrix
        b (T x n tensor)
        lower (bool): treat B as the lower or upper 1st block off-diagonal of
            matrix C
            DEFAULT: True
        transpose (bool): whether to transpose the off-diagonal blocks
            B[i, :, :] (useful if you want to solve the problem C^T x = b
            with a representation of C)
            DEFAULT: False

    Returns:
        X (T x n tensor): solutions of Cx = b

    """

    if transpose:
        D = tf.transpose(D, perm=[0, 2, 1])
        B = tf.transpose(B, perm=[0, 2, 1])

    def tf_dot(A, x):
        return tf.reduce_sum(tf.multiply(A, x), axis=1)

    def update(outputs, inputs):

        [Di, Bi, bi] = inputs
        xi = outputs
        Gi = bi - tf_dot(Bi, xi)

        return tf_dot(tf.matrix_inverse(Di), Gi)

    if lower:

        x0 = tf_dot(tf.matrix_inverse(D[0]), b[0])
        X = tf.scan(
            fn=update, elems=[D[1:], B, b[1:]], initializer=x0)
        X = tf.concat([tf.expand_dims(x0, axis=0), X], axis=0)

    else:

        # computation is the same, just need to reverse the order in which we
        # iterate over the blocks
        xN = tf_dot(tf.matrix_inverse(D[-1]), b[-1])
        X = tf.scan(
            fn=update, elems=[D[:-1][::-1], B[::-1], b[:-1][::-1]],
            initializer=xN)
        # reverse results to put back in correct order
        X = tf.concat([tf.expand_dims(xN, axis=0), X], axis=0)[::-1]

    return X


def blk_chol_inv_multi(D, B, y, lower=True, transpose=False):
    """
    Solve the equation C[x_1 ... x_N] = [y_1 ... y_N] for x_i, where C is
    assumed to be a block-bidiagonal matrix (where only the first (lower or
    upper) off-diagonal block is nonzero.

    Pseudo-code at https://software.intel.com/en-us/node/531897

    Args:
        D (T x K x K tensor): each D[i, :, :] is the ith block diagonal matrix
        B (T-1 x K x K tensor): each B[i,:,:] is the ith (upper or lower) 1st
            block off-diagonal matrix
        y (T x K x N tensor)
        lower (bool): treat B as the lower or upper 1st block off-diagonal of
            matrix C
            DEFAULT: True
        transpose (bool): whether to transpose the off-diagonal blocks
            B[i, :, :] (useful if you want to solve the problem C^T x = y
            with a representation of C)
            DEFAULT: False

    Returns:
        T x K x N tensor: solutions of CX = Y

    """

    if transpose:
        D = tf.transpose(D, perm=[0, 2, 1])
        B = tf.transpose(B, perm=[0, 2, 1])

    def update(outputs, inputs):
        """

        Args:
            outputs (K x S tf.Tensor): RHS of CX=Y for a single time point
            inputs (list of tf.Tensors): Di (K x K), Bi (K x K), yi (K x S):

        Returns:
            xi (K x S): LHS X of CX=Y for a single time point

        """
        [Di, Bi, yi] = inputs
        xi = outputs
        Gi = yi - tf.matmul(Bi, xi)

        return tf.matmul(tf.matrix_inverse(Di), Gi)

    if lower:

        x0 = tf.matmul(tf.matrix_inverse(D[0]), y[0])
        X = tf.scan(
            fn=update, elems=[D[1:], B, y[1:]], initializer=x0)
        X = tf.concat([tf.expand_dims(x0, axis=0), X], axis=0)

    else:

        # computation is the same, just need to reverse the order in which we
        # iterate over the blocks
        xN = tf.matmul(tf.matrix_inverse(D[-1]), y[-1])
        X = tf.scan(
            fn=update, elems=[D[:-1][::-1], B[::-1], y[:-1][::-1]],
            initializer=xN)
        # reverse results to put back in correct order
        X = tf.concat([tf.expand_dims(xN, axis=0), X], axis=0)[::-1]

    return X


if __name__ == '__main__':

    # build a block tridiagonal matrix
    np_a = np.array([[1, 0.2], [0.2, 7]], dtype=np.float32)
    np_b = np.array([[3, 0], [0, 1]], dtype=np.float32)
    np_c = np.array([[2, 0.4], [0.4, 3]], dtype=np.float32)
    np_d = np.array([[3, 0.8], [0.8, 1]], dtype=np.float32)
    np_e = 0.01 * np.array([[2, 7], [1, 4]], dtype=np.float32)
    np_f = 0.01 * np.array([[7, 2], [9, 3]], dtype=np.float32)
    np_g = 0.01 * np.array([[3, 0], [8, 1]], dtype=np.float32)
    np_z = np.array([[0, 0], [0, 0]], dtype=np.float32)

    np_full_mat = np.bmat(
        [[np_a, np_e.T, np_z, np_z],
         [np_e, np_b, np_f.T, np_z],
         [np_z, np_f, np_c, np_g.T],
         [np_z, np_z, np_g, np_d]])

    # build tf tensor with same structure as np_full_mat
    tf_d1 = tf.Variable(np_a)
    tf_d2 = tf.Variable(np_b)
    tf_d3 = tf.Variable(np_c)
    tf_d4 = tf.Variable(np_d)
    tf_ld1 = tf.Variable(np_e)
    tf_ld2 = tf.Variable(np_f)
    tf_ld3 = tf.Variable(np_g)

    tf_diag = tf.stack([tf_d1, tf_d2, tf_d3, tf_d4], axis=0)
    tf_ldiag = tf.stack([tf_ld1, tf_ld2, tf_ld3], axis=0)

    # perform Cholesky decomposition of tf tensor
    tf_chol_blocks = blk_tridiag_chol(tf_diag, tf_ldiag)

    # instantiate tf session and run Cholesky decomposition
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        DChol, LDChol = sess.run(tf_chol_blocks)

    tf_lower_mat = np.bmat(
        [[DChol[0], np_z, np_z, np_z],
         [LDChol[0], DChol[1], np_z, np_z],
         [np_z, LDChol[1], DChol[2], np_z],
         [np_z, np_z, LDChol[2], DChol[3]]])
    tf_full_mat = np.matmul(tf_lower_mat, tf_lower_mat.T)

    print("Successful Cholesky decomposition",
          np.allclose(np_full_mat, tf_full_mat))

    # check to see if inverse is correct by solving Ax = b
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                 dtype=np.float32)
    tf_b = tf.Variable(b)

    temp = blk_chol_inv(tf_chol_blocks[0], tf_chol_blocks[1], tf_b,
                        lower=True, transpose=False)
    soln = blk_chol_inv(tf_chol_blocks[0], tf_chol_blocks[1], temp,
                        lower=False, transpose=True)

    # instantiate tf session and run Cholesky inverse
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_soln = sess.run(soln)

    np_soln = np.linalg.inv(np_full_mat).dot(
        np.array([1, 2, 3, 4, 5, 6, 7, 8]))

    print('Successful Cholesky inversion',
          np.allclose(tf_soln.flatten(), np_soln))
