import numpy as np

import cv2

from orchard_bouman_clust import clustFunc

#This code was heavily inspired by a project by MarcoForte
#This is his github link: github.com/MarcoForte/bayesian-matting

#The orchard_bouman_clust code is directly from his code

#create a gaussian filter in the shape described by shape
def gauss_filter(shape, sigma):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

#creates a window of pixels from the image
def get_window(img, x, y, N):
    h, w, c = img.shape
    arm = N//2
    window = np.zeros((N, N, c))

    xmin = max(0, x-arm)
    xmax = min(w, x+arm+1)
    ymin = max(0, y-arm)
    ymax = min(h, y+arm+1)

    window[arm - (y-ymin):arm + (ymax - y), arm - (x-xmin):arm + (xmax-x)] = img[ymin:ymax, xmin:xmax]
    return window


#this solves for the alpha, foreground value, and background value using the bayesian matting technique
def solve(F_mean, F_covar, B_mean, B_covar, C, C_var, start_alpha, max_i, min_L):

    F_best = np.zeros(3)
    B_best = np.zeros(3)
    a_best = np.zeros(1)
    max_L = -np.inf

    inv_Cvar = 1/C_var**2

    for i in range(F_mean.shape[0]):

        F_mean_i = F_mean[i]
        inv_Fcovar_i = np.linalg.inv(F_covar[i])

        for j in range(B_mean.shape[0]):

            B_mean_j = B_mean[j]

            inv_Bcovar_j = np.linalg.inv(B_covar[j])

            alpha = start_alpha

            i = 0

            prev_L = -1.99e+300

            L = 0

            #this will repeatedly solve for the alpha value for
            #the alpha value and the foreground and background color values
            #for a set number of iterations or till the Log likehood changes
            #very little
            while i < max_i and abs(L - prev_L) > min_L:
                A = np.zeros((6, 6))

                A[:3, :3] = inv_Fcovar_i + (alpha**2) * np.eye(3) * inv_Cvar

                A[:3, 3:] = np.eye(3)*alpha*(1 - alpha) * inv_Cvar

                A[3:, :3] = np.eye(3)*alpha*(1 - alpha) * inv_Cvar

                A[3:, 3:] = inv_Bcovar_j + np.eye(3)*(1 - alpha)**2 * inv_Cvar

                b = np.zeros((6, 1))


                b[:3] = np.atleast_2d((inv_Fcovar_i.dot(F_mean_i) + C*(alpha) * inv_Cvar)).T

                b[3:] = np.atleast_2d((inv_Bcovar_j.dot(B_mean_j) + C*(1 - alpha) * inv_Cvar)).T

                #solve the Ax = b linear equation to get F and B
                #Note: the prev alpha variable is used to get this answer
                X = np.linalg.solve(A, b)


                F = np.maximum(0, np.minimum(1, X[0:3]))

                B = np.maximum(0, np.minimum(1, X[3:6]))

                #get the alpha variable based on this F and B
                alpha = np.maximum(0, np.minimum(1, ((np.atleast_2d(C).T - B).T.dot(F - B))/np.sum((F-B)**2)))[0, 0]

                #calculate the log likelihoods
                L_C = - np.sum((np.atleast_2d(C).T - alpha*F-(1-alpha)*B)**2) * inv_Cvar

                L_F = (- ((F - np.atleast_2d(F_mean_i).T).T.dot(inv_Fcovar_i).dot(F - np.atleast_2d(F_mean_i).T))/ 2)[0, 0]

                L_B = (- ((B - np.atleast_2d(B_mean_j).T).T.dot(inv_Bcovar_j).dot(B - np.atleast_2d(B_mean_j).T))/ 2)[0, 0]

                L = L_C + L_F + L_B

                #update if this log likehood is the new max
                if L > max_L:

                    a_best = alpha
                    max_L = L
                    F_best = F.ravel()
                    B_best = B.ravel()

                prev_L = L
                i += 1
                
    return F_best, B_best, a_best


def bayesian_matte(img, trimap, N, sigma, minN):

    img = np.array(img, dtype = 'float')

    trimap = np.array(trimap, dtype = 'float')

    img /= 255

    trimap /= 255

    h, w, c = img.shape

    gauss_w = gauss_filter((N, N), sigma)

    gauss_w /= np.max(gauss_w)

    #seperate the foreground, background, and unknown regions using the trimap
    
    F_map = (trimap == 1)

    F = img * np.reshape(F_map, (h, w, 1))

    B_map = (trimap == 0)

    B = img * np.reshape(B_map, (h, w, 1))

    unk_map = np.logical_or(F_map, B_map) == False

    alpha = np.zeros(unk_map.shape)

    alpha[F_map] = 1

    alpha[unk_map] = np.nan

    num_pixels = np.sum(unk_map)

    a, b = np.where(unk_map == True)

    visited = np.stack((a, b, np.zeros(a.shape))).T

    #visit all the unknown pixels
    while(sum(visited[:,2]) != num_pixels):

        last_n = sum(visited[:,2])

        for i in range(num_pixels):

            if visited[i, 2] == 1:
                continue

            else:

                y, x = map(int, visited[i,:2])

                a_window = get_window(alpha[:, :, np.newaxis], x, y, N)[:, :, 0]

                F_window = get_window(F, x, y, N)
                F_weights = np.reshape(a_window**2 * gauss_w, -1)
                vals = np.nan_to_num(F_weights) > 0
                F_pix = np.reshape(F_window, (-1, 3))[vals,:]
                F_weights = F_weights[vals]

                B_window = get_window(B, x, y, N)
                B_weights = np.reshape((1 - a_window)**2 * gauss_w, -1)
                vals = np.nan_to_num(B_weights) > 0
                B_pix = np.reshape(B_window, (-1, 3))[vals,:]
                B_weights = B_weights[vals]

                if len(B_weights) < minN or len(F_weights) < minN:
                    continue

                F_mean, F_cov = clustFunc(F_pix, F_weights)

                B_mean, B_cov = clustFunc(B_pix, B_weights)

                start_alpha = np.nanmean(a_window.ravel())

                #calculate the alpha values for these pixels
                F_pred, B_pred, alpha_pred = solve(F_mean, F_cov, B_mean, B_cov, img[y, x], 0.7, start_alpha, 50, 1e-6)

                F[y, x] = F_pred.ravel()
                B[y, x] = B_pred.ravel()
                alpha[y, x] = alpha_pred
                visited[i, 2] = 1

        #increase window size if all pixels in an area has been searched: i.e. need to search for more
        if sum(visited[:,2]) == last_n:
            N += 2
            gauss_w = gauss_filter((N, N), sigma)
            gauss_w /= np.max(gauss_w)

            
    return alpha


