import cv2
import sys

import numpy as np

import math

import heapq

from matte import bayesian_matte

if len(sys.argv) != 2:
    raise ValueError('Please provide an image file to parse.')


#store the color image and the image where are all the channels are averages
    
img1 = cv2.imread(str(sys.argv[1]), cv2.IMREAD_COLOR)

img = np.mean(img1, axis=2).astype(np.uint8)

laplace = cv2.Laplacian(img, cv2.CV_16S)

sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

#This code calculates G the gradient values for the entire image

G = []

for i in range(sobel_x.shape[0]):
    row = []
    for j in range(sobel_x.shape[1]):
        row.append(np.sqrt(sobel_x[i][j]**2 + sobel_y[i][j]**2))
    G.append(np.array(row))

G = np.array(G)

max_g = -100000

for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        max_ind = np.argmax(img[i][j])
        if(G[i][j] > max_g):
            max_g = G[i][j]

def norm(v):
    sum = float(0)
    for i in range(len(v)):
        sum += v[i]**2
    ans = sum**(0.5)
    return ans


def local_cost(p, q):

    """
    Parameters:
    p: (x, y) tuple, indicates source pixel
    q: (x, y) tuple, indicates ending pixel

    This function calculates the local cost between
    pixel p and pixel q using three factors:
    The laplace zero-crossing
    The gradient magnitude
    The gradient direction
    """

    w_z = 0.43
    w_d = 0.43
    w_g = 1

    #max_ind_q = np.argmax(img[q[0]][q[1]])
    #max_ind_p = np.argmax(img[p[0]][p[1]])

    #first get laplace

    f_z = 0 if laplace[q[0]][q[1]] == 0 else 1

    #now for gradient magnitude

    f_g = 1 - G[q[0]][q[1]]/max_g

    if(q[0] == p[0] or q[1] == p[1]):
        f_g *= 1/np.sqrt(2)

    #finally gradient direction

    dp_prime = np.array([sobel_y[p[0]][p[1]], -sobel_x[p[0]][p[1]]])

    dq_prime = np.array([sobel_y[q[0]][q[1]], -sobel_x[q[0]][q[1]]])

    save_prime = dp_prime

    if(norm(dp_prime) != 0):
        dp_prime = dp_prime/norm(dp_prime)

    if(norm(dq_prime) != 0):
       dq_prime = dq_prime/norm(dq_prime)
    
    L = np.array(q) - np.array(p)

    if(dp_prime.dot(L) < 0):
        L = np.array(p) - np.array(q)

    L_save = L
    norm_save = norm(L)

    if(norm_save != 0):
        L = L/norm(L)

    d_p = dp_prime.dot(L)

    d_q = dq_prime.dot(L)
        
    bracket = math.acos(d_p) + math.acos(d_q)
        
    f_d = (1/np.pi)*bracket



    return w_z*f_z + w_g*f_g + w_d*f_d

    
def neighbors(q):
    x = q[0]
    y = q[1]
    return [(x - 1, y + 1), (x, y + 1), (x + 1, y + 1), (x - 1, y), (x + 1, y), (x - 1, y - 1), (x, y - 1), (x + 1, y - 1)]

costs = {}

print("Please wait. Calculating local costs for the image...")

#calculate the costs for img
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for p in neighbors((i, j)):
            if not (p[0] < 0 or p[0] >= img.shape[0] or p[1] < 0 or p[1] >= img.shape[1]):
                costs[((i, j), p)] = local_cost((i, j), p)
    
print("Finished calculating costs.")

def scissors(seed):
    """
    parameters:
    seed: starting pixel state for the scissors
    This is main algorithm that computes all the paths
    from the seed pixel
    """
    
    
    #initialized the e and g tables
    e = []
    g = []
    for i in range(img.shape[0]):
        e_sub = []
        g_sub = []
        for j in range(img.shape[1]):
            e_sub.append(False)
            g_sub.append(11111110.0)
        e.append(e_sub)
        g.append(g_sub)

    #backpointer dictionary. Indicates preceding pixel
    p = {}

    #this is a priority queue that stores pixels, ordered by total cost
    L = []

    
    g[seed[0]][seed[1]] = 0

    
    
    heapq.heappush(L, (g[seed[0]][seed[1]], seed))
    #L.append(seed)

    i = 0
    while L:
        q_all = heapq.heappop(L)
        q = q_all[1]
        e[q[0]][q[1]] = True
        for r in neighbors(q):
            if r[0] < 0 or r[0] >= img.shape[0] or r[1] < 0 or r[1] >= img.shape[1] or e[r[0]][r[1]]:
                continue
            g_temp = g[q[0]][q[1]] + costs[(q, r)]#local_cost(q, r)
            new_cost = min(g_temp, g[r[0]][r[1]])
            if new_cost != g[r[0]][r[1]]:
                g[r[0]][r[1]] = new_cost
                p[r] = q
                heapq.heappush(L, (g[r[0]][r[1]], r))
        i += 1
    return p, L


cimage = img1


copy_cimage = cimage.copy()

seed = (0, 0)
p = {}

w = 1

full_path = []

last_path = []

def draw_path(event, x, y, flags, param):
    """
    This function draws the path used for scissoring
    """
    q = (y//w, x//w)
    global seed, p, cimage, copy_cimage, full_path, last_path
    if event == cv2.EVENT_LBUTTONDBLCLK:
        seed = (y//w, x//w)
        print("Seeding pixel...")
        p, L = scissors(seed)
        print("Pixel has been seeded at point:", seed)
        copy_cimage = cimage.copy()
        full_path.extend(last_path)
    cimage = copy_cimage.copy()
    last_path = []
    while len(p) != 0 and q != seed:
        cimage[q[0], q[1]] = np.array([255, 164, 0])
        last_path.append(q)
        for n in neighbors(q):
            if not (n[0] < 0 or n[0] >= img.shape[0] or n[1] < 0 or n[1] >= img.shape[1]):
                cimage[n[0], n[1]] = np.array([255, 164, 0])
        q = p[q]    

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_path)

def affine_to_tuple(affine):
    return (affine[0], affine[1])


    
def createTrimap(path):
    """
    creates a trimap based on a path:
    two regions are creates by scaling this path
    The in_region is the true foreground
    and the out_regions contains the true_foreground and the unknown
    The end result has the foreground shades as white
    the background shades as black
    and the unknown shaded as grey
    """
    x, y = map(list, zip(*path))

    max_x = max(x)
    min_x = min(x)
    max_y = max(y)
    min_y = min(y)

    origin = ((max_x + min_x)//2, (max_y + min_y)//2)

    num_path = np.array(path)
    num_path = np.concatenate([num_path, np.ones((num_path.shape[0], 1))], axis=1)

    in_scale_factor = 0.9

    out_scale_factor = 1.2
    
    
    trans = np.eye(3)
    trans[0, 2] = -origin[0]
    trans[1, 2] = -origin[1]

    rev = np.eye(3)
    rev[0, 2] = origin[0]
    rev[1, 2] = origin[1]

    scale1 = np.eye(3)
    scale1[0, 0] = out_scale_factor
    scale1[1, 1] = out_scale_factor

    scale2 = np.eye(3)
    scale2[0, 0] = in_scale_factor
    scale2[1, 1] = in_scale_factor

    
    in_path = rev.dot(scale2.dot(trans.dot(num_path.T)))
    in_path = in_path.astype(int)
    
    out_path = rev.dot(scale1.dot(trans.dot(num_path.T)))
    out_path = out_path.astype(int)
    
    trimap = np.ones(img.shape)*(255//2)



    new_in = []
    new_out = []
    
    for aff in in_path.T:
        new_in.append(affine_to_tuple(aff))

    for aff in out_path.T:
        new_out.append(affine_to_tuple(aff))

    for tup in new_in:
        if not (tup[0] < 0 or tup[0] >= img.shape[0] or tup[1] < 0 or tup[1] >= img.shape[1]):
            trimap[tup[0], tup[1]] = 255
        

    for tup in new_out:
        if not (tup[0] < 0 or tup[0] >= img.shape[0] or tup[1] < 0 or tup[1] >= img.shape[1]):
            trimap[tup[0], tup[1]] = 0
        for n in neighbors(tup):
            if not (n[0] < 0 or n[0] >= img.shape[0] or n[1] < 0 or n[1] >= img.shape[1]):
                trimap[n[0], n[1]] = 0

    h, w = img.shape
            
    mask = np.zeros((h+2, w+2), np.uint8)

    img_copy = img.copy()


    trimap = trimap.astype(np.uint8)
    
    cv2.floodFill(trimap, mask, origin, 255)

    cv2.floodFill(trimap, mask, (0, 0), 0)

    cv2.floodFill(trimap, mask, (0, trimap.shape[1] - 1), 0)

    cv2.floodFill(trimap, mask, (trimap.shape[0] - 1, 0), 0)

    cv2.floodFill(trimap, mask, (trimap.shape[0] - 1, trimap.shape[1] - 1), 0)


                
    cv2.imwrite('trimap.jpg', trimap)
    

    return trimap

def create_composite(img_name, shift, alpha_matte):
    """
    create a composite image using the composite equation
    img_name: name of the background image
    shift: the shift of this image on the background image
    alpha_matte: the alpha_matte of img
    """
    background_img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    shift_x = 0
    shift_y = 0
    alpha_matte /= 255
    if shift != "":
        shift_lst = shift.split()
        map(int, shift_lst)
    
        shift_x = shift_lst[0]
        shift_y = shift_lst[1]

    new_img = background_img.copy()

    foreground_img = img1.copy()


    for i in range(background_img.shape[0]):
        for j in range(background_img.shape[1]):
            if i - shift_x < foreground_img.shape[0] and i - shift_x >= 0 and j - shift_y < foreground_img.shape[1] and j - shift_y >= 0:
                alpha = alpha_matte[i - shift_x, j - shift_y]
                F = foreground_img[i - shift_x, j - shift_y]
                B = background_img[i, j]
                new_img[i, j] = alpha*F + (1 - alpha)*B

    cv2.imwrite("composite.jpg", new_img)
    

def detect_cut():
    """
    detects a cut
    """
    # a cycle is found, so you can cut it
    if len(full_path) != len(set(full_path)):
        print("Cut detected")
        trimap = createTrimap(full_path)
        alpha_matte = bayesian_matte(img1, trimap, 25, 8, 10)
        alpha_matte = alpha_matte * 255
        cv2.imwrite('alpha_matte.jpg', alpha_matte)
        print("Alpha Matte has been created.")
        value = ""
        while(value != "Y" and value != "N"):
            value = input("Would you like to create a composite image? [Y/N]: ")
        if value == "Y":
            img_name = input("Enter background image filename: ")
            shift = input("Enter x and y shift as two integers separated by a space [Default: (0, 0)]: ")
            create_composite(img_name, shift, alpha_matte)
    else:
        print("error: cycle not found")
    
        
while(1):
    #new_cimage = cv2.resize(cimage, (400, 400))
    new_cimage = cimage
    cv2.imshow('image', new_cimage)
    cv2.resizeWindow('image', 400, 400)
    if (cv2.waitKey(20) & 0xFF == 99):
        detect_cut()
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()